from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@dataclass
class DocumentCursor:
    source: str
    doc_id: str
    tokens: list[int]
    offset: int = 0


class BalancedBucketSchedule:
    def __init__(
        self,
        buckets: list[int],
        pack_weights: list[int],
        rng: random.Random,
    ) -> None:
        self.buckets = list(buckets)
        self.pack_weights = list(pack_weights)
        self.rng = rng
        self.pending: list[int] = []

    def next(self) -> int:
        if not self.pending:
            self.pending = [
                bucket
                for bucket, weight in zip(self.buckets, self.pack_weights)
                for _ in range(weight)
            ]
            self.rng.shuffle(self.pending)
        return self.pending.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack strict train/validation documents into real token-length buckets "
            "without mixing component documents across splits."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--train-output-jsonl", required=True)
    parser.add_argument("--validation-output-jsonl", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--length-buckets",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--bucket-pack-weights",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Pack-count weights aligned with --length-buckets. Defaults to inverse "
            "length (8:4:2:1 for 4K/8K/16K/32K), which approximately balances "
            "the number of sampled Oracle rows per bucket."
        ),
    )
    parser.add_argument(
        "--separator",
        default="\n\n--- DOCUMENT BOUNDARY ---\n\n",
    )
    parser.add_argument("--min-fill-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--max-validation-docs", type=int, default=None)
    parser.add_argument("--max-packs-per-split", type=int, default=None)
    parser.add_argument("--log-every-packs", type=int, default=25)
    return parser.parse_args()


def load_documents(
    path: Path,
    text_field: str,
    max_docs: int | None,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_docs is not None and len(documents) >= max_docs:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path}:{line_no} has no non-empty {text_field!r} string."
                )
            source = str(row.get("source", "unknown"))
            doc_id = str(row.get("doc_id", row.get("source_doc_id", line_no)))
            documents.append(
                {
                    "source": source,
                    "doc_id": doc_id,
                    "text": text,
                }
            )
    if not documents:
        raise ValueError(f"No documents loaded from {path}.")
    return documents


def document_keys(documents: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys = [(row["source"], row["doc_id"]) for row in documents]
    if len(set(keys)) != len(keys):
        counts: dict[tuple[str, str], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        duplicates = [key for key, count in counts.items() if count > 1]
        raise ValueError(f"Duplicate document ids detected; examples={duplicates[:5]}.")
    return set(keys)


def stable_pack_id(
    split: str,
    target_tokens: int,
    components: list[dict[str, Any]],
) -> str:
    identity = {
        "split": split,
        "target_tokens": target_tokens,
        "components": [
            (
                component["source"],
                component["doc_id"],
                component["start_token"],
                component["end_token"],
            )
            for component in components
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return f"{split}-pack-{target_tokens}-{digest}"


def decode_with_token_cap(tokenizer, token_ids: list[int], target: int) -> tuple[str, int]:
    bounded_ids = list(token_ids[:target])
    text = tokenizer.decode(
        bounded_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    for _ in range(3):
        roundtrip = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(roundtrip) <= target:
            return text, len(roundtrip)
        text = tokenizer.decode(
            roundtrip[:target],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    final_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(final_ids) > target:
        raise RuntimeError(
            f"Could not enforce token cap after decode/encode roundtrip: {len(final_ids)}>{target}."
        )
    return text, len(final_ids)


def pack_split(
    *,
    documents: list[dict[str, Any]],
    split: str,
    output_path: Path,
    tokenizer,
    separator_ids: list[int],
    buckets: list[int],
    bucket_pack_weights: list[int],
    min_fill_ratio: float,
    seed: int,
    max_packs: int | None,
    log_every: int,
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    rng = random.Random(seed)
    ordered_documents = list(documents)
    rng.shuffle(ordered_documents)
    schedule = BalancedBucketSchedule(buckets, bucket_pack_weights, rng)
    document_index = 0
    cursor: DocumentCursor | None = None
    pack_count = 0
    packed_token_sum = 0
    before_roundtrip_sum = 0
    discarded_tail_tokens = 0
    bucket_counts = {str(bucket): 0 for bucket in buckets}
    fill_ratios: list[float] = []
    used_documents: set[tuple[str, str]] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        while document_index < len(ordered_documents) or cursor is not None:
            if max_packs is not None and pack_count >= max_packs:
                break
            target_tokens = schedule.next()
            packed_ids: list[int] = []
            components: list[dict[str, Any]] = []

            while len(packed_ids) < target_tokens:
                if cursor is None:
                    if document_index >= len(ordered_documents):
                        break
                    document = ordered_documents[document_index]
                    document_index += 1
                    tokens = tokenizer(
                        document["text"],
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                    if not tokens:
                        continue
                    cursor = DocumentCursor(
                        source=document["source"],
                        doc_id=document["doc_id"],
                        tokens=tokens,
                    )

                if packed_ids and cursor.offset == 0:
                    remaining = target_tokens - len(packed_ids)
                    if remaining <= len(separator_ids):
                        break
                    packed_ids.extend(separator_ids)

                remaining = target_tokens - len(packed_ids)
                available = len(cursor.tokens) - cursor.offset
                take = min(remaining, available)
                if take <= 0:
                    cursor = None
                    continue
                start_token = cursor.offset
                end_token = start_token + take
                packed_ids.extend(cursor.tokens[start_token:end_token])
                components.append(
                    {
                        "source": cursor.source,
                        "doc_id": cursor.doc_id,
                        "start_token": start_token,
                        "end_token": end_token,
                    }
                )
                cursor.offset = end_token
                if cursor.offset >= len(cursor.tokens):
                    cursor = None

            before_roundtrip_count = len(packed_ids)
            if before_roundtrip_count < target_tokens * min_fill_ratio:
                discarded_tail_tokens += before_roundtrip_count
                break

            packed_text, actual_tokens = decode_with_token_cap(
                tokenizer, packed_ids, target_tokens
            )
            fill_ratio = actual_tokens / target_tokens
            if fill_ratio < min_fill_ratio:
                raise RuntimeError(
                    f"{split} pack roundtrip fill ratio {fill_ratio:.4f} is below "
                    f"--min-fill-ratio={min_fill_ratio}."
                )

            pack_id = stable_pack_id(split, target_tokens, components)
            component_doc_ids = [
                f"{component['source']}::{component['doc_id']}"
                for component in components
            ]
            row = {
                "doc_id": pack_id,
                "source": f"token_pack_{target_tokens}",
                "split": split,
                "text": packed_text,
                "tokenizer_name": tokenizer.name_or_path,
                "target_token_count": target_tokens,
                "actual_token_count": actual_tokens,
                "packed_token_count_before_roundtrip": before_roundtrip_count,
                "component_doc_ids": sorted(set(component_doc_ids)),
                "components": components,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            used_documents.update(
                (component["source"], component["doc_id"])
                for component in components
            )
            pack_count += 1
            packed_token_sum += actual_tokens
            before_roundtrip_sum += before_roundtrip_count
            bucket_counts[str(target_tokens)] += 1
            fill_ratios.append(fill_ratio)

            if log_every > 0 and pack_count % log_every == 0:
                print(
                    {
                        "event": "packing_progress",
                        "split": split,
                        "packs": pack_count,
                        "documents_started": document_index,
                        "documents_total": len(ordered_documents),
                        "packed_tokens": packed_token_sum,
                    },
                    flush=True,
                )

    stats = {
        "split": split,
        "input_documents": len(documents),
        "documents_started": document_index,
        "unique_component_documents": len(used_documents),
        "packs": pack_count,
        "bucket_counts": bucket_counts,
        "bucket_pack_weights": {
            str(bucket): weight
            for bucket, weight in zip(buckets, bucket_pack_weights)
        },
        "packed_tokens": packed_token_sum,
        "packed_tokens_before_roundtrip": before_roundtrip_sum,
        "average_actual_tokens": packed_token_sum / max(pack_count, 1),
        "average_fill_ratio": sum(fill_ratios) / max(len(fill_ratios), 1),
        "minimum_fill_ratio": min(fill_ratios) if fill_ratios else None,
        "discarded_tail_tokens": discarded_tail_tokens,
        "stopped_by_max_packs": max_packs is not None and pack_count >= max_packs,
    }
    print({"event": "packing_complete", **stats}, flush=True)
    return stats, used_documents


def main() -> None:
    args = parse_args()
    if not args.length_buckets or any(bucket < 2 for bucket in args.length_buckets):
        raise ValueError("--length-buckets must contain positive lengths >=2.")
    if len(set(args.length_buckets)) != len(args.length_buckets):
        raise ValueError("--length-buckets must not contain duplicates.")
    if args.bucket_pack_weights is not None:
        if len(args.bucket_pack_weights) != len(args.length_buckets):
            raise ValueError(
                "--bucket-pack-weights must have the same number of values as "
                "--length-buckets."
            )
        if any(weight < 1 for weight in args.bucket_pack_weights):
            raise ValueError("--bucket-pack-weights values must be positive integers.")
        bucket_pairs = sorted(zip(args.length_buckets, args.bucket_pack_weights))
        args.length_buckets = [bucket for bucket, _ in bucket_pairs]
        args.bucket_pack_weights = [weight for _, weight in bucket_pairs]
    else:
        args.length_buckets = sorted(args.length_buckets)
        maximum_bucket = max(args.length_buckets)
        args.bucket_pack_weights = [
            max(1, round(maximum_bucket / bucket))
            for bucket in args.length_buckets
        ]
    if not 0.0 < args.min_fill_ratio <= 1.0:
        raise ValueError("--min-fill-ratio must be in (0, 1].")
    if args.log_every_packs < 0:
        raise ValueError("--log-every-packs must be non-negative.")

    train_documents = load_documents(
        Path(args.train_jsonl), args.text_field, args.max_train_docs
    )
    validation_documents = load_documents(
        Path(args.validation_jsonl), args.text_field, args.max_validation_docs
    )
    train_keys = document_keys(train_documents)
    validation_keys = document_keys(validation_documents)
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(
            f"Train/validation component-document leakage: {len(overlap)} overlaps; "
            f"examples={sorted(overlap)[:5]}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    separator_ids = tokenizer(
        args.separator,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if not separator_ids:
        raise ValueError("--separator tokenizes to an empty sequence.")

    train_stats, train_used = pack_split(
        documents=train_documents,
        split="train",
        output_path=Path(args.train_output_jsonl),
        tokenizer=tokenizer,
        separator_ids=separator_ids,
        buckets=args.length_buckets,
        bucket_pack_weights=args.bucket_pack_weights,
        min_fill_ratio=args.min_fill_ratio,
        seed=args.seed,
        max_packs=args.max_packs_per_split,
        log_every=args.log_every_packs,
    )
    validation_stats, validation_used = pack_split(
        documents=validation_documents,
        split="validation",
        output_path=Path(args.validation_output_jsonl),
        tokenizer=tokenizer,
        separator_ids=separator_ids,
        buckets=args.length_buckets,
        bucket_pack_weights=args.bucket_pack_weights,
        min_fill_ratio=args.min_fill_ratio,
        seed=args.seed + 1,
        max_packs=args.max_packs_per_split,
        log_every=args.log_every_packs,
    )
    used_overlap = train_used & validation_used
    if used_overlap:
        raise RuntimeError(
            f"Packing introduced train/validation leakage: {sorted(used_overlap)[:5]}."
        )

    metadata = {
        "model_name": args.model_name,
        "tokenizer_name": tokenizer.name_or_path,
        "length_buckets": args.length_buckets,
        "bucket_pack_weights": args.bucket_pack_weights,
        "separator": args.separator,
        "separator_token_count": len(separator_ids),
        "min_fill_ratio": args.min_fill_ratio,
        "seed": args.seed,
        "input": {
            "train_jsonl": args.train_jsonl,
            "validation_jsonl": args.validation_jsonl,
        },
        "output": {
            "train_jsonl": args.train_output_jsonl,
            "validation_jsonl": args.validation_output_jsonl,
        },
        "train": train_stats,
        "validation": validation_stats,
        "train_validation_component_overlap": len(used_overlap),
    }
    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, ensure_ascii=False)
    print(
        {
            "event": "token_packing_complete",
            "metadata_json": str(metadata_path),
            "train_validation_component_overlap": len(used_overlap),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
