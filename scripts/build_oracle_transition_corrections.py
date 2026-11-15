from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_oracle_dataset import (
    SampledAttentionCollector,
    cuda_peak_memory_gib,
    iter_texts,
    load_dtype,
    normalize_device_map,
    reset_cuda_peak_memory,
)
from utils.block_oracle import (
    OracleConfig,
    cosine_block_scores,
    dynamic_oracle_budget_label,
    fixed_and_middle_blocks,
    make_block_ranges,
    reuse_label_from_previous_middle,
    top_blocks,
)


CORRECTION_VERSION = "adjacent_v1"
REUSE_PREVIOUS_SELECTOR = "cosine_topk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lightweight t-1 -> t transition corrections for an existing "
            "sampled Oracle without rewriting hidden states or budget labels."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Packed source text JSONL.")
    parser.add_argument("--oracle-jsonl", required=True, help="Existing gap-stride Oracle JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Lightweight correction sidecar.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--local-window-blocks", type=int, default=32)
    parser.add_argument(
        "--budget-buckets", nargs="+", type=int, default=[0, 4, 8, 16, 32]
    )
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--reuse-coverage", type=float, default=0.9)
    parser.add_argument(
        "--oracle-layer-indices", nargs="+", type=int, default=[12, 20, 27]
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--attention-query-chunk-size", type=int, default=8)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every-doc", action="store_true")
    return parser.parse_args()


def transition_key(row: dict[str, Any]) -> tuple[str, str, int]:
    source = str(row.get("source", "unknown"))
    source_doc_id = str(row.get("source_doc_id", row.get("doc_id", "unknown")))
    return source, source_doc_id, int(row["token_index"])


def reduced_oracle_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "source",
        "source_doc_id",
        "doc_id",
        "token_index",
        "oracle_attention_mode",
        "oracle_attn_implementation",
        "oracle_layer_indices",
        "oracle_sample_stride",
        "budget_label",
        "required_k",
        "budget_bucket_k",
        "allocated_k",
        "budget_saturated",
        "query_drift",
        "reuse_label",
    )
    return {field: row[field] for field in fields if field in row}


def iter_oracle_groups(path: Path) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    current_document: tuple[str, str] | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = reduced_oracle_row(json.loads(line))
            try:
                key = transition_key(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid Oracle key at {path}:{line_no}.") from exc
            document = key[:2]
            if current_document is None:
                current_document = document
            if document != current_document:
                yield current_document, rows
                current_document = document
                rows = []
            rows.append(row)
    if current_document is not None:
        yield current_document, rows


def load_existing_correction_keys(path: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = transition_key(row)
            if row.get("oracle_transition_version") != CORRECTION_VERSION:
                raise ValueError(
                    f"Unsupported existing correction version at {path}:{line_no}: "
                    f"{row.get('oracle_transition_version')!r}."
                )
            if int(row.get("oracle_transition_gap", -1)) != 1:
                raise ValueError(f"Existing correction is not gap=1 at {path}:{line_no}.")
            if row.get("reuse_previous_selector") != REUSE_PREVIOUS_SELECTOR:
                raise ValueError(
                    f"Existing correction selector mismatch at {path}:{line_no}."
                )
            if int(row.get("previous_token_index", -1)) != key[2] - 1:
                raise ValueError(
                    f"Existing correction has a non-adjacent previous token at {path}:{line_no}."
                )
            if key in keys:
                raise ValueError(f"Duplicate correction key at {path}:{line_no}: {key}")
            keys.add(key)
    return keys


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length < 1 or args.block_size < 1:
        raise ValueError("--max-length and --block-size must be positive.")
    if args.attention_query_chunk_size < 1:
        raise ValueError("--attention-query-chunk-size must be positive.")
    if not 0.0 < args.coverage <= 1.0 or not 0.0 < args.reuse_coverage <= 1.0:
        raise ValueError("--coverage and --reuse-coverage must be in (0, 1].")
    if args.budget_buckets != sorted(set(args.budget_buckets)):
        raise ValueError("--budget-buckets must be strictly increasing.")
    if not args.budget_buckets or any(value < 0 for value in args.budget_buckets):
        raise ValueError("--budget-buckets must contain non-negative values.")
    if len(set(args.oracle_layer_indices)) != len(args.oracle_layer_indices):
        raise ValueError("--oracle-layer-indices must not contain duplicates.")
    if not args.oracle_layer_indices:
        raise ValueError("--oracle-layer-indices must not be empty.")
    input_paths = {
        Path(args.input_jsonl).resolve(),
        Path(args.oracle_jsonl).resolve(),
    }
    if Path(args.output_jsonl).resolve() in input_paths:
        raise ValueError("--output-jsonl must differ from both input paths.")


def assert_oracle_metadata(row: dict[str, Any], args: argparse.Namespace) -> None:
    if row.get("oracle_attention_mode") != "sampled_recompute":
        raise ValueError("Transition repair requires a sampled_recompute Oracle.")
    if row.get("oracle_attn_implementation") != args.attn_implementation:
        raise ValueError(
            "Attention backend mismatch: "
            f"Oracle={row.get('oracle_attn_implementation')!r}, "
            f"repair={args.attn_implementation!r}."
        )
    row_layers = [int(value) for value in row.get("oracle_layer_indices", [])]
    if row_layers != args.oracle_layer_indices:
        raise ValueError(
            f"Oracle layer mismatch: Oracle={row_layers}, repair={args.oracle_layer_indices}."
        )


def budget_state(
    block_mass: torch.Tensor,
    current_block: int,
    config: OracleConfig,
) -> dict[str, Any]:
    sink, local, middle = fixed_and_middle_blocks(
        num_blocks=block_mass.numel(),
        sink_blocks=config.sink_blocks,
        local_window_blocks=config.local_window_blocks,
        current_block=current_block,
    )
    fixed = sink | local
    label, required_k = dynamic_oracle_budget_label(
        block_mass=block_mass,
        fixed_blocks=fixed,
        middle_blocks=middle,
        coverage_threshold=config.coverage_threshold,
        budget_buckets=config.budget_buckets,
    )
    bucket_k = int(config.budget_buckets[label])
    allocated_k = min(bucket_k, len(middle))
    return {
        "fixed": fixed,
        "middle": middle,
        "budget_label": label,
        "required_k": required_k,
        "budget_bucket_k": bucket_k,
        "allocated_k": allocated_k,
        "budget_saturated": required_k > allocated_k,
    }


def assert_current_budget_matches(
    oracle_row: dict[str, Any], current_state: dict[str, Any]
) -> None:
    for field in (
        "budget_label",
        "required_k",
        "budget_bucket_k",
        "allocated_k",
        "budget_saturated",
    ):
        if field not in oracle_row:
            raise ValueError(
                f"Existing Oracle lacks {field!r}; rebuild it with the budget-cap-aware builder."
            )
        if oracle_row[field] != current_state[field]:
            raise ValueError(
                f"Current-label mismatch for key={transition_key(oracle_row)}, field={field}: "
                f"Oracle={oracle_row[field]!r}, recomputed={current_state[field]!r}. "
                "Check model, dtype, layers, backend, coverage, and budget arguments."
            )


def main() -> None:
    args = parse_args()
    args.oracle_layer_indices = sorted(args.oracle_layer_indices)
    validate_args(args)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str, int]] = set()
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{output_path} already exists. Remove it or pass --resume."
            )
        existing_keys = load_existing_correction_keys(output_path)
    output_mode = "a" if args.resume and output_path.exists() else "w"

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for an intentional CPU run.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    device_map = normalize_device_map(args.device_map)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if cuda_available else "cpu"))
    model.eval()
    transformer = getattr(model, "model", model)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise TypeError("Transition repair requires a Qwen/Llama-style model with model.layers.")

    config = OracleConfig(
        block_size=args.block_size,
        sink_blocks=(args.sink_tokens + args.block_size - 1) // args.block_size,
        local_window_blocks=args.local_window_blocks,
        budget_buckets=tuple(args.budget_buckets),
        coverage_threshold=args.coverage,
        reuse_coverage_threshold=args.reuse_coverage,
    )
    print(
        {
            "event": "transition_repair_setup",
            "version": CORRECTION_VERSION,
            "input_jsonl": args.input_jsonl,
            "oracle_jsonl": args.oracle_jsonl,
            "output_jsonl": args.output_jsonl,
            "resume_existing_rows": len(existing_keys),
            "model": args.model_name,
            "device_map": getattr(model, "hf_device_map", None),
            "layers": args.oracle_layer_indices,
            "budget_buckets": args.budget_buckets,
            "transition_gap": 1,
            "reuse_previous_selector": REUSE_PREVIOUS_SELECTOR,
        },
        flush=True,
    )

    oracle_groups = iter(iter_oracle_groups(Path(args.oracle_jsonl)))
    text_iter = iter_texts(Path(args.input_jsonl), args.text_field, args.max_docs)
    written_rows = 0
    skipped_rows = 0
    processed_documents = 0
    changed_reuse = 0
    old_reuse_counts: Counter[int] = Counter()
    new_reuse_counts: Counter[int] = Counter()
    drift_sum = 0.0
    drift_max = 0.0

    with output_path.open(output_mode, encoding="utf-8") as output:
        for _, text, source, source_doc_id in tqdm(
            text_iter, desc="repairing transitions", unit="doc"
        ):
            try:
                oracle_document, oracle_rows = next(oracle_groups)
            except StopIteration as exc:
                raise ValueError("Packed input has more documents than the existing Oracle.") from exc
            expected_document = (str(source), str(source_doc_id))
            if oracle_document != expected_document:
                raise ValueError(
                    f"Document order mismatch: packed={expected_document}, Oracle={oracle_document}."
                )
            for row in oracle_rows:
                assert_oracle_metadata(row, args)

            missing_rows = [row for row in oracle_rows if transition_key(row) not in existing_keys]
            if not missing_rows:
                skipped_rows += len(oracle_rows)
                processed_documents += 1
                continue
            if len(missing_rows) != len(oracle_rows):
                expected_missing = oracle_rows[len(oracle_rows) - len(missing_rows) :]
                if [transition_key(row) for row in missing_rows] != [
                    transition_key(row) for row in expected_missing
                ]:
                    raise ValueError(
                        f"Existing corrections for {expected_document} are not a contiguous prefix."
                    )

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            seq_len = int(input_ids.shape[-1])
            anchor_positions = [int(row["token_index"]) for row in oracle_rows]
            if len(anchor_positions) != len(set(anchor_positions)):
                raise ValueError(f"Duplicate anchor positions in {expected_document}.")
            if any(position < 1 or position >= seq_len for position in anchor_positions):
                raise ValueError(
                    f"Oracle anchor outside tokenized sequence for {expected_document}: seq_len={seq_len}."
                )
            attention_positions = sorted(
                {position for anchor in anchor_positions for position in (anchor - 1, anchor)}
            )
            reset_cuda_peak_memory()
            collector = SampledAttentionCollector(
                layers=layers,
                layer_indices=args.oracle_layer_indices,
                sample_positions=attention_positions,
                seq_len=seq_len,
                block_size=args.block_size,
                query_chunk_size=args.attention_query_chunk_size,
            )
            try:
                with torch.no_grad():
                    outputs = transformer(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=False,
                        output_hidden_states=False,
                        use_cache=False,
                    )
            except Exception:
                collector.remove()
                raise
            hidden = outputs.last_hidden_state[0].detach().cpu().float()
            sampled_mass = collector.finish()
            del outputs
            if not torch.isfinite(hidden).all() or not torch.isfinite(sampled_mass).all():
                raise FloatingPointError(f"Non-finite transition tensors for {expected_document}.")
            position_to_index = {
                position: index for index, position in enumerate(attention_positions)
            }
            block_ranges = make_block_ranges(seq_len, args.block_size)
            document_output: list[str] = []

            for oracle_row in oracle_rows:
                key = transition_key(oracle_row)
                if key in existing_keys:
                    continue
                current_t = int(oracle_row["token_index"])
                previous_t = current_t - 1
                current_block = current_t // args.block_size
                previous_block = previous_t // args.block_size
                current_mass = sampled_mass[
                    position_to_index[current_t], : current_block + 1
                ].clone()
                previous_mass = sampled_mass[
                    position_to_index[previous_t], : previous_block + 1
                ].clone()
                current_state = budget_state(current_mass, current_block, config)
                previous_state = budget_state(previous_mass, previous_block, config)
                assert_current_budget_matches(oracle_row, current_state)

                previous_ranges = block_ranges[: previous_block + 1]
                previous_scores = cosine_block_scores(hidden, previous_ranges, previous_t)
                previous_middle_selection = top_blocks(
                    previous_scores,
                    previous_state["allocated_k"],
                    candidates=previous_state["middle"],
                )
                corrected_reuse = reuse_label_from_previous_middle(
                    block_mass=current_mass,
                    fixed_blocks=current_state["fixed"],
                    previous_middle_blocks=previous_middle_selection,
                    reuse_coverage_threshold=config.reuse_coverage_threshold,
                )
                corrected_drift = float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        hidden[current_t].unsqueeze(0),
                        hidden[previous_t].unsqueeze(0),
                        dim=-1,
                    ).item()
                )
                if not math.isfinite(corrected_drift):
                    raise FloatingPointError(f"Non-finite query drift for key={key}.")
                old_reuse = int(oracle_row["reuse_label"])
                correction = {
                    "source": key[0],
                    "source_doc_id": key[1],
                    "token_index": key[2],
                    "oracle_transition_version": CORRECTION_VERSION,
                    "oracle_anchor_stride": int(oracle_row["oracle_sample_stride"]),
                    "oracle_transition_gap": 1,
                    "previous_token_index": previous_t,
                    "reuse_previous_selector": REUSE_PREVIOUS_SELECTOR,
                    "query_drift": corrected_drift,
                    "reuse_label": corrected_reuse,
                    "previous_budget_label": previous_state["budget_label"],
                    "previous_required_k": previous_state["required_k"],
                    "previous_budget_bucket_k": previous_state["budget_bucket_k"],
                    "previous_allocated_k": previous_state["allocated_k"],
                    "previous_budget_saturated": previous_state["budget_saturated"],
                    "legacy_query_drift": float(oracle_row["query_drift"]),
                    "legacy_reuse_label": old_reuse,
                }
                document_output.append(json.dumps(correction, allow_nan=False) + "\n")
                written_rows += 1
                changed_reuse += corrected_reuse != old_reuse
                old_reuse_counts[old_reuse] += 1
                new_reuse_counts[corrected_reuse] += 1
                drift_sum += corrected_drift
                drift_max = max(drift_max, corrected_drift)

            output.write("".join(document_output))
            output.flush()
            processed_documents += 1
            if args.log_every_doc:
                print(
                    {
                        "event": "transition_doc_done",
                        "document": expected_document,
                        "seq_len": seq_len,
                        "anchors": len(anchor_positions),
                        "attention_queries": len(attention_positions),
                        "written_rows": len(document_output),
                        "cuda_peak_allocated_gib": cuda_peak_memory_gib(),
                    },
                    flush=True,
                )
            del input_ids, attention_mask, hidden, sampled_mass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.max_docs is None:
        try:
            extra_document, _ = next(oracle_groups)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"Existing Oracle has additional document {extra_document} not found in packed input."
            )
    print(
        {
            "event": "transition_repair_complete",
            "documents": processed_documents,
            "written_rows": written_rows,
            "resume_skipped_rows": skipped_rows,
            "changed_reuse_rows": changed_reuse,
            "old_reuse_counts_written": dict(sorted(old_reuse_counts.items())),
            "new_reuse_counts_written": dict(sorted(new_reuse_counts.items())),
            "avg_adjacent_query_drift_written": drift_sum / max(written_rows, 1),
            "max_adjacent_query_drift_written": drift_max,
            "output_jsonl": str(output_path),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
