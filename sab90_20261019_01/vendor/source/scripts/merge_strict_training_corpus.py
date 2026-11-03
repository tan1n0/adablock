from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_strict_training_corpus import (
    LongBenchDecontaminator,
    count_sources,
    length_bucket_counts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independently built strict training source shards."
    )
    parser.add_argument("--source-dir", default="data/strict_sources")
    parser.add_argument("--train-output", default="data/train_strict_clean.jsonl")
    parser.add_argument(
        "--validation-output", default="data/validation_strict_clean.jsonl"
    )
    parser.add_argument(
        "--metadata-output", default="data/train_strict_clean_metadata.json"
    )
    parser.add_argument("--longbench-dir", default="data/longbench")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(paths: list[Path], expected_split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != expected_split:
                    raise ValueError(
                        f"{path} contains split={row.get('split')!r}, expected {expected_split!r}"
                    )
                doc_id = str(row["doc_id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                rows.append(
                    {
                        "doc_id": doc_id,
                        "source": str(row["source"]),
                        "text": str(row["text"]),
                    }
                )
    return rows


def load_source_metadata(source_dir: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*_metadata.json")):
        with path.open("r", encoding="utf-8") as handle:
            metadata.append(json.load(handle))
    return metadata


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    train_files = sorted(source_dir.glob("*_train.jsonl"))
    validation_files = sorted(source_dir.glob("*_validation.jsonl"))
    if not train_files or not validation_files:
        raise FileNotFoundError(f"Missing source shards under {source_dir}")

    train_rows = load_rows(train_files, "train")
    validation_rows = load_rows(validation_files, "validation")
    train_ids = {row["doc_id"] for row in train_rows}
    validation_ids = {row["doc_id"] for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"Train/validation overlap after merge: {len(overlap)}")

    decontaminator = LongBenchDecontaminator(
        data_dir=Path(args.longbench_dir),
        ngram_words=13,
        stride=5,
        min_matches=3,
    )
    contaminated: list[tuple[str, str]] = []
    for row in train_rows + validation_rows:
        if decontaminator.contaminated(row["text"]):
            contaminated.append((row["source"], row["doc_id"]))
            if len(contaminated) >= 20:
                break
    if contaminated:
        raise RuntimeError(
            f"LongBench contamination found after merge; examples={contaminated}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    write_jsonl(train_output, train_rows, "train")
    write_jsonl(validation_output, validation_rows, "validation")

    source_metadata = load_source_metadata(source_dir)
    removed: Counter[str] = Counter()
    for metadata in source_metadata:
        removed.update(metadata.get("decontamination", {}).get("removed", {}))
    output = {
        "seed": args.seed,
        "source_dir": str(source_dir),
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
        "train_output": str(train_output),
        "validation_output": str(validation_output),
        "total_documents": len(train_rows) + len(validation_rows),
        "train_documents": len(train_rows),
        "validation_documents": len(validation_rows),
        "train_source_counts": count_sources(train_rows),
        "validation_source_counts": count_sources(validation_rows),
        "train_length_buckets_chars": length_bucket_counts(train_rows),
        "validation_length_buckets_chars": length_bucket_counts(validation_rows),
        "document_overlap": 0,
        "longbench_ngram_overlap": 0,
        "longbench_ngram_fingerprints": len(decontaminator.hashes),
        "removed_candidates": dict(sorted(removed.items())),
        "source_metadata_files": [
            str(path) for path in sorted(source_dir.glob("*_metadata.json"))
        ],
    }
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
