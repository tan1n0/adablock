from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize utility-vs-layer overlap from block ablation JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--topk", type=int, default=None, help="Override comparison top-k. Defaults to record length.")
    return parser.parse_args()


def set_overlap(a: list[int], b: list[int]) -> dict[str, float]:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return {"overlap": 0.0, "precision": 1.0, "recall": 1.0, "jaccard": 1.0}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "overlap": float(intersection),
        "precision": intersection / max(len(set_a), 1),
        "recall": intersection / max(len(set_b), 1),
        "jaccard": intersection / max(union, 1),
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def trim(values: list[int], topk: int | None) -> list[int]:
    if topk is None:
        return values
    return values[:topk]


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    summary: dict[str, dict[str, list[float] | int]] = {}

    for row in rows:
        utility = trim(row.get("utility_top_blocks", []), args.topk)
        comparisons = {
            "score_top_blocks": trim(row.get("score_top_blocks", []), args.topk),
            "attention_top_blocks": trim(row.get("attention_top_blocks", []), args.topk),
        }
        for layer_name, blocks in row.get("layer_attention_top_blocks", {}).items():
            comparisons[f"layer_{layer_name}"] = trim(blocks, args.topk)

        for name, blocks in comparisons.items():
            stats = set_overlap(blocks, utility)
            bucket = summary.setdefault(
                name,
                {
                    "count": 0,
                    "overlap": [],
                    "precision": [],
                    "recall": [],
                    "jaccard": [],
                },
            )
            bucket["count"] += 1
            for metric in ("overlap", "precision", "recall", "jaccard"):
                bucket[metric].append(stats[metric])

    output = {}
    for name, metrics in summary.items():
        output[name] = {
            "count": metrics["count"],
            "avg_overlap": mean(metrics["overlap"]),
            "avg_precision": mean(metrics["precision"]),
            "avg_recall": mean(metrics["recall"]),
            "avg_jaccard": mean(metrics["jaccard"]),
        }

    print(json.dumps({"input": str(path), "summary": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
