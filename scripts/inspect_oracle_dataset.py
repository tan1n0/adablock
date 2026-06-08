from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect AdaBlock oracle JSONL quality.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, int) or value is None or isinstance(value, str):
        return False
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    return False


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    total = 0
    bad_rows = 0
    budget_counts: Counter[int] = Counter()
    required_k_counts: Counter[int] = Counter()
    reuse_counts: Counter[int] = Counter()
    source_docs: Counter[int] = Counter()
    category_sums: list[float] | None = None
    hidden_size: int | None = None
    first_bad: tuple[int, dict[str, Any]] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.max_rows is not None and total >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if has_nonfinite(row):
                bad_rows += 1
                if first_bad is None:
                    first_bad = (line_no, row)
                continue

            budget_counts[int(row["budget_label"])] += 1
            required_k_counts[int(row.get("required_k", -1))] += 1
            reuse_counts[int(row["reuse_label"])] += 1
            source_docs[int(row.get("doc_id", -1))] += 1
            hidden_size = hidden_size or len(row["hidden_state"])
            category = row["category_target"]
            if category_sums is None:
                category_sums = [0.0] * len(category)
            for idx, value in enumerate(category):
                category_sums[idx] += float(value)

    category_mean = None
    valid_rows = total - bad_rows
    if category_sums is not None and valid_rows > 0:
        category_mean = [value / valid_rows for value in category_sums]

    print(
        {
            "path": str(path),
            "total_rows": total,
            "valid_rows": valid_rows,
            "bad_rows": bad_rows,
            "hidden_size": hidden_size,
            "num_docs": len(source_docs),
            "budget_label_counts": dict(sorted(budget_counts.items())),
            "required_k_top20": required_k_counts.most_common(20),
            "reuse_label_counts": dict(sorted(reuse_counts.items())),
            "category_target_mean": category_mean,
            "first_bad_line": first_bad[0] if first_bad else None,
        }
    )


if __name__ == "__main__":
    main()
