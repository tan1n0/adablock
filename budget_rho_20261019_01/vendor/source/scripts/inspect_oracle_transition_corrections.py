from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect adjacent-token Oracle transition correction sidecars."
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def transition_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("source", "unknown")),
        str(row.get("source_doc_id", row.get("doc_id", "unknown"))),
        int(row["token_index"]),
    )


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    total = 0
    bad_rows = 0
    duplicate_rows = 0
    changed_reuse = 0
    drift_sum = 0.0
    legacy_drift_sum = 0.0
    drift_min = float("inf")
    drift_max = float("-inf")
    keys: set[tuple[str, str, int]] = set()
    documents: set[tuple[str, str]] = set()
    versions: Counter[str] = Counter()
    gaps: Counter[int] = Counter()
    anchor_strides: Counter[int] = Counter()
    selectors: Counter[str] = Counter()
    reuse_counts: Counter[int] = Counter()
    legacy_reuse_counts: Counter[int] = Counter()
    previous_allocated_counts: Counter[int] = Counter()
    previous_saturated_counts: Counter[bool] = Counter()
    first_bad_line: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.max_rows is not None and total >= args.max_rows:
                break
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                key = transition_key(row)
                drift = float(row["query_drift"])
                legacy_drift = float(row["legacy_query_drift"])
                reuse = int(row["reuse_label"])
                legacy_reuse = int(row["legacy_reuse_label"])
                if not math.isfinite(drift) or not math.isfinite(legacy_drift):
                    raise ValueError("non-finite drift")
                if int(row["previous_token_index"]) != key[2] - 1:
                    raise ValueError("previous_token_index is not token_index - 1")
                if int(row["oracle_transition_gap"]) != 1:
                    raise ValueError("transition gap is not one")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                bad_rows += 1
                if first_bad_line is None:
                    first_bad_line = line_no
                continue

            if key in keys:
                duplicate_rows += 1
            keys.add(key)
            documents.add(key[:2])
            versions[str(row["oracle_transition_version"])] += 1
            gaps[int(row["oracle_transition_gap"])] += 1
            anchor_strides[int(row["oracle_anchor_stride"])] += 1
            selectors[str(row["reuse_previous_selector"])] += 1
            reuse_counts[reuse] += 1
            legacy_reuse_counts[legacy_reuse] += 1
            previous_allocated_counts[int(row["previous_allocated_k"])] += 1
            previous_saturated_counts[bool(row["previous_budget_saturated"])] += 1
            changed_reuse += reuse != legacy_reuse
            drift_sum += drift
            legacy_drift_sum += legacy_drift
            drift_min = min(drift_min, drift)
            drift_max = max(drift_max, drift)

    valid_rows = total - bad_rows
    print(
        {
            "path": str(path),
            "total_rows": total,
            "valid_rows": valid_rows,
            "bad_rows": bad_rows,
            "duplicate_rows": duplicate_rows,
            "num_docs": len(documents),
            "transition_versions": dict(versions),
            "transition_gaps": dict(gaps),
            "anchor_strides": dict(anchor_strides),
            "reuse_previous_selectors": dict(selectors),
            "legacy_reuse_label_counts": dict(sorted(legacy_reuse_counts.items())),
            "corrected_reuse_label_counts": dict(sorted(reuse_counts.items())),
            "changed_reuse_rows": changed_reuse,
            "changed_reuse_rate": changed_reuse / max(valid_rows, 1),
            "avg_legacy_query_drift": legacy_drift_sum / max(valid_rows, 1),
            "avg_adjacent_query_drift": drift_sum / max(valid_rows, 1),
            "min_adjacent_query_drift": drift_min if valid_rows else None,
            "max_adjacent_query_drift": drift_max if valid_rows else None,
            "previous_allocated_k_counts": dict(sorted(previous_allocated_counts.items())),
            "previous_budget_saturated_counts": {
                str(key).lower(): value
                for key, value in sorted(previous_saturated_counts.items())
            },
            "first_bad_line": first_bad_line,
        }
    )


if __name__ == "__main__":
    main()
