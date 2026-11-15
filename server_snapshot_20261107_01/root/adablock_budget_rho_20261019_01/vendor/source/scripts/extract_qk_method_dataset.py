#!/usr/bin/env python3
"""Extract one method-specific training JSONL from a multi-target Q/K Oracle.

The multi-target Oracle already contains the labels for all selectors under
``budget_targets``.  This utility avoids another model forward pass and makes
an ordinary single-target JSONL whose top-level fields match the selected
method.  The original Q/K target dictionary can optionally be retained, but
is omitted by default to keep the extracted files smaller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--selector", required=True,
                        choices=("quest_aabb", "cobs"))
    parser.add_argument("--keep-all-targets", action="store_true")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input_jsonl.open(encoding="utf-8") as source, \
            args.output_jsonl.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                target = row["budget_targets"][args.selector]
            except KeyError as exc:
                raise KeyError(
                    f"line {line_number} has no budget_targets[{args.selector!r}]"
                ) from exc

            # Promote the selected method's current and previous transition
            # targets to the fields consumed by the regular policy dataset.
            row["oracle_score_selector"] = target["score_selector"]
            row["oracle_budget_label_selector"] = target["budget_label_selector"]
            row["reuse_previous_selector"] = target["reuse_previous_selector"]
            for name in (
                "budget_label", "required_k", "budget_bucket_k", "allocated_k",
                "budget_saturated", "reuse_label", "reuse_coverage",
                "previous_budget_label", "previous_required_k",
                "previous_budget_bucket_k", "previous_allocated_k",
                "previous_budget_saturated",
            ):
                if name in target:
                    row[name] = target[name]

            if not args.keep_all_targets:
                row.pop("budget_targets", None)
            row["extracted_budget_target_selector"] = args.selector
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"selector={args.selector} rows={written} output={args.output_jsonl}")


if __name__ == "__main__":
    main()
