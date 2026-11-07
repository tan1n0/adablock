from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOSS_NAMES = ("loss", "budget_loss", "reuse_loss", "cost_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-epoch AdaBlock losses and compare policy architectures."
    )
    parser.add_argument("metrics_jsonl", nargs="+")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No epoch records found in {path}")
    return records


def summarize_values(records: list[dict[str, Any]], split: str, name: str) -> dict[str, Any] | None:
    points = [
        (int(record["epoch"]), float(record[split][name]))
        for record in records
        if record.get(split) and name in record[split]
    ]
    if not points:
        return None
    values = [value for _, value in points]
    best_index = min(range(len(values)), key=values.__getitem__)
    downward_steps = sum(
        current < previous for previous, current in zip(values, values[1:])
    )
    transitions = max(len(values) - 1, 0)
    return {
        "first": values[0],
        "last": values[-1],
        "delta_last_minus_first": values[-1] - values[0],
        "decreased_overall": values[-1] < values[0],
        "best": values[best_index],
        "best_epoch": points[best_index][0],
        "downward_steps": downward_steps,
        "transitions": transitions,
        "downward_step_ratio": downward_steps / max(transitions, 1),
    }


def summarize_run(path: Path) -> dict[str, Any]:
    records = load_records(path)
    output: dict[str, Any] = {
        "metrics_jsonl": str(path),
        "architecture": records[0].get("architecture", "unknown"),
        "epochs": len(records),
        "train": {},
        "validation": {},
    }
    for split in ("train", "validation"):
        for name in LOSS_NAMES:
            summary = summarize_values(records, split, name)
            if summary is not None:
                output[split][name] = summary
    return output


def main() -> None:
    args = parse_args()
    runs = [summarize_run(Path(path)) for path in args.metrics_jsonl]
    comparable = [
        run
        for run in runs
        if run["validation"].get("loss") is not None
    ]
    ranking = sorted(
        (
            {
                "architecture": run["architecture"],
                "metrics_jsonl": run["metrics_jsonl"],
                "best_validation_loss": run["validation"]["loss"]["best"],
                "best_epoch": run["validation"]["loss"]["best_epoch"],
            }
            for run in comparable
        ),
        key=lambda item: item["best_validation_loss"],
    )
    output = {"runs": runs, "validation_loss_ranking": ranking}
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
