#!/usr/bin/env python3
"""Plot per-sample block budgets for a LongBench Oracle run.

The default input is the 95% attention-coverage sweep.  Two figures are
written:

* ``budget_by_sample.png``: samples in their original order;
* ``budget_sorted.png``: samples sorted by budget, which makes the spread
  and long tail easy to see.

The variable budget is ``avg_selected_middle_blocks`` by default.  Use
``--field avg_selected_blocks`` if the fixed prefix/suffix blocks should be
included in the y-axis instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TASKS = ("hotpotqa", "musique", "multifieldqa_en")
COLORS = {
    "hotpotqa": "#2563eb",
    "musique": "#dc2626",
    "multifieldqa_en": "#059669",
}


def read_values(path: Path, field: str) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row.get("idx", 0))
    values = []
    for row in rows:
        value = row.get(field)
        if value is None and field == "avg_selected_middle_blocks":
            # Older result files may only contain the total count.
            value = row.get("avg_selected_blocks")
        if value is None:
            raise KeyError(f"{path}: field {field!r} is missing")
        values.append(float(value))
    if not values:
        raise ValueError(f"No samples found in {path}")
    return np.asarray(values, dtype=float)


def draw_panel(ax, task: str, values: np.ndarray, *, sort_values: bool) -> None:
    y = np.sort(values) if sort_values else values
    x = np.arange(len(y))
    color = COLORS.get(task, "#4b5563")
    # A faint line makes the trend visible without hiding individual samples.
    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.35)
    ax.scatter(x, y, s=11, color=color, alpha=0.72, edgecolors="none")
    median = float(np.median(values))
    mean = float(np.mean(values))
    p90 = float(np.percentile(values, 90))
    ax.axhline(median, color="#111827", linestyle="--", linewidth=1,
               label=f"median {median:.1f}")
    ax.axhline(p90, color="#6b7280", linestyle=":", linewidth=1,
               label=f"p90 {p90:.1f}")
    ax.set_title(f"{task}  (n={len(values)})", loc="left")
    ax.set_xlabel("sorted sample index" if sort_values else "sample index")
    ax.set_ylabel("dynamic blocks" if not sort_values else "selected blocks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.01,
        0.02,
        f"mean={mean:.1f}, min={values.min():.1f}, max={values.max():.1f}",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )


def make_figure(data: dict[str, np.ndarray], output: Path, *, sort_values: bool, field: str) -> None:
    fig, axes = plt.subplots(len(data), 1, figsize=(14, 3.6 * len(data)), squeeze=False)
    for ax, (task, values) in zip(axes[:, 0], data.items()):
        draw_panel(ax, task, values, sort_values=sort_values)
    mode = "sorted by budget" if sort_values else "original sample order"
    fig.suptitle(f"Per-sample block budget ({mode}; field={field})", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/longbench_oracle_coverage_sweep/cov095"),
        help="Directory containing <task>.jsonl files.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/longbench_oracle_coverage_sweep"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--field", choices=("avg_selected_middle_blocks", "avg_selected_blocks"),
                        default="avg_selected_middle_blocks")
    parser.add_argument("--no-sorted", action="store_true",
                        help="Only write the original-order figure.")
    args = parser.parse_args()

    data = {
        task: read_values(args.input_dir / f"{task}.jsonl", args.field)
        for task in args.tasks
    }
    make_figure(data, args.output_dir / "budget_by_sample.png",
                sort_values=False, field=args.field)
    if not args.no_sorted:
        make_figure(data, args.output_dir / "budget_sorted.png",
                    sort_values=True, field=args.field)

    print(f"saved {args.output_dir / 'budget_by_sample.png'}")
    if not args.no_sorted:
        print(f"saved {args.output_dir / 'budget_sorted.png'}")
    for task, values in data.items():
        print(
            f"{task}: n={len(values)}, min={values.min():.1f}, "
            f"median={np.median(values):.1f}, mean={values.mean():.1f}, "
            f"p90={np.percentile(values, 90):.1f}, max={values.max():.1f}"
        )


if __name__ == "__main__":
    main()
