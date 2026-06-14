from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize aligned LongBench method outputs.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-tsv", default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_jsonl_metrics(path: Path) -> dict[str, float]:
    scores: list[float] = []
    selected_blocks: list[float] = []
    selected_tokens: list[float] = []
    reuse_rates: list[float] = []
    used_prompt_tokens: list[float] = []
    prompt_tokens: list[float] = []
    generated_tokens: list[float] = []

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "score" in row:
                scores.append(float(row["score"]))
            if "avg_selected_blocks" in row:
                selected_blocks.append(float(row["avg_selected_blocks"]))
            if "avg_selected_tokens" in row:
                selected_tokens.append(float(row["avg_selected_tokens"]))
            if "reuse_rate" in row:
                reuse_rates.append(float(row["reuse_rate"]))
            if "used_prompt_tokens" in row:
                used_prompt_tokens.append(float(row["used_prompt_tokens"]))
            if "used_tokens" in row:
                used_prompt_tokens.append(float(row["used_tokens"]))
            if "prompt_tokens" in row:
                prompt_tokens.append(float(row["prompt_tokens"]))
            if "generated_tokens" in row:
                generated_tokens.append(float(row["generated_tokens"]))

    metrics = {
        "num_samples": float(len(scores)),
        "score": mean(scores),
    }
    optional = {
        "avg_selected_blocks": selected_blocks,
        "avg_selected_tokens": selected_tokens,
        "avg_reuse_rate": reuse_rates,
        "avg_used_prompt_tokens": used_prompt_tokens,
        "avg_prompt_tokens": prompt_tokens,
        "avg_generated_tokens": generated_tokens,
    }
    for key, values in optional.items():
        if values:
            metrics[key] = mean(values)
    return metrics


def summarize_method(method_dir: Path) -> dict[str, Any]:
    summary_path = method_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}

    task_metrics: dict[str, dict[str, float]] = {}
    for pred_path in sorted(method_dir.glob("*.jsonl")):
        task_metrics[pred_path.stem] = read_jsonl_metrics(pred_path)

    if "average" not in summary and task_metrics:
        scores = [metrics["score"] for metrics in task_metrics.values() if "score" in metrics]
        summary["average"] = mean(scores)

    return {
        "method": method_dir.name,
        "summary": summary,
        "tasks": task_metrics,
    }


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    task_names = sorted({task for row in rows for task in row["tasks"]})
    columns = ["method", "average", *task_names, "avg_selected_tokens", "avg_reuse_rate"]
    with path.open("w", encoding="utf-8") as out:
        out.write("\t".join(columns) + "\n")
        for row in rows:
            task_scores = [
                row["tasks"].get(task, {}).get("score", row["summary"].get(task, ""))
                for task in task_names
            ]
            sparse_tokens = [
                metrics["avg_selected_tokens"]
                for metrics in row["tasks"].values()
                if "avg_selected_tokens" in metrics
            ]
            reuse_rates = [
                metrics["avg_reuse_rate"]
                for metrics in row["tasks"].values()
                if "avg_reuse_rate" in metrics
            ]
            values: list[Any] = [
                row["method"],
                row["summary"].get("average", ""),
                *task_scores,
                mean(sparse_tokens) if sparse_tokens else "",
                mean(reuse_rates) if reuse_rates else "",
            ]
            out.write("\t".join(str(value) for value in values) + "\n")


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    rows = [
        summarize_method(path)
        for path in sorted(result_root.iterdir())
        if path.is_dir()
    ]

    payload = {
        "result_root": str(result_root),
        "methods": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
    if args.output_tsv:
        output_tsv = Path(args.output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(rows, output_tsv)


if __name__ == "__main__":
    main()
