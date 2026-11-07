from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TABLE2_QWEN25_LOUISKV = {
    "narrativeqa": 19.39,
    "qasper": 37.67,
    "multifieldqa_en": 46.63,
    "hotpotqa": 52.14,
    "musique": 28.51,
    "gov_report": 30.57,
}

PRESET_TASKS = {
    "tiny": ["hotpotqa", "musique"],
    "quick": ["hotpotqa", "musique", "multifieldqa_en"],
    "paper": ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique", "gov_report"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LouisKV-style accuracy proxy and compare with LouisKV Table 2."
    )
    parser.add_argument("--preset", default="quick", choices=sorted(PRESET_TASKS))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-data-dir", default="data/longbench")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens-override", type=int, default=None)
    parser.add_argument("--retrieval-budget-tokens", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--local-window-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--selection-score", default="cosine", choices=["cosine", "quest"])
    parser.add_argument("--refresh-stride", type=int, default=1)
    parser.add_argument("--reuse-similarity-threshold", type=float, default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(printable + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def write_target_table(tasks: list[str], output_path: Path) -> None:
    rows = {
        task: TABLE2_QWEN25_LOUISKV[task]
        for task in tasks
        if task in TABLE2_QWEN25_LOUISKV
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(
            {
                "source": "LouisKV Table 2, Qwen2.5-7B-Instruct, KV cache budget 512",
                "scores": rows,
                "average": sum(rows.values()) / len(rows) if rows else 0.0,
            },
            out,
            indent=2,
            ensure_ascii=False,
        )


def proxy_method_name(args: argparse.Namespace) -> str:
    name = (
        f"louiskv_proxy_{args.selection_score}"
        f"_b{args.retrieval_budget_tokens}"
        f"_sink{args.sink_tokens}"
        f"_local{args.local_window_tokens}"
        f"_refresh{args.refresh_stride}"
    )
    if args.reuse_similarity_threshold is not None:
        name += f"_sim{args.reuse_similarity_threshold:g}"
    return name


def main() -> None:
    args = parse_args()
    tasks = args.tasks or PRESET_TASKS[args.preset]
    result_root = Path(
        args.result_root
        or (
            "results/louiskv_accuracy_proxy_table2_"
            f"{args.preset}_{args.max_input_length}_b{args.retrieval_budget_tokens}"
        )
    )
    log_dir = Path(
        args.log_dir
        or (
            "logs/louiskv_accuracy_proxy_table2_"
            f"{args.preset}_{args.max_input_length}_b{args.retrieval_budget_tokens}"
        )
    )
    method_name = proxy_method_name(args)
    output_dir = result_root / method_name

    if not args.dry_run:
        result_root.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        write_target_table(tasks, result_root / "louiskv_table2_target.json")

    command = [
        sys.executable,
        "scripts/eval_longbench_louiskv_accuracy_proxy.py",
        "--model-name",
        args.model_name,
        "--tasks",
        *tasks,
        "--local-data-dir",
        args.local_data_dir,
        "--output-dir",
        str(output_dir),
        "--max-input-length",
        str(args.max_input_length),
        "--retrieval-budget-tokens",
        str(args.retrieval_budget_tokens),
        "--sink-tokens",
        str(args.sink_tokens),
        "--local-window-tokens",
        str(args.local_window_tokens),
        "--block-size",
        str(args.block_size),
        "--selection-score",
        args.selection_score,
        "--refresh-stride",
        str(args.refresh_stride),
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--device-map",
        args.device_map,
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.max_new_tokens_override is not None:
        command.extend(["--max-new-tokens-override", str(args.max_new_tokens_override)])
    if args.reuse_similarity_threshold is not None:
        command.extend(["--reuse-similarity-threshold", str(args.reuse_similarity_threshold)])

    run_command(command, log_dir / f"{method_name}.log", args.dry_run)

    summary_command = [
        sys.executable,
        "scripts/summarize_aligned_longbench.py",
        "--result-root",
        str(result_root),
        "--target-profile",
        "louiskv-qwen25-512",
        "--output-json",
        str(result_root / "louiskv_accuracy_proxy_summary.json"),
        "--output-tsv",
        str(result_root / "louiskv_accuracy_proxy_summary.tsv"),
    ]
    run_command(summary_command, log_dir / "summarize.log", args.dry_run)

    print()
    print(f"Results root: {result_root}")
    print(f"Summary TSV: {result_root / 'louiskv_accuracy_proxy_summary.tsv'}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
