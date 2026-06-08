from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.longbench_eval import LONG_INPUT_SHORT_OUTPUT_TASKS, load_longbench_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LongBench v1 JSONL task files.")
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--output-dir", default="data/longbench")
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        rows = load_longbench_task(task, cache_dir=args.cache_dir)
        output_path = output_dir / f"{task}.jsonl"
        with output_path.open("w", encoding="utf-8") as out:
            for row in rows:
                import json

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print({"task": task, "rows": len(rows), "output": str(output_path)})


if __name__ == "__main__":
    main()
