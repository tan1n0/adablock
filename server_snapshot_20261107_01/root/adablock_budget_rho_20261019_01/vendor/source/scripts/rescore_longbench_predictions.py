from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.longbench_eval import mean, score_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score saved LongBench prediction JSONL files.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--pred-jsonl", required=True)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.pred_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    scores: list[float] = []
    rescored_rows: list[dict] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = row.get("prediction", "")
            answers = row.get("answers", [])
            score = score_prediction(args.task, prediction, answers)
            row["old_score"] = row.get("score")
            row["score"] = score
            scores.append(score)
            rescored_rows.append(row)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out:
            for row in rescored_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "task": args.task,
        "pred_jsonl": str(input_path),
        "num_samples": len(scores),
        "score": mean(scores),
    }
    if output_path:
        result["output_jsonl"] = str(output_path)
    print(result)


if __name__ == "__main__":
    main()
