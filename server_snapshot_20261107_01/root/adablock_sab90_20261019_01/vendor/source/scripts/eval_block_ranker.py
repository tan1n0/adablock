from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.block_ranker import load_ranker
from utils.ranking_data import RankingDataset, check_metadata
from utils.ranking_eval import evaluate_ranker


def main():
    p = argparse.ArgumentParser(description="Compare ranker/COBS/Quest/mean/oracle at fixed middle-block budgets.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.cpu_threads)
    dataset = RankingDataset(args.data_dir)
    model, checkpoint = load_ranker(args.checkpoint, args.device)
    check_metadata(checkpoint["data_metadata"], dataset.metadata)
    result = evaluate_ranker(model, dataset, args.device, args.budgets, args.coverage, args.max_scenes)
    result.update(checkpoint=args.checkpoint, data_dir=args.data_dir)
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["groups"]["overall"], indent=2))


if __name__ == "__main__":
    main()
