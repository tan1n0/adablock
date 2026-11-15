from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.block_ranker import BlockRanker, BlockRankerConfig, save_ranker, to_device
from utils.ranking_data import RankingDataset, ShardShuffleSampler, assert_disjoint
from utils.ranking_eval import evaluate_ranker
from utils.ranking_loss import PairSamplingConfig, sample_pairs, weighted_rank_loss


def train(args):
    if min(args.epochs, args.batch_size, args.lr) <= 0:
        raise ValueError("epochs, batch_size, lr must be positive.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    train_data, val_data = RankingDataset(args.train_dir), RankingDataset(args.val_dir)
    assert_disjoint(train_data, val_data)
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation data must contain scenes.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new checkpoint directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    meta = train_data.metadata
    config = BlockRankerConfig(meta["head_dim"], meta["num_layers"], meta["num_kv_heads"],
                               projection_size=args.projection_size, hidden_size=args.hidden_size,
                               use_position=not args.no_position, identity_size=args.identity_size,
                               block_chunk_size=args.block_chunk_size)
    sampling = PairSamplingConfig(args.coverage, args.max_positives, args.hard_negatives,
                                   args.random_negatives, args.positive_pairs, args.pair_delta)
    sampling.validate()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BlockRanker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    pair_generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(train_data, batch_size=args.batch_size,
                        sampler=ShardShuffleSampler(train_data, generator), collate_fn=list)
    history, best = [], -float("inf")
    training_meta = {"sampling": asdict(sampling), "seed": args.seed, "lr": args.lr,
                     "budgets": args.eval_budgets, "train_scenes": len(train_data),
                     "validation_scenes": len(val_data), "parameters": sum(p.numel() for p in model.parameters())}
    for epoch in range(args.epochs):
        model.train()
        total_loss = total_pairs = valid_scenes = seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for scene in batch:
                if args.max_train_scenes is not None and seen >= args.max_train_scenes:
                    break
                seen += 1
                sample = to_device(scene, device)
                scores = model(sample)
                pairs = sample_pairs(scores, sample["teacher_mass"], sample["fixed_mass"], sampling, pair_generator)
                if not pairs.numel():
                    continue
                loss = weighted_rank_loss(scores, sample["teacher_mass"], pairs)
                if not bool(loss.isfinite()):
                    raise FloatingPointError("Nonfinite rank loss.")
                losses.append(loss)
                total_pairs += len(pairs)
                total_loss += float(loss.detach())
                valid_scenes += 1
            if losses:
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            if args.max_train_scenes is not None and seen >= args.max_train_scenes:
                break
        if not valid_scenes:
            raise ValueError("No effective ranking pairs. Inspect fixed coverage, labels and pair delta.")
        validation = evaluate_ranker(model, val_data, device, args.eval_budgets, args.coverage, args.max_val_scenes)
        learned = validation["groups"]["overall"]["methods"]["learned"]
        criterion = sum(learned[f"coverage@{k}"] for k in validation["budgets"]) / len(validation["budgets"])
        row = {"epoch": epoch + 1, "train_loss": total_loss / valid_scenes,
               "scenes": seen, "valid_scenes": valid_scenes, "pairs": total_pairs,
               "mean_validation_coverage": criterion, "validation": validation}
        history.append(row)
        save_ranker(output / "last.pt", model, meta, dict(training_meta, epoch=epoch + 1))
        if criterion > best:
            best = criterion
            save_ranker(output / "best.pt", model, meta, dict(training_meta, epoch=epoch + 1))
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "validation"}), flush=True)
    return history


def parse_args():
    p = argparse.ArgumentParser(description="Train current-query COBS ranking without loading the base LLM.")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--projection-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--identity-size", type=int, default=8)
    p.add_argument("--block-chunk-size", type=int, default=256)
    p.add_argument("--no-position", action="store_true")
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--max-positives", type=int, default=16)
    p.add_argument("--hard-negatives", type=int, default=4)
    p.add_argument("--random-negatives", type=int, default=4)
    p.add_argument("--positive-pairs", type=int, default=32)
    p.add_argument("--pair-delta", type=float, default=1e-6)
    p.add_argument("--eval-budgets", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--max-train-scenes", type=int)
    p.add_argument("--max-val-scenes", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--cpu-threads", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
