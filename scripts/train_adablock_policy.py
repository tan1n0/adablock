from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import (
    AdaBlockPolicy,
    AdaBlockPolicyConfig,
    adablock_policy_loss,
)


class AdaBlockOracleDataset(Dataset):
    def __init__(self, path: str | Path, drop_nonfinite: bool = True) -> None:
        self.rows: list[dict[str, Any]] = []
        self.dropped_nonfinite = 0
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if drop_nonfinite and self._has_nonfinite(row):
                        self.dropped_nonfinite += 1
                        continue
                    self.rows.append(row)

    @staticmethod
    def _has_nonfinite(value: Any) -> bool:
        if isinstance(value, float):
            return not math.isfinite(value)
        if isinstance(value, int) or value is None or isinstance(value, str):
            return False
        if isinstance(value, list):
            return any(AdaBlockOracleDataset._has_nonfinite(item) for item in value)
        if isinstance(value, dict):
            return any(AdaBlockOracleDataset._has_nonfinite(item) for item in value.values())
        return False

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def hidden_size(self) -> int:
        if not self.rows:
            raise ValueError("Training dataset is empty.")
        return len(self.rows[0]["hidden_state"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "hidden_state": torch.tensor(row["hidden_state"], dtype=torch.float32),
            "query_drift": torch.tensor(row["query_drift"], dtype=torch.float32),
            "score_features": torch.tensor(row["score_features"], dtype=torch.float32),
            "prev_feedback": torch.tensor(row.get("prev_feedback", [0.0] * 8), dtype=torch.float32),
            "budget_label": torch.tensor(row["budget_label"], dtype=torch.long),
            "reuse_label": torch.tensor(row["reuse_label"], dtype=torch.float32),
            "category_target": torch.tensor(row["category_target"], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", default="checkpoints/adablock_policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--lambda-reuse", type=float, default=1.0)
    parser.add_argument("--lambda-category", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=0.01)
    parser.add_argument("--keep-nonfinite", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: AdaBlockPolicy, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "budget_acc": 0.0, "reuse_acc": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            hidden_state=batch["hidden_state"],
            query_drift=batch["query_drift"],
            score_features=batch["score_features"],
            prev_feedback=batch["prev_feedback"],
        )
        losses = adablock_policy_loss(
            outputs,
            budget_label=batch["budget_label"],
            reuse_label=batch["reuse_label"],
            category_target=batch["category_target"],
            lambda_budget=args.lambda_budget,
            lambda_reuse=args.lambda_reuse,
            lambda_category=args.lambda_category,
            lambda_cost=args.lambda_cost,
        )
        batch_size = batch["hidden_state"].shape[0]
        totals["loss"] += float(losses["loss"].item()) * batch_size
        totals["budget_acc"] += (
            outputs["budget_logits"].argmax(dim=-1) == batch["budget_label"]
        ).float().sum().item()
        totals["reuse_acc"] += (
            (outputs["reuse_prob"] > 0.5).float() == batch["reuse_label"]
        ).float().sum().item()
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = AdaBlockOracleDataset(args.train_jsonl, drop_nonfinite=not args.keep_nonfinite)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if dataset.dropped_nonfinite:
        print({"event": "dropped_nonfinite_rows", "count": dataset.dropped_nonfinite})

    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )

    model = AdaBlockPolicy(AdaBlockPolicyConfig(hidden_size=dataset.hidden_size)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in progress:
            batch = move_batch(batch, device)
            outputs = model(
                hidden_state=batch["hidden_state"],
                query_drift=batch["query_drift"],
                score_features=batch["score_features"],
                prev_feedback=batch["prev_feedback"],
            )
            losses = adablock_policy_loss(
                outputs,
                budget_label=batch["budget_label"],
                reuse_label=batch["reuse_label"],
                category_target=batch["category_target"],
                lambda_budget=args.lambda_budget,
                lambda_reuse=args.lambda_reuse,
                lambda_category=args.lambda_category,
                lambda_cost=args.lambda_cost,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.set_postfix(loss=f"{losses['loss'].item():.4f}")

        metrics = evaluate(model, val_loader, device, args) if val_loader is not None else {}
        checkpoint = {
            "model": model.state_dict(),
            "config": model.config.__dict__,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics and metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
        elif not metrics:
            torch.save(checkpoint, output_dir / "best.pt")

        print({"epoch": epoch, **metrics})


if __name__ == "__main__":
    main()
