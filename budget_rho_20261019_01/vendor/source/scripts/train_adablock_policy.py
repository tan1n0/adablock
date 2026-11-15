from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import (
    POLICY_ARCHITECTURES,
    AdaBlockPolicy,
    AdaBlockPolicyConfig,
    adablock_policy_loss,
)
from utils.qk_block_oracle import QK_SCORE_SELECTORS


class AdaBlockOracleDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        drop_nonfinite: bool = True,
        allow_legacy_oracle: bool = False,
        transition_corrections_path: str | Path | None = None,
        allow_nonadjacent_transitions: bool = False,
        budget_target_selector: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.transition_corrections_path = (
            Path(transition_corrections_path) if transition_corrections_path else None
        )
        self.rows: list[dict[str, Any]] = []
        self.dropped_nonfinite = 0
        self.legacy_rows = 0
        self.corrected_rows = 0
        self.nonadjacent_transition_rows = 0
        self.budget_target_selector = budget_target_selector
        self.transition_versions: dict[str, int] = {}
        correction_handle = (
            self.transition_corrections_path.open("r", encoding="utf-8")
            if self.transition_corrections_path
            else None
        )
        correction_iter = self._jsonl_rows(correction_handle) if correction_handle else None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if correction_iter is not None:
                        try:
                            correction = next(correction_iter)
                        except StopIteration as exc:
                            raise ValueError(
                                f"{self.transition_corrections_path} ended before "
                                f"{self.path}:{line_no}."
                            ) from exc
                        self._apply_transition_correction(row, correction, line_no)
                    if drop_nonfinite and self._has_nonfinite(row):
                        self.dropped_nonfinite += 1
                        continue
                    if "eligible_middle_blocks" not in row:
                        self.legacy_rows += 1
                    transition_version = str(
                        row.get("oracle_transition_version", "legacy_or_unknown")
                    )
                    self.transition_versions[transition_version] = (
                        self.transition_versions.get(transition_version, 0) + 1
                    )
                    hidden_transition_valid = (
                        transition_version == "causal_next_token_v1"
                        and int(row.get("oracle_transition_gap", -1)) == 1
                        and row.get("reuse_previous_selector")
                        == "causal_cosine_topk"
                        and bool(row.get("oracle_causal_features", False))
                        and row.get("oracle_policy_input_mode")
                        == "previous_hidden_only"
                        and int(row.get("controller_token_index", -1))
                        == int(row.get("attention_target_token_index", -1)) - 1
                        and int(row.get("score_query_token_index", -1))
                        == int(row.get("controller_token_index", -2))
                        and int(row.get("policy_feature_max_token_index", -1))
                        == int(row.get("controller_token_index", -2))
                    )
                    effective_qk_selector = (
                        self.budget_target_selector
                        or str(row.get("oracle_score_selector", ""))
                    )
                    effective_qk_target = row.get("budget_targets", {}).get(
                        effective_qk_selector, {}
                    )
                    # Method-specific datasets extracted from a multi-target
                    # Oracle promote the selected target to top-level fields
                    # and intentionally remove ``budget_targets``.  Rebuild
                    # the small validation view here so those datasets retain
                    # the same causal-transition checks as the multi-target
                    # representation.
                    if (
                        not effective_qk_target
                        and row.get("oracle_retrieval_space") == "post_rope_qk"
                        and row.get("oracle_score_selector") == effective_qk_selector
                    ):
                        effective_qk_target = {
                            "score_selector": row.get("oracle_score_selector"),
                            "budget_label_selector": row.get(
                                "oracle_budget_label_selector"
                            ),
                            "reuse_previous_selector": row.get(
                                "reuse_previous_selector"
                            ),
                        }
                    qk_transition_valid = (
                        transition_version == "causal_next_token_v1"
                        and row.get("oracle_retrieval_space") == "post_rope_qk"
                        and bool(row.get("oracle_causal_features", False))
                        and row.get("oracle_policy_input_mode") == "previous_hidden_only"
                        and int(row.get("controller_token_index", -1))
                        == int(row.get("attention_target_token_index", -1)) - 1
                        and int(row.get("policy_feature_max_token_index", -1))
                        == int(row.get("controller_token_index", -2))
                        and int(row.get("selector_query_token_index", -1))
                        == int(row.get("attention_target_token_index", -2))
                        and effective_qk_target.get("score_selector")
                        == effective_qk_selector
                        and effective_qk_target.get("budget_label_selector")
                        == (
                            f"post_rope_qk_{effective_qk_selector}"
                            "_topk_attention_coverage"
                        )
                        and effective_qk_target.get("reuse_previous_selector")
                        == f"causal_{effective_qk_selector}_topk"
                    )
                    if not (hidden_transition_valid or qk_transition_valid):
                        self.nonadjacent_transition_rows += 1
                    self.rows.append(row)
            if correction_iter is not None:
                try:
                    extra = next(correction_iter)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        f"{self.transition_corrections_path} has extra row "
                        f"{self._transition_key(extra)} after {self.path} ended."
                    )
        finally:
            if correction_handle is not None:
                correction_handle.close()
        if self.legacy_rows and not allow_legacy_oracle:
            raise ValueError(
                f"{self.path} contains {self.legacy_rows} legacy Oracle rows without "
                "dynamic-middle metadata. Rebuild it with the current "
                "build_oracle_dataset.py, or pass --allow-legacy-oracle intentionally."
            )
        if self.nonadjacent_transition_rows and not allow_nonadjacent_transitions:
            raise ValueError(
                f"{self.path} contains {self.nonadjacent_transition_rows} rows without "
                "causal_next_token_v1 previous-hidden-only features. Rebuild the Oracle with "
                "--transition-mode causal_next_token, or pass "
                "--allow-nonadjacent-transitions only for an intentional legacy ablation."
            )

    @staticmethod
    def _jsonl_rows(handle):
        for line in handle:
            if line.strip():
                yield json.loads(line)

    @staticmethod
    def _transition_key(row: dict[str, Any]) -> tuple[str, str, int]:
        source = str(row.get("source", "unknown"))
        source_doc_id = str(row.get("source_doc_id", row.get("doc_id", "unknown")))
        return source, source_doc_id, int(row["token_index"])

    def _apply_transition_correction(
        self,
        row: dict[str, Any],
        correction: dict[str, Any],
        oracle_line_no: int,
    ) -> None:
        oracle_key = self._transition_key(row)
        correction_key = self._transition_key(correction)
        if oracle_key != correction_key:
            raise ValueError(
                f"Transition correction order/key mismatch at {self.path}:{oracle_line_no}: "
                f"Oracle={oracle_key}, correction={correction_key}."
            )
        if correction.get("oracle_transition_version") != "adjacent_v1":
            raise ValueError(
                f"Unsupported transition correction version for {correction_key}: "
                f"{correction.get('oracle_transition_version')!r}."
            )
        if int(correction.get("oracle_transition_gap", -1)) != 1:
            raise ValueError(f"Correction {correction_key} is not an adjacent-token transition.")
        if correction.get("reuse_previous_selector") != "cosine_topk":
            raise ValueError(
                f"Correction {correction_key} does not use the inference-aligned cosine selector."
            )
        required_fields = (
            "query_drift",
            "reuse_label",
            "previous_token_index",
            "previous_budget_label",
            "previous_required_k",
            "previous_budget_bucket_k",
            "previous_allocated_k",
            "previous_budget_saturated",
        )
        missing = [field for field in required_fields if field not in correction]
        if missing:
            raise ValueError(f"Correction {correction_key} lacks fields: {missing}.")
        for field in required_fields:
            row[field] = correction[field]
        row["oracle_transition_version"] = correction["oracle_transition_version"]
        row["oracle_anchor_stride"] = correction["oracle_anchor_stride"]
        row["oracle_transition_gap"] = correction["oracle_transition_gap"]
        row["reuse_previous_selector"] = correction["reuse_previous_selector"]
        self.corrected_rows += 1

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

    def document_key(self, idx: int) -> tuple[str, str]:
        row = self.rows[idx]
        source = str(row.get("source", "unknown"))
        doc_id = row.get("source_doc_id", row.get("doc_id", idx))
        return source, str(doc_id)

    @property
    def document_keys(self) -> set[tuple[str, str]]:
        return {self.document_key(idx) for idx in range(len(self.rows))}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        target = row
        if self.budget_target_selector is not None:
            targets = row.get("budget_targets", {})
            if self.budget_target_selector in targets:
                target = targets[self.budget_target_selector]
            elif (
                row.get("oracle_retrieval_space") == "post_rope_qk"
                and row.get("oracle_score_selector") == self.budget_target_selector
            ):
                # Single-method extracted JSONL promotes the selected target
                # fields to the row itself and removes ``budget_targets``.
                target = row
            else:
                raise KeyError(
                    f"Row {idx} does not contain Q/K target "
                    f"{self.budget_target_selector!r}."
                )
        return {
            "hidden_state": torch.tensor(row["hidden_state"], dtype=torch.float32),
            "query_drift": torch.tensor(row["query_drift"], dtype=torch.float32),
            "score_features": torch.tensor(row["score_features"], dtype=torch.float32),
            "prev_feedback": torch.tensor(row.get("prev_feedback", [0.0] * 8), dtype=torch.float32),
            "budget_label": torch.tensor(target["budget_label"], dtype=torch.long),
            "reuse_label": torch.tensor(target["reuse_label"], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument(
        "--train-transition-corrections",
        default=None,
        help="Adjacent-token transition sidecar aligned one-to-one with --train-jsonl.",
    )
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help="Independent document-level validation Oracle. Recommended.",
    )
    parser.add_argument(
        "--val-transition-corrections",
        default=None,
        help="Adjacent-token transition sidecar aligned one-to-one with --val-jsonl.",
    )
    parser.add_argument("--output-dir", default="checkpoints/adablock_policy")
    parser.add_argument("--architecture", choices=POLICY_ARCHITECTURES, default="mlp")
    parser.add_argument("--projection-size", type=int, default=64)
    parser.add_argument("--mlp-hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--budget-buckets",
        nargs="+",
        type=int,
        default=[0, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--lambda-reuse", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=0.01)
    parser.add_argument("--keep-nonfinite", action="store_true")
    parser.add_argument("--allow-legacy-oracle", action="store_true")
    parser.add_argument(
        "--budget-target-selector",
        choices=list(QK_SCORE_SELECTORS),
        default=None,
        help="Train on one selector target from a post-RoPE Q/K multi-target Oracle.",
    )
    parser.add_argument(
        "--allow-nonadjacent-transitions",
        action="store_true",
        help="Allow legacy previous-anchor transitions for an intentional ablation.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Per-epoch metrics path. Defaults to <output-dir>/metrics.jsonl.",
    )
    return parser.parse_args()


def grouped_train_val_split(
    dataset: AdaBlockOracleDataset,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset | None]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if val_ratio == 0 or len(dataset.document_keys) < 2:
        return Subset(dataset, list(range(len(dataset)))), None

    group_to_indices: dict[tuple[str, str], list[int]] = {}
    for idx in range(len(dataset)):
        group_to_indices.setdefault(dataset.document_key(idx), []).append(idx)
    group_keys = list(group_to_indices)
    random.Random(seed).shuffle(group_keys)

    target_val_rows = max(1, int(round(len(dataset) * val_ratio)))
    val_keys: set[tuple[str, str]] = set()
    val_rows = 0
    for key in group_keys:
        if len(val_keys) >= len(group_keys) - 1:
            break
        val_keys.add(key)
        val_rows += len(group_to_indices[key])
        if val_rows >= target_val_rows:
            break

    train_indices: list[int] = []
    val_indices: list[int] = []
    for key, indices in group_to_indices.items():
        (val_indices if key in val_keys else train_indices).extend(indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_metrics(
    model: AdaBlockPolicy,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        budget_buckets=model.config.budget_buckets,
        lambda_budget=args.lambda_budget,
        lambda_reuse=args.lambda_reuse,
        lambda_cost=args.lambda_cost,
    )
    return outputs, losses


def empty_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "budget_loss": 0.0,
        "reuse_loss": 0.0,
        "cost_loss": 0.0,
        "budget_acc": 0.0,
        "reuse_acc": 0.0,
    }


def update_totals(
    totals: dict[str, float],
    outputs: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> int:
    batch_size = batch["hidden_state"].shape[0]
    for name in ("loss", "budget_loss", "reuse_loss", "cost_loss"):
        totals[name] += float(losses[name].item()) * batch_size
    totals["budget_acc"] += (
        outputs["budget_logits"].argmax(dim=-1) == batch["budget_label"]
    ).float().sum().item()
    totals["reuse_acc"] += (
        (outputs["reuse_prob"] > 0.5).float() == batch["reuse_label"]
    ).float().sum().item()
    return batch_size


def finalize_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: AdaBlockPolicy,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = empty_totals()
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs, losses = batch_metrics(model, batch, args)
        count += update_totals(totals, outputs, losses, batch)
    return finalize_totals(totals, count)


def main() -> None:
    args = parse_args()
    if args.val_transition_corrections and not args.val_jsonl:
        raise ValueError("--val-transition-corrections requires --val-jsonl.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train_dataset_full = AdaBlockOracleDataset(
        args.train_jsonl,
        drop_nonfinite=not args.keep_nonfinite,
        allow_legacy_oracle=args.allow_legacy_oracle,
        transition_corrections_path=args.train_transition_corrections,
        allow_nonadjacent_transitions=args.allow_nonadjacent_transitions,
        budget_target_selector=args.budget_target_selector,
    )
    if len(train_dataset_full) == 0:
        raise ValueError("Training dataset is empty.")
    if train_dataset_full.dropped_nonfinite:
        print({"event": "dropped_nonfinite_train_rows", "count": train_dataset_full.dropped_nonfinite})

    if args.val_jsonl:
        val_dataset_full = AdaBlockOracleDataset(
            args.val_jsonl,
            drop_nonfinite=not args.keep_nonfinite,
            allow_legacy_oracle=args.allow_legacy_oracle,
            transition_corrections_path=args.val_transition_corrections,
            allow_nonadjacent_transitions=args.allow_nonadjacent_transitions,
            budget_target_selector=args.budget_target_selector,
        )
        overlap = train_dataset_full.document_keys & val_dataset_full.document_keys
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(
                f"Train/validation document leakage: {len(overlap)} overlapping ids; examples={preview}"
            )
        train_dataset: Dataset = train_dataset_full
        val_dataset: Dataset | None = val_dataset_full
    else:
        train_dataset, val_dataset = grouped_train_val_split(
            train_dataset_full, args.val_ratio, args.seed
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    config = AdaBlockPolicyConfig(
        hidden_size=train_dataset_full.hidden_size,
        projection_size=args.projection_size,
        mlp_hidden_size=args.mlp_hidden_size,
        budget_buckets=tuple(args.budget_buckets),
        dropout=args.dropout,
        architecture=args.architecture,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        policy_input_mode="previous_hidden_only",
    )
    model = AdaBlockPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "event": "training_setup",
            "architecture": args.architecture,
            "parameters": parameter_count,
            "train_rows": len(train_dataset),
            "validation_rows": len(val_dataset) if val_dataset is not None else 0,
            "train_documents": len(train_dataset_full.document_keys),
            "validation_documents": (
                len(val_dataset_full.document_keys) if args.val_jsonl else None
            ),
            "train_transition_corrections": args.train_transition_corrections,
            "train_corrected_rows": train_dataset_full.corrected_rows,
            "train_transition_versions": train_dataset_full.transition_versions,
            "validation_transition_corrections": args.val_transition_corrections,
            "validation_corrected_rows": (
                val_dataset_full.corrected_rows if args.val_jsonl else 0
            ),
            "validation_transition_versions": (
                val_dataset_full.transition_versions if args.val_jsonl else {}
            ),
            "lr": args.lr,
            "budget_buckets": config.budget_buckets,
            "policy_input_mode": config.policy_input_mode,
            "metrics_jsonl": str(metrics_path),
        }
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals = empty_totals()
        train_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in progress:
            batch = move_batch(batch, device)
            outputs, losses = batch_metrics(model, batch, args)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_count += update_totals(train_totals, outputs, losses, batch)
            progress.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                budget=f"{losses['budget_loss'].item():.4f}",
                reuse=f"{losses['reuse_loss'].item():.4f}",
                cost=f"{losses['cost_loss'].item():.4f}",
            )

        train_metrics = finalize_totals(train_totals, train_count)
        val_metrics = evaluate(model, val_loader, device, args) if val_loader is not None else {}
        record: dict[str, Any] = {
            "epoch": epoch,
            "architecture": args.architecture,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "config": model.config.__dict__,
            "epoch": epoch,
            "metrics": val_metrics,
            "train_metrics": train_metrics,
            "history": history,
            "training_args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        selection_loss = val_metrics.get("loss", train_metrics["loss"])
        if selection_loss < best_val:
            best_val = selection_loss
            torch.save(checkpoint, output_dir / "best.pt")

        print(record)

    with (output_dir / "history.json").open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
