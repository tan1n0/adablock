from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class AdaBlockPolicyConfig:
    hidden_size: int = 4096
    projection_size: int = 64
    score_feature_size: int = 7
    feedback_size: int = 8
    mlp_hidden_size: int = 128
    budget_buckets: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128)
    num_categories: int = 5
    dropout: float = 0.05

    @property
    def policy_input_size(self) -> int:
        return self.projection_size + 1 + self.score_feature_size + self.feedback_size


class AdaBlockPolicy(nn.Module):
    """Lightweight policy head for AdaBlock block retrieval decisions.

    The policy predicts:
    - reuse probability: whether to reuse the previous retrieved block set
    - budget distribution: which discrete K bucket to use
    - category distribution: how to allocate K over block categories
    """

    def __init__(self, config: Optional[AdaBlockPolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or AdaBlockPolicyConfig()

        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.projection_size),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.config.policy_input_size),
            nn.Linear(self.config.policy_input_size, self.config.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.mlp_hidden_size, self.config.mlp_hidden_size // 2),
            nn.GELU(),
        )

        head_size = self.config.mlp_hidden_size // 2
        self.reuse_head = nn.Linear(head_size, 1)
        self.budget_head = nn.Linear(head_size, len(self.config.budget_buckets))
        self.category_head = nn.Linear(head_size, self.config.num_categories)

    def forward(
        self,
        hidden_state: torch.Tensor,
        query_drift: torch.Tensor,
        score_features: torch.Tensor,
        prev_feedback: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query_drift.ndim == 1:
            query_drift = query_drift.unsqueeze(-1)
        if prev_feedback is None:
            prev_feedback = hidden_state.new_zeros(
                hidden_state.shape[0], self.config.feedback_size
            )

        z = self.hidden_proj(hidden_state)
        x = torch.cat([z, query_drift, score_features, prev_feedback], dim=-1)
        h = self.mlp(x)

        reuse_logits = self.reuse_head(h).squeeze(-1)
        budget_logits = self.budget_head(h)
        category_logits = self.category_head(h)

        return {
            "reuse_logits": reuse_logits,
            "budget_logits": budget_logits,
            "category_logits": category_logits,
            "reuse_prob": torch.sigmoid(reuse_logits),
            "budget_prob": F.softmax(budget_logits, dim=-1),
            "category_prob": F.softmax(category_logits, dim=-1),
        }


def adablock_policy_loss(
    outputs: Dict[str, torch.Tensor],
    budget_label: torch.Tensor,
    reuse_label: torch.Tensor,
    category_target: torch.Tensor,
    budget_buckets: tuple[int, ...] = AdaBlockPolicyConfig().budget_buckets,
    lambda_budget: float = 1.0,
    lambda_reuse: float = 1.0,
    lambda_category: float = 1.0,
    lambda_cost: float = 0.01,
) -> Dict[str, torch.Tensor]:
    budget_loss = F.cross_entropy(outputs["budget_logits"], budget_label.long())
    reuse_loss = F.binary_cross_entropy_with_logits(
        outputs["reuse_logits"], reuse_label.float()
    )

    category_log_prob = F.log_softmax(outputs["category_logits"], dim=-1)
    category_target = category_target / category_target.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    category_loss = F.kl_div(category_log_prob, category_target, reduction="batchmean")

    bucket_values = outputs["budget_logits"].new_tensor(budget_buckets)
    expected_k = (outputs["budget_prob"] * bucket_values).sum(dim=-1)
    cost_loss = expected_k.mean() / max(float(max(budget_buckets)), 1.0)

    total_loss = (
        lambda_budget * budget_loss
        + lambda_reuse * reuse_loss
        + lambda_category * category_loss
        + lambda_cost * cost_loss
    )
    return {
        "loss": total_loss,
        "budget_loss": budget_loss.detach(),
        "reuse_loss": reuse_loss.detach(),
        "category_loss": category_loss.detach(),
        "cost_loss": cost_loss.detach(),
    }
