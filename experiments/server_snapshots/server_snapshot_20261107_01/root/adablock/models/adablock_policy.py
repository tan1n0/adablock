from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


POLICY_ARCHITECTURES = ("mlp", "residual_mlp", "feature_transformer")


@dataclass
class AdaBlockPolicyConfig:
    hidden_size: int = 4096
    projection_size: int = 64
    score_feature_size: int = 7
    feedback_size: int = 8
    mlp_hidden_size: int = 128
    budget_buckets: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.05
    architecture: str = "mlp"
    num_layers: int = 2
    num_attention_heads: int = 4
    policy_input_mode: str = "legacy_unspecified"

    @property
    def policy_input_size(self) -> int:
        return self.projection_size + 1 + self.score_feature_size + self.feedback_size

    def validate(self) -> None:
        if self.policy_input_mode not in {
            "legacy_unspecified",
            "previous_hidden_only",
        }:
            raise ValueError(
                "policy_input_mode must be 'previous_hidden_only' or "
                "'legacy_unspecified'."
            )
        if self.architecture not in POLICY_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture={self.architecture!r}; choose from {POLICY_ARCHITECTURES}."
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.architecture == "feature_transformer":
            if self.projection_size % self.num_attention_heads != 0:
                raise ValueError(
                    "projection_size must be divisible by num_attention_heads for feature_transformer."
                )


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AdaBlockPolicy(nn.Module):
    """Controller for dynamic middle-block retrieval.

    Sink blocks and the local sliding window are fixed by the retrieval code. The
    policy only predicts whether to reuse the previous middle-block set and how
    many middle blocks to retrieve for the current token.
    """

    def __init__(self, config: Optional[AdaBlockPolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or AdaBlockPolicyConfig()
        self.config.validate()

        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.projection_size),
            nn.GELU(),
        )

        if self.config.architecture == "mlp":
            self.backbone = nn.Sequential(
                nn.LayerNorm(self.config.policy_input_size),
                nn.Linear(self.config.policy_input_size, self.config.mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.mlp_hidden_size, self.config.mlp_hidden_size // 2),
                nn.GELU(),
            )
            head_size = self.config.mlp_hidden_size // 2
        elif self.config.architecture == "residual_mlp":
            layers: list[nn.Module] = [
                nn.LayerNorm(self.config.policy_input_size),
                nn.Linear(self.config.policy_input_size, self.config.mlp_hidden_size),
                nn.GELU(),
            ]
            layers.extend(
                ResidualMLPBlock(self.config.mlp_hidden_size, self.config.dropout)
                for _ in range(self.config.num_layers)
            )
            layers.append(nn.LayerNorm(self.config.mlp_hidden_size))
            self.backbone = nn.Sequential(*layers)
            head_size = self.config.mlp_hidden_size
        else:
            token_size = self.config.projection_size
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_size))
            self.query_token_proj = nn.Linear(1, token_size)
            self.score_value_proj = nn.Linear(1, token_size)
            self.score_feature_embedding = nn.Parameter(
                torch.zeros(1, self.config.score_feature_size, token_size)
            )
            self.feedback_token_proj = nn.Linear(self.config.feedback_size, token_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.mlp_hidden_size * 2,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.num_layers,
                norm=nn.LayerNorm(token_size),
            )
            head_size = token_size
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.score_feature_embedding, std=0.02)

        self.reuse_head = nn.Linear(head_size, 1)
        self.budget_head = nn.Linear(head_size, len(self.config.budget_buckets))

    def _encode_features(
        self,
        hidden_state: torch.Tensor,
        query_drift: torch.Tensor,
        score_features: torch.Tensor,
        prev_feedback: torch.Tensor,
    ) -> torch.Tensor:
        hidden_token = self.hidden_proj(hidden_state)
        if self.config.architecture != "feature_transformer":
            x = torch.cat([hidden_token, query_drift, score_features, prev_feedback], dim=-1)
            return self.backbone(x)

        batch_size = hidden_state.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        query_token = self.query_token_proj(query_drift).unsqueeze(1)
        score_tokens = self.score_value_proj(score_features.unsqueeze(-1))
        score_tokens = score_tokens + self.score_feature_embedding
        feedback_token = self.feedback_token_proj(prev_feedback).unsqueeze(1)
        tokens = torch.cat(
            [cls_token, hidden_token.unsqueeze(1), query_token, score_tokens, feedback_token],
            dim=1,
        )
        return self.backbone(tokens)[:, 0]

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

        h = self._encode_features(
            hidden_state=hidden_state,
            query_drift=query_drift,
            score_features=score_features,
            prev_feedback=prev_feedback,
        )
        reuse_logits = self.reuse_head(h).squeeze(-1)
        budget_logits = self.budget_head(h)

        return {
            "reuse_logits": reuse_logits,
            "budget_logits": budget_logits,
            "reuse_prob": torch.sigmoid(reuse_logits),
            "budget_prob": F.softmax(budget_logits, dim=-1),
        }


def adablock_policy_loss(
    outputs: Dict[str, torch.Tensor],
    budget_label: torch.Tensor,
    reuse_label: torch.Tensor,
    budget_buckets: tuple[int, ...] = AdaBlockPolicyConfig().budget_buckets,
    lambda_budget: float = 1.0,
    lambda_reuse: float = 1.0,
    lambda_cost: float = 0.01,
) -> Dict[str, torch.Tensor]:
    budget_loss = F.cross_entropy(outputs["budget_logits"], budget_label.long())
    reuse_loss = F.binary_cross_entropy_with_logits(
        outputs["reuse_logits"], reuse_label.float()
    )

    bucket_values = outputs["budget_logits"].new_tensor(budget_buckets)
    expected_k = (outputs["budget_prob"] * bucket_values).sum(dim=-1)
    cost_loss = expected_k.mean() / max(float(max(budget_buckets)), 1.0)

    total_loss = (
        lambda_budget * budget_loss
        + lambda_reuse * reuse_loss
        + lambda_cost * cost_loss
    )
    return {
        "loss": total_loss,
        "budget_loss": budget_loss.detach(),
        "reuse_loss": reuse_loss.detach(),
        "cost_loss": cost_loss.detach(),
    }
