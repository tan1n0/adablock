"""Block selectors with explicit per-query-head scores and cacheable block work.

Inputs use post-RoPE query/key coordinates. Query vectors already include the
attention scale, matching BlockRanker and the existing ranking shards.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.block_ranker import cobs_terms, position_features


def exact_block_summary(keys: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
    """Summarize complete blocks [H, B, 16, D] using population variance.

    K mean and diagonal variance are exact block statistics, not the diagonal
    of the existing rank-4 covariance approximation. V uses the same layout.
    """
    if keys.ndim != 4 or keys.shape[-2] != 16:
        raise ValueError("Expected complete 16-token blocks [H,B,16,D]")
    k = keys.float()
    parts = [k.mean(-2), k.var(-2, unbiased=False)]
    if values is not None:
        if values.shape != keys.shape:
            raise ValueError("K and V block shapes must match")
        v = values.float()
        parts += [v.mean(-2), v.var(-2, unbiased=False)]
    return torch.cat(parts, -1)


def rank4_diagonal_summary(mean: torch.Tensor, basis: torch.Tensor,
                           eigenvalues: torch.Tensor) -> torch.Tensor:
    """K mean plus diagonal of the stored rank-4 covariance approximation.

    This is the only variance available in existing 2K/4K/8K labels. The
    resulting checkpoints must use this same definition at inference time.
    """
    variance = (basis.float().square() * eigenvalues.float().unsqueeze(-1)).sum(-2)
    return torch.cat((mean.float(), variance), -1)


def group_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Normalize each GQA query head before averaging shared block scores."""
    if logits.ndim < 2:
        raise ValueError("Expected [..., G, B]")
    return logits.float().softmax(-1).mean(-2)


class SimpleCOBS(nn.Module):
    """Three learned terms per query head; no per-pair interaction network."""

    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([1.0, 0.5, 0.0, 0.0]))

    def forward(self, query, mean, basis, eigenvalues, block_ids, counts,
                token_position: int, block_size: int = 16):
        w = self.coefficients
        pos = position_features(block_ids, counts, token_position, block_size)
        if query.ndim == 3:
            # [H,G,D] x [H,B,D] and [H,B,R,D], one launch per term.
            q = query.float()
            first = torch.einsum("hgd,hbd->hgb", q, mean.float())
            projected = torch.einsum("hgd,hbrd->hgbr", q, basis.float())
            second = (projected.square() * eigenvalues.float()[:, None]).sum(-1)
            position = (pos @ w[2:]).unsqueeze(-2)
            return w[0] * first + w[1] * second + position
        first, second = cobs_terms(query, mean, basis, eigenvalues)
        return w[0] * first + w[1] * second + w[2] * pos[:, 0] + w[3] * pos[:, 1]


class DiagonalCOBS(nn.Module):
    """Mean plus exact diagonal variance; both block terms cacheable."""

    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([1.0, 0.5, 0.0, 0.0]))

    def forward(self, scaled_query, summary, block_ids, counts,
                token_position: int, block_size: int = 16):
        d = scaled_query.shape[-1]
        if summary.shape[-1] != 2*d:
            raise ValueError("Expected K mean and exact diagonal variance")
        # One batched GEMM scans the cached mean and variance together.
        w = self.coefficients
        q = scaled_query.float()
        encoded_query = torch.cat((w[0]*q, w[1]*q.square()), dim=-1).to(summary.dtype)
        logits = torch.matmul(encoded_query, summary.transpose(-1,-2)).float()
        pos = position_features(block_ids, counts, token_position, block_size)
        return logits + (pos @ w[2:]).unsqueeze(-2)


@dataclass(frozen=True)
class DualEncoderConfig:
    head_dim: int = 128
    embedding_dim: int = 32
    hidden_dim: int = 128
    use_value: bool = False


class DualEncoder(nn.Module):
    """Encode blocks once, then scan cached embeddings with each query head."""

    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config
        d, e, hidden = config.head_dim, config.embedding_dim, config.hidden_dim
        self.block_encoder = nn.Sequential(nn.LayerNorm(d * (4 if config.use_value else 2)),
                                           nn.Linear(d * (4 if config.use_value else 2), hidden),
                                           nn.GELU(), nn.Linear(hidden, e))
        self.query_encoder = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden),
                                           nn.GELU(), nn.Linear(hidden, e))
        self.position_weight = nn.Parameter(torch.zeros(2))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def encode_blocks(self, summary: torch.Tensor) -> torch.Tensor:
        return self.block_encoder(summary.float())

    def encode_query(self, scaled_query: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(scaled_query.float())

    def score_encoded(self, encoded_query, encoded_blocks, block_ids, counts,
                      token_position: int, block_size: int = 16):
        logits = encoded_query.to(encoded_blocks.dtype) @ encoded_blocks.transpose(-1, -2)
        logits = logits * self.log_temperature.exp().clamp(max=100)
        pos = position_features(block_ids, counts, token_position, block_size)
        return logits + (pos @ self.position_weight).unsqueeze(-2)

    def forward(self, scaled_query, summary, block_ids, counts,
                token_position: int, block_size: int = 16):
        return self.score_encoded(self.encode_query(scaled_query),
                                  self.encode_blocks(summary), block_ids, counts,
                                  token_position, block_size)


class BlockEmbeddingCache:
    """Per-layer/KV-head cache. Append only after all 16 K/V tokens are written."""

    def __init__(self, model: DualEncoder, capacity: int, num_kv_heads: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        self.model = model
        self.data = torch.empty((num_kv_heads, capacity, model.config.embedding_dim),
                                device=device, dtype=dtype)
        self.count = 0

    @torch.no_grad()
    def append(self, keys: torch.Tensor, values: torch.Tensor | None = None):
        """Append [H, number_of_new_blocks, 16, D] without rebuilding old blocks."""
        if self.model.config.use_value and values is None:
            raise ValueError("V summary is required by this encoder")
        summary = exact_block_summary(keys, values if self.model.config.use_value else None)
        encoded = self.model.encode_blocks(summary).to(self.data.dtype)
        n = encoded.shape[1]
        if self.count + n > self.data.shape[1]:
            raise ValueError("Embedding cache capacity exceeded")
        self.data[:, self.count:self.count + n].copy_(encoded)
        self.count += n

    @torch.no_grad()
    def append_summary(self, summary: torch.Tensor):
        """Append prepared summaries [H,N,2D or 4D], e.g. stored rank-4 K."""
        encoded = self.model.encode_blocks(summary).to(self.data.dtype)
        n = encoded.shape[1]
        if self.count + n > self.data.shape[1]:
            raise ValueError("Embedding cache capacity exceeded")
        self.data[:, self.count:self.count + n].copy_(encoded)
        self.count += n

    @torch.no_grad()
    def write_summary(self, block_index: int, summary: torch.Tensor):
        """Write a complete block at its absolute index; supports decode replay."""
        encoded = self.model.encode_blocks(summary).to(self.data.dtype)
        n = encoded.shape[1]
        if block_index < 0 or block_index + n > self.data.shape[1]:
            raise ValueError("Embedding cache index out of bounds")
        self.data[:, block_index:block_index+n].copy_(encoded)
        self.count = max(self.count, block_index+n)

    def view(self):
        return self.data[:, :self.count]
