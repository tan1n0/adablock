from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


QK_SCORE_SELECTORS = (
    "mean_pool",
    "quest_aabb",
    "arkvale_sphere",
    "spla",
    "cobs",
    "phsa",
)


@dataclass
class QKBlockDigests:
    """Block summaries for post-RoPE keys of one attention layer.

    Every tensor starts with ``[num_kv_heads, num_blocks, ...]``.  COBS stores
    a truncated eigendecomposition of the population covariance instead of a
    dense ``head_dim x head_dim`` matrix.
    """

    counts: torch.Tensor
    mean: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    sphere_radius: torch.Tensor
    diagonal_variance: torch.Tensor
    covariance_basis: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    punctuation_mean: torch.Tensor


def _pad_keys_to_blocks(
    keys: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 3:
        raise ValueError(f"keys must have shape [kv_heads, sequence, head_dim], got {keys.shape}.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    num_heads, sequence, head_dim = keys.shape
    if sequence < 1:
        raise ValueError("At least one key is required.")
    num_blocks = (sequence + block_size - 1) // block_size
    padded_length = num_blocks * block_size
    padded = F.pad(keys.float(), (0, 0, 0, padded_length - sequence))
    blocks = padded.view(num_heads, num_blocks, block_size, head_dim)
    valid = torch.arange(padded_length, device=keys.device).view(num_blocks, block_size)
    valid = valid < sequence
    return blocks, valid


@torch.no_grad()
def build_qk_block_digests(
    keys: torch.Tensor,
    block_size: int,
    punctuation_mask: torch.Tensor | None = None,
    cobs_rank: int = 4,
    cobs_svd_block_chunk: int = 128,
) -> QKBlockDigests:
    """Build all six selector digests once for one layer's post-RoPE keys."""

    blocks, valid = _pad_keys_to_blocks(keys, block_size)
    num_heads, num_blocks, _page_size, head_dim = blocks.shape
    valid_f = valid.to(blocks.dtype).view(1, num_blocks, block_size, 1)
    counts = valid.sum(dim=-1).clamp_min(1).to(blocks.dtype)
    mean = (blocks * valid_f).sum(dim=2) / counts.view(1, num_blocks, 1)

    positive_inf = torch.tensor(float("inf"), device=blocks.device, dtype=blocks.dtype)
    negative_inf = torch.tensor(float("-inf"), device=blocks.device, dtype=blocks.dtype)
    minimum = blocks.masked_fill(~valid.view(1, num_blocks, block_size, 1), positive_inf).amin(dim=2)
    maximum = blocks.masked_fill(~valid.view(1, num_blocks, block_size, 1), negative_inf).amax(dim=2)

    centered = (blocks - mean.unsqueeze(2)) * valid_f
    squared_norm = centered.square().sum(dim=-1)
    squared_norm = squared_norm.masked_fill(~valid.view(1, num_blocks, block_size), 0.0)
    sphere_radius = squared_norm.amax(dim=2).sqrt()
    diagonal_variance = centered.square().sum(dim=2) / counts.view(1, num_blocks, 1)

    rank = min(max(int(cobs_rank), 0), block_size - 1, head_dim)
    if rank == 0:
        covariance_basis = blocks.new_empty(num_heads, num_blocks, 0, head_dim)
        covariance_eigenvalues = blocks.new_empty(num_heads, num_blocks, 0)
    else:
        basis_chunks: list[torch.Tensor] = []
        eigenvalue_chunks: list[torch.Tensor] = []
        flat = centered.reshape(num_heads * num_blocks, block_size, head_dim)
        flat_counts = counts.repeat(num_heads)
        chunk_size = max(int(cobs_svd_block_chunk), 1)
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            _u, singular_values, vh = torch.linalg.svd(
                flat[start:end], full_matrices=False
            )
            basis_chunks.append(vh[:, :rank, :])
            eigenvalue_chunks.append(
                singular_values[:, :rank].square()
                / flat_counts[start:end].view(-1, 1)
            )
        covariance_basis = torch.cat(basis_chunks, dim=0).view(
            num_heads, num_blocks, rank, head_dim
        )
        covariance_eigenvalues = torch.cat(eigenvalue_chunks, dim=0).view(
            num_heads, num_blocks, rank
        )

    if punctuation_mask is None:
        punctuation_mean = mean
    else:
        if punctuation_mask.ndim != 1 or punctuation_mask.numel() != keys.shape[1]:
            raise ValueError(
                "punctuation_mask must have shape [sequence] matching the keys."
            )
        padded_mask = F.pad(
            punctuation_mask.to(device=keys.device, dtype=torch.bool),
            (0, num_blocks * block_size - keys.shape[1]),
        ).view(num_blocks, block_size)
        punct_valid = valid & padded_mask
        punct_counts = punct_valid.sum(dim=-1)
        punct_sum = (
            blocks
            * punct_valid.to(blocks.dtype).view(1, num_blocks, block_size, 1)
        ).sum(dim=2)
        punctuation_mean = punct_sum / punct_counts.clamp_min(1).to(blocks.dtype).view(
            1, num_blocks, 1
        )
        punctuation_mean = torch.where(
            (punct_counts > 0).view(1, num_blocks, 1), punctuation_mean, mean
        )

    return QKBlockDigests(
        counts=counts,
        mean=mean,
        minimum=minimum,
        maximum=maximum,
        sphere_radius=sphere_radius,
        diagonal_variance=diagonal_variance,
        covariance_basis=covariance_basis,
        covariance_eigenvalues=covariance_eigenvalues,
        punctuation_mean=punctuation_mean,
    )


@torch.no_grad()
def score_qk_block_digests(
    grouped_query: torch.Tensor,
    digests: QKBlockDigests,
    selectors: Sequence[str] = QK_SCORE_SELECTORS,
    phsa_lambda: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Score blocks for grouped-query attention.

    ``grouped_query`` has shape ``[kv_heads, q_heads_per_kv_head, queries,
    head_dim]`` and must already include RoPE and the model's attention scale.
    Returned scores have shape ``[queries, kv_heads, num_blocks]``.  Query
    heads sharing a KV head are averaged because they share one page choice in
    the layer/head sparse runtime.
    """

    unknown = sorted(set(selectors) - set(QK_SCORE_SELECTORS))
    if unknown:
        raise ValueError(f"Unknown Q/K score selectors: {unknown}.")
    if grouped_query.ndim != 4:
        raise ValueError(
            "grouped_query must have shape [kv_heads, group_size, queries, head_dim]."
        )
    if grouped_query.shape[0] != digests.mean.shape[0]:
        raise ValueError("Query and digest KV-head counts do not match.")
    if not 0.0 <= phsa_lambda <= 1.0:
        raise ValueError("phsa_lambda must be in [0, 1].")

    q = grouped_query.float()
    mean_dot = torch.einsum("hgqd,hbd->hgqb", q, digests.mean)
    outputs: dict[str, torch.Tensor] = {}

    if "mean_pool" in selectors:
        outputs["mean_pool"] = mean_dot.mean(dim=1).permute(1, 0, 2)

    if "quest_aabb" in selectors:
        # max(q * k_min, q * k_max) is q_pos*k_max + q_neg*k_min.
        # Two einsums avoid materializing [H, G, Q, B, D].
        score = torch.einsum(
            "hgqd,hbd->hgqb", q.clamp_min(0.0), digests.maximum
        ) + torch.einsum(
            "hgqd,hbd->hgqb", q.clamp_max(0.0), digests.minimum
        )
        outputs["quest_aabb"] = score.mean(dim=1).permute(1, 0, 2)

    if "arkvale_sphere" in selectors:
        q_norm = q.norm(dim=-1).unsqueeze(-1)
        score = mean_dot + q_norm * digests.sphere_radius[:, None, None, :]
        outputs["arkvale_sphere"] = score.mean(dim=1).permute(1, 0, 2)

    if "spla" in selectors:
        variance_term = torch.einsum(
            "hgqd,hbd->hgqb", q.square(), digests.diagonal_variance
        )
        score = mean_dot + torch.log1p(0.5 * variance_term.clamp_min(0.0))
        score = score + digests.counts.log().view(1, 1, 1, -1)
        outputs["spla"] = score.mean(dim=1).permute(1, 0, 2)

    if "cobs" in selectors:
        if digests.covariance_basis.shape[2] == 0:
            covariance_term = torch.zeros_like(mean_dot)
        else:
            projected = torch.einsum(
                "hgqd,hbrd->hgqbr", q, digests.covariance_basis
            )
            covariance_term = (
                projected.square()
                * digests.covariance_eigenvalues[:, None, None, :, :]
            ).sum(dim=-1)
        score = mean_dot + 0.5 * covariance_term
        score = score + digests.counts.log().view(1, 1, 1, -1)
        outputs["cobs"] = score.mean(dim=1).permute(1, 0, 2)

    if "phsa" in selectors:
        representation = (
            phsa_lambda * digests.mean
            + (1.0 - phsa_lambda) * digests.punctuation_mean
        )
        score = torch.einsum("hgqd,hbd->hgqb", q, representation)
        outputs["phsa"] = score.mean(dim=1).permute(1, 0, 2)

    return outputs
