from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


CATEGORY_NAMES = ("local", "input", "generated", "anchor", "high_hit")


@dataclass
class OracleConfig:
    block_size: int = 16
    local_window_blocks: int = 4
    budget_buckets: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128)
    coverage_threshold: float = 0.9
    reuse_coverage_threshold: float = 0.9
    high_hit_top_fraction: float = 0.1


def make_block_ranges(seq_len: int, block_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + block_size, seq_len)) for i in range(0, seq_len, block_size)]


def aggregate_block_mass(
    token_attention: torch.Tensor,
    block_ranges: Sequence[tuple[int, int]],
) -> torch.Tensor:
    return torch.stack(
        [token_attention[start:end].sum() for start, end in block_ranges], dim=0
    )


def bucket_index(k: int, buckets: Sequence[int]) -> int:
    for idx, value in enumerate(buckets):
        if k <= value:
            return idx
    return len(buckets) - 1


def oracle_budget_label(
    block_mass: torch.Tensor,
    coverage_threshold: float,
    budget_buckets: Sequence[int],
) -> tuple[int, int]:
    total = block_mass.sum().clamp_min(1e-8)
    sorted_mass = torch.sort(block_mass, descending=True).values
    cumulative = torch.cumsum(sorted_mass, dim=0) / total
    required_k = int((cumulative < coverage_threshold).sum().item() + 1)
    required_k = min(required_k, block_mass.numel())
    return bucket_index(required_k, budget_buckets), required_k


def reuse_label_from_previous(
    block_mass: torch.Tensor,
    previous_blocks: Iterable[int],
    reuse_coverage_threshold: float,
) -> int:
    previous = list(previous_blocks)
    if not previous:
        return 0
    total = block_mass.sum().clamp_min(1e-8)
    valid = [idx for idx in previous if 0 <= idx < block_mass.numel()]
    if not valid:
        return 0
    coverage = block_mass[valid].sum() / total
    return int(float(coverage) >= reuse_coverage_threshold)


def top_blocks(block_mass: torch.Tensor, k: int) -> list[int]:
    if k <= 0:
        return []
    k = min(k, block_mass.numel())
    return torch.topk(block_mass, k=k).indices.tolist()


def block_categories(
    num_blocks: int,
    prompt_blocks: int,
    current_block: int,
    high_hit_blocks: set[int],
    config: OracleConfig,
) -> torch.Tensor:
    categories = torch.zeros(num_blocks, dtype=torch.long)
    for idx in range(num_blocks):
        if idx >= max(0, current_block - config.local_window_blocks + 1):
            categories[idx] = 0
        elif idx < prompt_blocks:
            categories[idx] = 1
        else:
            categories[idx] = 2
        if idx == 0:
            categories[idx] = 3
        if idx in high_hit_blocks and categories[idx] != 0:
            categories[idx] = 4
    return categories


def category_target(block_mass: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(len(CATEGORY_NAMES), dtype=block_mass.dtype)
    for idx in range(len(CATEGORY_NAMES)):
        target[idx] = block_mass[categories == idx].sum()
    return target / target.sum().clamp_min(1e-8)


def score_summary_features(block_scores: torch.Tensor) -> torch.Tensor:
    if block_scores.numel() == 0:
        return torch.zeros(7)
    scores = F.softmax(block_scores.float(), dim=0)
    sorted_scores = torch.sort(scores, descending=True).values
    top8 = sorted_scores[: min(8, sorted_scores.numel())].mean()
    top32 = sorted_scores[: min(32, sorted_scores.numel())].mean()
    margin = (
        sorted_scores[0] - sorted_scores[1]
        if sorted_scores.numel() > 1
        else sorted_scores.new_tensor(1.0)
    )
    entropy = -(scores * scores.clamp_min(1e-8).log()).sum()
    tail_count = max(1, int(0.25 * sorted_scores.numel()))
    tail_mass = sorted_scores[-tail_count:].sum()
    return torch.stack(
        [
            scores.max(),
            scores.mean(),
            top8,
            top32,
            margin,
            entropy,
            tail_mass,
        ]
    )


def cosine_block_scores(hidden_states: torch.Tensor, block_ranges: Sequence[tuple[int, int]], t: int) -> torch.Tensor:
    query = hidden_states[t].float()
    centroids: List[torch.Tensor] = []
    for start, end in block_ranges:
        centroids.append(hidden_states[start:end].float().mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)
    return F.cosine_similarity(query.unsqueeze(0), centroid_tensor, dim=-1)


def quest_block_scores(hidden_states: torch.Tensor, block_ranges: Sequence[tuple[int, int]], t: int) -> torch.Tensor:
    query = hidden_states[t].float()
    scores: List[torch.Tensor] = []
    for start, end in block_ranges:
        block = hidden_states[start:end].float()
        block_min = block.min(dim=0).values
        block_max = block.max(dim=0).values
        # Quest-style upper-bound proxy: for each dimension, choose the key envelope endpoint
        # that maximizes the query-key contribution, then sum across dimensions.
        upper = torch.maximum(query * block_min, query * block_max).sum()
        scores.append(upper)
    return torch.stack(scores, dim=0)
