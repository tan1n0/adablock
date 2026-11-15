from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


@dataclass
class OracleConfig:
    block_size: int = 16
    sink_blocks: int = 2
    local_window_blocks: int = 4
    budget_buckets: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128)
    coverage_threshold: float = 0.9
    reuse_coverage_threshold: float = 0.9


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


def fixed_and_middle_blocks(
    num_blocks: int,
    sink_blocks: int,
    local_window_blocks: int,
    current_block: int | None = None,
) -> tuple[set[int], set[int], set[int]]:
    """Return fixed sink, fixed local, and eligible middle block ids."""
    if num_blocks <= 0:
        return set(), set(), set()
    current = num_blocks - 1 if current_block is None else min(current_block, num_blocks - 1)
    sink = set(range(min(max(sink_blocks, 0), num_blocks)))
    local_start = max(0, current - max(local_window_blocks, 0) + 1)
    local = set(range(local_start, current + 1))
    middle = set(range(num_blocks)) - sink - local
    return sink, local, middle


def dynamic_oracle_budget_label(
    block_mass: torch.Tensor,
    fixed_blocks: Iterable[int],
    middle_blocks: Iterable[int],
    coverage_threshold: float,
    budget_buckets: Sequence[int],
    always_retained_mass: float | torch.Tensor = 0.0,
) -> tuple[int, int]:
    """Minimum number of middle blocks needed after fixed sink/local blocks."""
    coverage_floor = coverage_threshold - 1e-7
    always_mass = block_mass.new_tensor(always_retained_mass)
    total = (block_mass.sum() + always_mass).clamp_min(1e-8)
    fixed = [idx for idx in fixed_blocks if 0 <= idx < block_mass.numel()]
    fixed_mass = always_mass + (
        block_mass[fixed].sum() if fixed else block_mass.new_tensor(0.0)
    )
    if float(fixed_mass / total) >= coverage_floor:
        return bucket_index(0, budget_buckets), 0

    middle = [idx for idx in middle_blocks if 0 <= idx < block_mass.numel()]
    if not middle:
        return bucket_index(0, budget_buckets), 0
    sorted_mass = torch.sort(block_mass[middle], descending=True).values
    cumulative = (fixed_mass + torch.cumsum(sorted_mass, dim=0)) / total
    required_k = int((cumulative < coverage_floor).sum().item() + 1)
    required_k = min(required_k, len(middle))
    return bucket_index(required_k, budget_buckets), required_k


def selector_oracle_budget_label(
    block_mass: torch.Tensor,
    block_scores: torch.Tensor,
    fixed_blocks: Iterable[int],
    middle_blocks: Iterable[int],
    coverage_threshold: float,
    budget_buckets: Sequence[int],
    always_retained_mass: float | torch.Tensor = 0.0,
) -> tuple[int, int]:
    """Budget label when blocks are selected by a score-based selector.

    Blocks are ranked by ``block_scores`` (for example Quest), while the
    achieved coverage is measured with the full-attention ``block_mass``.
    This separates the selection policy from the ground-truth coverage metric.
    """
    coverage_floor = coverage_threshold - 1e-7
    always_mass = block_mass.new_tensor(always_retained_mass)
    total = (block_mass.sum() + always_mass).clamp_min(1e-8)
    fixed = [idx for idx in fixed_blocks if 0 <= idx < block_mass.numel()]
    fixed_mass = always_mass + (
        block_mass[fixed].sum() if fixed else block_mass.new_tensor(0.0)
    )
    if float(fixed_mass / total) >= coverage_floor:
        return bucket_index(0, budget_buckets), 0

    middle = [
        idx
        for idx in middle_blocks
        if 0 <= idx < block_mass.numel() and idx < block_scores.numel()
    ]
    if not middle:
        return bucket_index(0, budget_buckets), 0
    score_tensor = block_scores.detach().float()
    ordered = sorted(middle, key=lambda idx: float(score_tensor[idx]), reverse=True)
    cumulative = fixed_mass
    required_k = len(ordered)
    for rank, block_id in enumerate(ordered, start=1):
        cumulative = cumulative + block_mass[block_id]
        if float(cumulative / total) >= coverage_floor:
            required_k = rank
            break
    return bucket_index(required_k, budget_buckets), required_k


def reuse_label_from_previous_middle(
    block_mass: torch.Tensor,
    fixed_blocks: Iterable[int],
    previous_middle_blocks: Iterable[int],
    reuse_coverage_threshold: float,
    always_retained_mass: float | torch.Tensor = 0.0,
) -> int:
    previous = set(previous_middle_blocks)
    if not previous:
        return 0
    selected = set(fixed_blocks) | previous
    valid = [idx for idx in selected if 0 <= idx < block_mass.numel()]
    if not valid:
        return 0
    always_mass = block_mass.new_tensor(always_retained_mass)
    total = (block_mass.sum() + always_mass).clamp_min(1e-8)
    coverage = (block_mass[valid].sum() + always_mass) / total
    return int(float(coverage) >= reuse_coverage_threshold)


def top_blocks(
    block_mass: torch.Tensor,
    k: int,
    candidates: Iterable[int] | None = None,
) -> list[int]:
    if k <= 0:
        return []
    if candidates is None:
        valid = list(range(block_mass.numel()))
    else:
        valid = [idx for idx in candidates if 0 <= idx < block_mass.numel()]
    if not valid:
        return []
    take = min(k, len(valid))
    candidate_tensor = torch.tensor(valid, dtype=torch.long, device=block_mass.device)
    local_top = torch.topk(block_mass[candidate_tensor], k=take).indices
    return candidate_tensor[local_top].tolist()


def select_middle_blocks(
    block_scores: torch.Tensor,
    middle_blocks: Iterable[int],
    k: int,
) -> set[int]:
    return set(top_blocks(block_scores, k, candidates=middle_blocks))


def compose_block_selection(
    block_scores: torch.Tensor,
    dynamic_k: int,
    sink_blocks: int,
    local_window_blocks: int,
    current_block: int | None = None,
) -> tuple[set[int], set[int], set[int]]:
    sink, local, middle = fixed_and_middle_blocks(
        num_blocks=block_scores.numel(),
        sink_blocks=sink_blocks,
        local_window_blocks=local_window_blocks,
        current_block=current_block,
    )
    dynamic = select_middle_blocks(block_scores, middle, dynamic_k)
    return sink | local | dynamic, dynamic, middle


def fill_selection_to_total_budget(
    selected: set[int],
    scores: torch.Tensor,
    total_budget_blocks: int,
    required_blocks: Iterable[int],
) -> set[int]:
    required = {idx for idx in required_blocks if 0 <= idx < scores.numel()}
    target = min(max(total_budget_blocks, len(required)), scores.numel())
    ordered: list[int] = []
    seen: set[int] = set()
    for idx in sorted(required):
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    for idx in sorted(selected - required, key=lambda item: float(scores[item]), reverse=True):
        if 0 <= idx < scores.numel() and idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    for idx in torch.argsort(scores, descending=True).tolist():
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
        if len(ordered) >= target:
            break
    return set(ordered[:target])


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


def cosine_block_scores(
    hidden_states: torch.Tensor,
    block_ranges: Sequence[tuple[int, int]],
    t: int,
) -> torch.Tensor:
    query = hidden_states[t].float()
    centroids: List[torch.Tensor] = []
    for start, end in block_ranges:
        centroids.append(hidden_states[start:end].float().mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)
    return F.cosine_similarity(query.unsqueeze(0), centroid_tensor, dim=-1)


def quest_block_scores(
    hidden_states: torch.Tensor,
    block_ranges: Sequence[tuple[int, int]],
    t: int,
) -> torch.Tensor:
    query = hidden_states[t].float()
    scores: List[torch.Tensor] = []
    for start, end in block_ranges:
        block = hidden_states[start:end].float()
        block_min = block.min(dim=0).values
        block_max = block.max(dim=0).values
        upper = torch.maximum(query * block_min, query * block_max).sum()
        scores.append(upper)
    return torch.stack(scores, dim=0)


def arkvale_block_scores(
    hidden_states: torch.Tensor,
    block_ranges: Sequence[tuple[int, int]],
    t: int,
) -> torch.Tensor:
    """ArkVale page-digest upper-bound scores.

    ArkVale's strongest reported variant is the cuboid-mean digest. It uses
    the AABB center and element-wise mean radius, producing a tighter cuboid
    than the raw min/max (rmax) bound used by Quest here.
    """
    query = hidden_states[t].float()
    scores: List[torch.Tensor] = []
    for start, end in block_ranges:
        block = hidden_states[start:end].float()
        block_min = block.min(dim=0).values
        block_max = block.max(dim=0).values
        center = 0.5 * (block_min + block_max)
        radius_mean = (block - center).abs().mean(dim=0)
        bound_max = center + radius_mean
        bound_min = center - radius_mean
        scores.append(torch.maximum(query * bound_max, query * bound_min).sum())
    return torch.stack(scores, dim=0)
