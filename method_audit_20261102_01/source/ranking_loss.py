"""Coverage-defined positives and sampled, mass-weighted RankNet."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PairSamplingConfig:
    coverage: float = 0.95
    max_positives: int = 16
    hard_negatives: int = 4
    random_negatives: int = 4
    positive_pairs: int = 32
    delta: float = 1e-6

    def validate(self):
        if not 0 < self.coverage <= 1 or self.delta < 0:
            raise ValueError("Invalid coverage or delta.")
        if self.max_positives < 1 or min(self.hard_negatives, self.random_negatives, self.positive_pairs) < 0:
            raise ValueError("Invalid pair sampling counts.")


def coverage_positive_ids(mass, fixed_mass, coverage):
    mass = mass.detach().float()
    if mass.numel() == 0 or float(fixed_mass) >= coverage - 1e-7:
        return torch.empty(0, dtype=torch.long, device=mass.device)
    order = torch.argsort(mass, descending=True, stable=True)
    cumulative = float(fixed_mass) + mass[order].cumsum(0)
    k = min(int((cumulative < coverage - 1e-7).sum()) + 1, mass.numel())
    return order[:k]


def sample_pairs(scores, mass, fixed_mass, config, generator=None):
    """Sampling on CPU is detached; only selected score differences get loss."""
    config.validate()
    mass_cpu = mass.detach().float().cpu()
    score_cpu = scores.detach().float().cpu()
    positive = coverage_positive_ids(mass_cpu, fixed_mass, config.coverage)
    if not positive.numel():
        return torch.empty((0, 2), dtype=torch.long, device=scores.device)
    positive_mask = torch.zeros(mass.numel(), dtype=torch.bool)
    positive_mask[positive] = True
    negative = torch.where(~positive_mask)[0]
    picked = positive[torch.randperm(positive.numel(), generator=generator)[:config.max_positives]]
    pairs = set()
    for i in picked.tolist():
        valid = negative[mass_cpu[i] - mass_cpu[negative] > config.delta]
        hard = valid[torch.argsort(score_cpu[valid], descending=True, stable=True)[:config.hard_negatives]]
        for j in hard.tolist():
            pairs.add((i, j))
        remaining = valid[~torch.isin(valid, hard)]
        random = remaining[torch.randperm(remaining.numel(), generator=generator)[:config.random_negatives]]
        for j in random.tolist():
            pairs.add((i, j))
    # Bounded rejection sampling rather than a quadratic positive-pair matrix.
    internal = set()
    if positive.numel() > 1:
        for _ in range(max(32, config.positive_pairs * 10)):
            if len(internal) >= config.positive_pairs:
                break
            indices = torch.randint(positive.numel(), (2,), generator=generator)
            i, j = positive[indices].tolist()
            if mass_cpu[i] < mass_cpu[j]:
                i, j = j, i
            if mass_cpu[i] - mass_cpu[j] > config.delta:
                internal.add((i, j))
    pairs.update(internal)
    return torch.tensor(sorted(pairs), dtype=torch.long, device=scores.device).reshape(-1, 2)


def weighted_rank_loss(scores, mass, pairs):
    if not pairs.numel():
        return scores.sum() * 0.0
    i, j = pairs.unbind(-1)
    weights = (mass[i] - mass[j]).detach().float()
    if not bool((weights > 0).all()):
        raise ValueError("Pairs must be oriented by strictly decreasing teacher mass.")
    return (weights * F.softplus(scores[j].float() - scores[i].float())).sum() / weights.sum().clamp_min(1e-8)
