"""Fixed-budget, teacher-mass evaluation; no labels used to score learned blocks."""
from __future__ import annotations

from collections import defaultdict

import torch

from models.block_ranker import cobs_terms, to_device


def baseline_scores(sample):
    q = sample["query"].float()
    first, second = cobs_terms(q, sample["mean"], sample["basis"], sample["eigenvalues"])
    quest = (torch.einsum("gd,bd->gb", q.clamp_min(0), sample["maximum"].float())
             + torch.einsum("gd,bd->gb", q.clamp_max(0), sample["minimum"].float()))
    # Deliberately preserve the repository's legacy mean-of-log-scores GQA
    # baseline. This is NOT the normalized-head aggregation in the COBS paper.
    return {"mean_pool": first.mean(0), "quest_aabb": quest.mean(0),
            "cobs_legacy_mean_logits": (first + .5 * second + sample["counts"].float().log()).mean(0),
            "oracle": sample["teacher_mass"].float()}


@torch.no_grad()
def evaluate_ranker(model, dataset, device="cpu", budgets=(4, 8, 16, 32, 64, 128),
                    coverage=0.95, max_scenes=None):
    budgets = sorted(set(int(k) for k in budgets))
    if not budgets or min(budgets) < 0 or not 0 < coverage <= 1:
        raise ValueError("Invalid evaluation budgets/coverage.")
    model.eval()
    indices = range(len(dataset))
    if max_scenes is not None and max_scenes < len(dataset):
        if max_scenes < 1:
            raise ValueError("max_scenes must be positive.")
        indices = torch.linspace(0, len(dataset) - 1, max_scenes).long().unique().tolist()
    accumulators = {}
    counts = defaultdict(int)
    for index in indices:
        sample = to_device(dataset[index], device)
        mass, fixed = sample["teacher_mass"].float(), float(sample["fixed_mass"])
        if not mass.numel():
            continue
        scores = baseline_scores(sample)
        scores["learned"] = model(sample)
        t = sample["token_position"]
        length_bin = next((name for end, name in [(4096, "0-4K"), (8192, "4-8K"),
                          (16384, "8-16K"), (32768, "16-32K")] if t < end), "32K+")
        oracle_curve = mass.sort(descending=True).values.cumsum(0)
        for group in ("overall", length_bin):
            counts[group] += 1
            if group not in accumulators:
                accumulators[group] = {name: defaultdict(float) for name in scores}
            for name, score in scores.items():
                if not bool(score.isfinite().all()):
                    raise FloatingPointError(f"Nonfinite scores from {name}.")
                curve = mass[score.argsort(descending=True, stable=True)].cumsum(0)
                total_curve = fixed + curve
                required = 0 if fixed >= coverage - 1e-7 else min(
                    int((total_curve < coverage - 1e-7).sum()) + 1, mass.numel())
                out = accumulators[group][name]
                out["required_k"] += required
                out["over_max_budget_rate"] += float(required > max(budgets))
                for k in budgets:
                    take = min(k, mass.numel())
                    selected_mass = float(curve[take - 1]) if take else 0.0
                    best_mass = float(oracle_curve[take - 1]) if take else 0.0
                    out[f"coverage@{k}"] += fixed + selected_mass
                    out[f"middle_mass@{k}"] += selected_mass
                    out[f"mass_gap@{k}"] += max(0.0, best_mass - selected_mass)
                    out[f"actual_k@{k}"] += take
    if not counts["overall"]:
        raise ValueError("No ranking scenes to evaluate.")
    return {"scenes": counts["overall"], "dataset_scenes": len(dataset), "budgets": budgets,
            "coverage_target": coverage, "gqa_baseline": "mean raw per-query scores (legacy repository)",
            "groups": {g: {"scenes": counts[g], "methods": {
                name: {key: value / counts[g] for key, value in values.items()}
                for name, values in methods.items()}} for g, methods in accumulators.items()}}
