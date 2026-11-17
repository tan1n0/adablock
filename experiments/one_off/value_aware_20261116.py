"""Exact block attention contributions and bounded value-aware swap search.

The search is a local reference, not a global optimum. Inputs are post-RoPE
Q/K, current token included in K/V, and query is not pre-scaled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def contributions(query, keys, values, scale: float, block_size: int = 16):
    """Return mass [H,G,B+1], weighted V [H,G,B+1,D], and Full output.

    Last block is the visible current block (1..16 tokens). Full historical
    blocks occupy [0:B]. Attention is computed in fp32 for stable reduction.
    """
    if query.ndim != 3 or keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError("Expected Q [H,G,D] and K/V [H,T,D]")
    if query.shape[0] != keys.shape[0] or query.shape[-1] != keys.shape[-1]:
        raise ValueError("GQA dimensions disagree")
    h, length, d = keys.shape
    full, tail = divmod(length - 1, block_size)
    if tail + 1 < 1 or tail + 1 > block_size:
        raise AssertionError("Invalid visible current block")
    logits = torch.matmul(query.float(), keys.float().transpose(-1, -2)) * scale
    probabilities = logits.softmax(-1)
    hist = probabilities[..., :full * block_size].reshape(h, query.shape[1], full, block_size)
    hist_v = values[:, :full * block_size].float().reshape(h, full, block_size, d)
    mass = hist.sum(-1)
    contribution = torch.einsum("hgbi,hbid->hgbd", hist, hist_v)
    current_p = probabilities[..., full * block_size:]
    current_v = values[:, full * block_size:].float()
    current_mass = current_p.sum(-1, keepdim=True)
    current_contribution = torch.einsum("hgi,hid->hgd", current_p, current_v).unsqueeze(-2)
    mass = torch.cat((mass, current_mass), -1)
    contribution = torch.cat((contribution, current_contribution), -2)
    return mass, contribution, contribution.sum(-2)


def reconstructed(mass, contribution, selected):
    """Selected historical block IDs; current block is always included."""
    chosen = torch.cat((selected.long(), selected.new_tensor([mass.shape[-1] - 1])), -1)
    m = mass.index_select(-1, chosen).sum(-1).clamp_min(torch.finfo(mass.dtype).tiny)
    c = contribution.index_select(-2, chosen).sum(-2)
    return c / m.unsqueeze(-1)


def mass_oracle(mass, take: int):
    historical = mass.shape[-1] - 1
    if take > historical or take < 0:
        raise ValueError("Invalid historical budget")
    return mass[..., :historical].mean(-2).argsort(dim=-1, descending=True, stable=True)[..., :take]


def swap_search(mass, contribution, initial, max_swaps: int = 8,
                candidate_chunk: int = 128, min_improvement: float = 1e-9,
                candidate_pool: int | None = None):
    """Steepest improving one-block swap, all unselected blocks considered.

    For each KV head independently, cost is O((max_swaps+1)*K*B*G*D),
    memory O(K*candidate_chunk*G*D). Stable ties keep the first exchange.
    Optional candidate_pool retains the top mass blocks as incoming candidates.
    Returns selected IDs and accepted (out, in, squared-error gain) records.
    """
    if mass.ndim != 2 or contribution.ndim != 3 or initial.ndim != 1:
        raise ValueError("Expected one KV head: [G,B+1], [G,B+1,D], [K]")
    historical = mass.shape[-1] - 1
    if len(initial) > historical or len(initial.unique()) != len(initial):
        raise ValueError("Invalid initial selection")
    if candidate_chunk < 1 or max_swaps < 0:
        raise ValueError("Invalid search budget")
    selected = initial.clone().long()
    if candidate_pool is not None and candidate_pool < len(selected):
        raise ValueError("candidate_pool must cover the initial selection")
    ranked = mass[:, :historical].mean(0).argsort(dim=-1, descending=True, stable=True)
    candidate_universe = ranked if candidate_pool is None else ranked[:candidate_pool]
    target = contribution.sum(-2)
    records = []
    def error(ids):
        return (reconstructed(mass, contribution, ids) - target).square().sum(-1).mean()
    current_error = error(selected)
    for _ in range(max_swaps):
        available = torch.ones(historical, dtype=torch.bool, device=mass.device)
        available[selected] = False
        incoming = candidate_universe[available[candidate_universe]]
        if not incoming.numel() or not selected.numel():
            break
        base_m = mass.index_select(-1, selected).sum(-1) + mass[:, -1]
        base_c = contribution.index_select(-2, selected).sum(-2) + contribution[:, -1]
        best_gain = 0.0
        best_pair = None
        for start in range(0, len(incoming), candidate_chunk):
            candidate_ids = incoming[start:start + candidate_chunk]
            new_m = (base_m[:, None, None] - mass[:, selected, None] +
                     mass[:, None, candidate_ids])
            new_c = (base_c[:, None, None, :] - contribution[:, selected, None, :] +
                     contribution[:, None, candidate_ids, :])
            output = new_c / new_m.clamp_min(torch.finfo(mass.dtype).tiny)[..., None]
            errors = (output - target[:, None, None, :]).square().sum(-1).mean(0)
            flat = errors.reshape(-1)
            index = int(flat.argmin())
            gain = float(current_error - flat[index])
            if gain > best_gain:
                best_gain = gain
                best_pair = (index // len(candidate_ids), int(candidate_ids[index % len(candidate_ids)]))
        if best_pair is None or best_gain <= min_improvement:
            break
        slot, incoming_id = best_pair
        outgoing_id = int(selected[slot])
        selected[slot] = incoming_id
        records.append((outgoing_id, incoming_id, best_gain))
        current_error = error(selected)
    return selected, records


def direct_gather(query, keys, values, selected, scale: float, block_size: int = 16):
    """Independent gather plus SDPA check for one KV head and its GQA group."""
    length = keys.shape[0]
    full = (length - 1) // block_size
    ids = (selected[:, None] * block_size +
           torch.arange(block_size, device=selected.device)).flatten()
    ids = torch.cat((ids, torch.arange(full * block_size, length, device=keys.device)))
    return F.scaled_dot_product_attention(
        query[None, :, None].float(),
        keys.index_select(0, ids)[None, None].float().expand(1, query.shape[0], -1, -1),
        values.index_select(0, ids)[None, None].float().expand(1, query.shape[0], -1, -1),
        dropout_p=0, is_causal=False, scale=scale)[0, :, 0]


def selftest(seed=20261116):
    torch.manual_seed(seed)
    checks = []
    for length in (1, 16, 17, 31, 32, 33, 48, 65, 81, 97, 129):
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, length, 8)
        v = torch.randn_like(k)
        # Explicitly rotate each K at its own position and Q at the current
        # position. The API consumes these post-RoPE tensors, as the decoder does.
        angles = torch.arange(length).float()[:, None] / torch.tensor([2., 4., 8., 16.])
        cos, sin = angles.cos()[None], angles.sin()[None]
        k0, k1 = k[..., :4].clone(), k[..., 4:].clone()
        k = torch.cat((k0*cos-k1*sin, k0*sin+k1*cos), -1)
        q0, q1 = q[..., :4].clone(), q[..., 4:].clone()
        qc, qs = cos[:, -1:, :], sin[:, -1:, :]
        q = torch.cat((q0*qc-q1*qs, q0*qs+q1*qc), -1)
        m, c, full = contributions(q, k, v, 8 ** -0.5)
        take = min(2, (length - 1) // 16)
        selected = mass_oracle(m, take)
        assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-6)
        for head in range(2):
            all_history = torch.arange((length-1)//16)
            full_gather = direct_gather(q[head], k[head], v[head], all_history, 8 ** -0.5)
            assert torch.allclose(full[head], full_gather, atol=2e-6)
            reconstructed_y = reconstructed(m[head], c[head], selected[head])
            gathered_y = direct_gather(q[head], k[head], v[head], selected[head], 8 ** -0.5)
            deviation = float((reconstructed_y - gathered_y).abs().max())
            assert deviation < 2e-6, (length, head, deviation)
            improved, swaps = swap_search(m[head], c[head], selected[head], max_swaps=4)
            old = (reconstructed_y - full[head]).square().sum(-1).mean()
            new = (reconstructed(m[head], c[head], improved) - full[head]).square().sum(-1).mean()
            assert new <= old + 1e-7
            checks.append({"length": length, "head": head, "max_abs_error": deviation,
                           "accepted_swaps": len(swaps), "mse_before": float(old),
                           "mse_after": float(new)})
    assert sum(x["accepted_swaps"] for x in checks) > 0, "Search test accepted no swaps"
    q = torch.randn(1, 4, 8) * 100
    k = torch.randn(1, 65, 8) * 100
    v = torch.randn(1, 65, 8)
    m, c, full = contributions(q, k, v, 8 ** -0.5)
    assert torch.isfinite(m).all() and torch.isfinite(c).all() and torch.isfinite(full).all()
    return {"seed": seed, "checks": checks,
            "accepted_swaps": sum(x["accepted_swaps"] for x in checks),
            "max_abs_formula_vs_gather": max(x["max_abs_error"] for x in checks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.selftest:
        parser.error("Use --selftest")
    result = selftest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
