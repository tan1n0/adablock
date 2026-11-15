# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the JSSA decode path: FP8 projected-key store + score,
and the gather-free indexed attention against a PyTorch reference."""

import pytest
import torch

from triton_jssa_gather import jssa_gather
from triton_jssa_indexed_attend import jssa_indexed_attend
from triton_jssa_score_topk import jssa_score, jssa_score_topk
from triton_jssa_store import (
    jssa_project_and_store,
    jssa_project_and_store_contig_fp8,
)

DEVICE = "cuda"


def _orthonormal_basis(H_kv, r, d, dtype):
    g = torch.randn(H_kv, d, r, device=DEVICE, dtype=torch.float32)
    q, _ = torch.linalg.qr(g)  # [H_kv, d, r], orthonormal columns
    return q.transpose(1, 2).contiguous().to(dtype)  # [H_kv, r, d]


def _paged_setup(num_reqs, H_kv, d, seq_len, block_size, slot, dtype):
    blocks_per_req = (seq_len + block_size - 1) // block_size
    num_blocks = num_reqs * blocks_per_req + 3
    kv_cache = torch.zeros(num_blocks, block_size, H_kv, slot, device=DEVICE, dtype=dtype)
    # shuffled physical blocks
    perm = torch.randperm(num_blocks - 3, device=DEVICE)[: num_reqs * blocks_per_req] + 3
    block_table = perm.view(num_reqs, blocks_per_req).to(torch.int32)
    pos = torch.arange(seq_len, device=DEVICE)
    slot_mapping = torch.cat(
        [block_table[r, pos // block_size].long() * block_size + pos % block_size
         for r in range(num_reqs)]
    )
    K = torch.randn(num_reqs * seq_len, H_kv, d, device=DEVICE, dtype=dtype)
    V = torch.randn(num_reqs * seq_len, H_kv, d, device=DEVICE, dtype=dtype)
    return kv_cache, block_table, slot_mapping, K, V


@pytest.mark.parametrize("d,r", [(128, 32), (512, 64)])
def test_fp8_contig_score_matches_bf16(d, r):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_reqs, H_kv, seq_len, block_size = 2, 4, 1000, 16
    kv_cache, bt, sm, K, V = _paged_setup(num_reqs, H_kv, d, seq_len, block_size, 2 * d, dtype)
    M = _orthonormal_basis(H_kv, r, d, dtype)
    projk = torch.zeros(kv_cache.shape[0], block_size, H_kv, r, device=DEVICE,
                        dtype=torch.float8_e4m3fn)
    scale = M.float().norm(dim=2).clamp_min(1e-6).to(dtype).contiguous()
    jssa_project_and_store_contig_fp8(K, V, kv_cache, projk, scale, sm, M)
    # K|V landed in the slot
    blk, off = sm // block_size, sm % block_size
    torch.testing.assert_close(kv_cache[blk, off, :, :d], K)
    torch.testing.assert_close(kv_cache[blk, off, :, d:], V)

    q = torch.randn(num_reqs, H_kv, d, device=DEVICE, dtype=dtype)
    proj_q = torch.einsum("rhd,hcd->rhc", q.float(), M.float())
    seq_lens = torch.full((num_reqs,), seq_len, device=DEVICE, dtype=torch.int32)
    s8 = jssa_score((proj_q * scale.float().unsqueeze(0)).to(dtype), kv_cache, bt,
                    seq_lens, d, r, projk_cache=projk)
    ref = torch.einsum("rhc,rthc->rht", proj_q,
                       torch.einsum("nhd,hcd->nhc", K.float(), M.float())
                       .view(num_reqs, seq_len, H_kv, r))
    got = s8[..., :seq_len]
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.08, f"fp8 score relative error {rel:.4f}"
    assert torch.isinf(s8[..., seq_len:]).all()


@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("num_reqs", [1, 16])
@pytest.mark.parametrize("d", [128, 512])
def test_indexed_attend_matches_reference(G, num_reqs, d):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    H_kv, r, seq_len, block_size, n_fac = 2, 32, 3000, 16, 256
    slot = 2 * d + r
    kv_cache, bt, sm, K, V = _paged_setup(num_reqs, H_kv, d, seq_len, block_size, slot, dtype)
    M = _orthonormal_basis(H_kv, r, d, dtype)
    jssa_project_and_store(K, V, kv_cache, sm, M)

    q = torch.randn(num_reqs, H_kv * G, d, device=DEVICE, dtype=dtype)
    q_kv = q.view(num_reqs, H_kv, G, d).mean(dim=2)
    proj_q = torch.einsum("rhd,hcd->rhc", q_kv.float(), M.float()).to(dtype)
    seq_lens = torch.full((num_reqs,), seq_len, device=DEVICE, dtype=torch.int32)
    top_idx, _ = jssa_score_topk(proj_q, kv_cache, bt, seq_lens, n_fac, d, r, use_radix=True)
    assert top_idx.shape == (num_reqs, H_kv, n_fac)
    # distinct, in-range selections
    srt = top_idx.sort(dim=-1).values
    assert (srt[..., 1:] != srt[..., :-1]).all()
    assert (top_idx >= 0).all() and (top_idx < seq_len).all()

    scale = d**-0.5
    out = torch.empty_like(q)
    jssa_indexed_attend(q, kv_cache, bt, top_idx, head_size=d, scale=scale, out=out)

    # reference: gather + softmax attention per query head over its kv head's set
    Ksel, Vsel = jssa_gather(kv_cache, bt, top_idx, head_size=d, dtype=dtype)
    Ksel = Ksel.view(num_reqs, n_fac, H_kv, d).permute(0, 2, 1, 3).float()
    Vsel = Vsel.view(num_reqs, n_fac, H_kv, d).permute(0, 2, 1, 3).float()
    qg = q.view(num_reqs, H_kv, G, d).float()
    p = torch.softmax(torch.einsum("nhgd,nhfd->nhgf", qg, Ksel) * scale, dim=-1)
    ref = torch.einsum("nhgf,nhfd->nhgd", p, Vsel).reshape(num_reqs, H_kv * G, d)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # the gathered rows are the cached K/V of the selected positions
    r0, h0 = 0, 1
    pos = top_idx[r0, h0].long()
    torch.testing.assert_close(Ksel[r0, h0], K[r0 * seq_len + pos, h0].float())


def test_radix_short_sequence_cycles_real_tokens():
    torch.manual_seed(2)
    dtype = torch.bfloat16
    d, r, H_kv, block_size, n_fac = 128, 32, 2, 16, 64
    seq_len = 20  # < n_fac
    kv_cache, bt, sm, K, V = _paged_setup(1, H_kv, d, seq_len, block_size, 2 * d + r, dtype)
    M = _orthonormal_basis(H_kv, r, d, dtype)
    jssa_project_and_store(K, V, kv_cache, sm, M)
    proj_q = torch.randn(1, H_kv, r, device=DEVICE, dtype=dtype)
    seq_lens = torch.tensor([seq_len], device=DEVICE, dtype=torch.int32)
    top_idx, _ = jssa_score_topk(proj_q, kv_cache, bt, seq_lens, n_fac, d, r, use_radix=True)
    k_eff = top_idx.shape[-1]
    assert (top_idx >= 0).all() and (top_idx < seq_len).all()
    # every real token is selected at least once
    for h in range(H_kv):
        assert set(top_idx[0, h, :k_eff].tolist()) == set(range(seq_len))
