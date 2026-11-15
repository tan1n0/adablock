# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSSA projected score + top-B selection.

``score[r, h, t] = sum_c proj_q[r, h, c] * proj_K[r, h, t, c]`` for every cached
position t of request r and KV head h, with positions >= seq_lens[r] masked to
-inf, followed by a per-row top-B. proj_K is read either from the proj_K region
of the combined [K | V | proj_K] slot (bf16) or from the separate contiguous
FP8 projected-key cache.

The helpers accept pre-allocated output buffers so the decode path performs no
allocation inside a captured CUDA graph.
"""

import torch

import triton
import triton.language as tl

# Row tile for the contiguous proj_K score kernel (coalesced reads favor a
# large tile; tuned on B200 with r=32 fp8). The in-slot kernel uses 64.
_CONTIG_SCORE_BLOCK_T = 512


@triton.jit
def _jssa_score_kernel(
    proj_q_ptr,  # [num_reqs, H_kv, cs_h]
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens_ptr,  # [num_reqs]
    scores_ptr,  # [num_reqs, H_kv, max_kv_len]  fp32
    max_kv_len,
    head_size,
    cs_h,
    block_size,
    proj_q_stride_r,
    proj_q_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    scores_stride_r,
    scores_stride_h,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    in_range = t_offs < max_kv_len
    # Also bound by the block-table capacity (max_kv_len == max_blocks *
    # block_size) so every block-table / cache load stays in bounds.
    in_seq = (t_offs < seq_len) & in_range

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h

    pq = tl.load(
        proj_q_ptr + pid_r * proj_q_stride_r + pid_h * proj_q_stride_h + c_offs,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    bt_offset = pid_r * bt_stride_r + block_idx
    # int64 addressing: block_id * cache_stride_block exceeds int32 once the KV
    # cache holds more than ~58k blocks.
    block_id = tl.load(block_table_ptr + bt_offset, mask=in_seq, other=0).to(tl.int64)

    # proj_K region starts after K (head_size) and V (head_size).
    proj_K_base = (
        block_id * cache_stride_block
        + pos_in_block * cache_stride_pos
        + pid_h * cache_stride_head
        + 2 * head_size
    )

    proj_K = tl.load(
        kv_cache_ptr + proj_K_base[:, None] + c_offs[None, :],
        mask=in_seq[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    score = tl.sum(proj_K * pq[None, :], axis=1)  # [BLOCK_T]
    score = tl.where(in_seq, score, float("-inf"))

    out_off = pid_r * scores_stride_r + pid_h * scores_stride_h + t_offs
    tl.store(scores_ptr + out_off, score, mask=in_range)


@triton.jit
def _jssa_score_contig_kernel(
    proj_q_ptr,  # [num_reqs, H_kv, cs_h]
    projk_ptr,  # [num_blocks, block_size, H_kv, cs_h]  contiguous proj_K cache
    block_table_ptr,
    seq_lens_ptr,
    scores_ptr,
    max_kv_len,
    cs_h,
    block_size,
    proj_q_stride_r, proj_q_stride_h,
    pk_stride_block, pk_stride_pos, pk_stride_head,
    bt_stride_r,
    scores_stride_r, scores_stride_h,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Same score as _jssa_score_kernel, reading proj_K from the separate
    contiguous cache so consecutive positions are adjacent in memory."""
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    in_range = t_offs < max_kv_len
    in_seq = (t_offs < seq_len) & in_range

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h
    pq = tl.load(
        proj_q_ptr + pid_r * proj_q_stride_r + pid_h * proj_q_stride_h + c_offs,
        mask=c_mask, other=0.0,
    ).to(tl.float32)

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    block_id = tl.load(
        block_table_ptr + pid_r * bt_stride_r + block_idx, mask=in_seq, other=0
    ).to(tl.int64)
    proj_K_base = (
        block_id * pk_stride_block
        + pos_in_block * pk_stride_pos
        + pid_h * pk_stride_head
    )
    proj_K = tl.load(
        projk_ptr + proj_K_base[:, None] + c_offs[None, :],
        mask=in_seq[:, None] & c_mask[None, :], other=0.0,
    ).to(tl.float32)

    score = tl.sum(proj_K * pq[None, :], axis=1)
    score = tl.where(in_seq, score, float("-inf"))
    out_off = pid_r * scores_stride_r + pid_h * scores_stride_h + t_offs
    tl.store(scores_ptr + out_off, score, mask=in_range)


def jssa_score(
    proj_q: torch.Tensor,  # (num_reqs, H_kv, cs_h)  bf16/fp16
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks)  int32
    seq_lens: torch.Tensor,  # (num_reqs,)  int32
    head_size: int,
    cs_h: int,
    block_t: int = 64,
    scores_out: torch.Tensor | None = None,
    projk_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scores for every cached position: fp32 ``(num_reqs, H_kv, max_kv_len)``
    with positions >= seq_lens[r] set to -inf. Writes into ``scores_out`` when
    given. ``projk_cache`` routes the read to the contiguous proj_K cache."""
    num_reqs, H_kv, cs_h_q = proj_q.shape
    assert cs_h_q == cs_h, f"proj_q.shape[-1] {cs_h_q} != cs_h {cs_h}"
    block_size = kv_cache.shape[1]
    max_blocks = block_table.shape[1]
    max_kv_len = max_blocks * block_size

    if scores_out is None:
        scores = torch.empty(
            (num_reqs, H_kv, max_kv_len),
            dtype=torch.float32,
            device=proj_q.device,
        )
    else:
        assert scores_out.shape[0] >= num_reqs and scores_out.shape[1] == H_kv
        assert scores_out.shape[-1] >= max_kv_len
        scores = scores_out[:num_reqs, :, :max_kv_len]

    BLOCK_C = triton.next_power_of_2(cs_h)
    if projk_cache is not None:
        BLOCK_T = _CONTIG_SCORE_BLOCK_T
        grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))
        _jssa_score_contig_kernel[grid](
            proj_q, projk_cache, block_table, seq_lens, scores,
            max_kv_len, cs_h, block_size,
            proj_q.stride(0), proj_q.stride(1),
            projk_cache.stride(0), projk_cache.stride(1), projk_cache.stride(2),
            block_table.stride(0), scores.stride(0), scores.stride(1),
            BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C,
        )
        return scores

    BLOCK_T = block_t
    grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))
    _jssa_score_kernel[grid](
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        scores,
        max_kv_len,
        head_size,
        cs_h,
        block_size,
        proj_q.stride(0),
        proj_q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        scores.stride(0),
        scores.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
    )
    return scores


def _radix_topk_available() -> bool:
    """True iff vLLM's radix top-k binding (csrc/topk.cu, used by DeepSeek
    Sparse Attention) is compiled in. It is O(seq) vs torch.topk's full sort."""
    if not hasattr(torch.ops._C, "top_k_per_row_decode"):
        try:
            import vllm._custom_ops  # noqa: F401  (registers the torch ops)
        except Exception:
            return False
    return hasattr(torch.ops._C, "top_k_per_row_decode")


# Shape-keyed persistent scratch for the radix path: allocations inside the
# captured decode path would come from the graph pool and alias at batch > 1.
_RADIX_SCRATCH: dict = {}


def _radix_scratch(name, shape, dtype, device):
    key = (name, tuple(shape), dtype, device)
    t = _RADIX_SCRATCH.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _RADIX_SCRATCH[key] = t
    return t


def _radix_topk(
    scores: torch.Tensor,  # (num_reqs, H_kv, max_kv_len) fp32
    seq_lens: torch.Tensor,  # (num_reqs,) int32
    k: int,
    idx_out: torch.Tensor | None = None,  # (num_reqs, H_kv, k) int32
) -> torch.Tensor:
    """Radix top-k over the last dim. Returns int32 (num_reqs, H_kv, k) indices,
    not sorted by score (the attention only needs the set). Allocation-free."""
    num_reqs, H_kv, max_kv = scores.shape
    rows = num_reqs * H_kv
    dev = scores.device
    logits2d = _radix_scratch("logits", (rows, max_kv), torch.float32, dev)
    logits2d.view(num_reqs, H_kv, max_kv).copy_(scores)
    seq_lens_rows = _radix_scratch("seqrows", (rows,), torch.int32, dev)
    seq_lens_rows.view(num_reqs, H_kv).copy_(
        seq_lens.to(torch.int32).view(num_reqs, 1).expand(num_reqs, H_kv)
    )
    topk_idx = _radix_scratch("topk", (rows, k), torch.int32, dev)
    next_n = 1  # one query token per decode step
    torch.ops._C.top_k_per_row_decode(
        logits2d, next_n, seq_lens_rows, topk_idx, rows,
        logits2d.stride(0), logits2d.stride(1), k,
    )
    # When seq_len < k the kernel pads unfilled slots with -1, which would be an
    # out-of-bounds read in the attention. Replace every invalid slot (negative
    # or >= seq_len) by cycling through the real tokens [0, seq_len): the
    # attention then covers every real token with near-equal multiplicity
    # (exact when k % seq_len == 0), never an unwritten cache slot.
    res = _radix_scratch("res", (rows, k), torch.int32, dev)
    slot_off = _radix_scratch("slotoff", (k,), torch.int32, dev)
    torch.arange(k, out=slot_off)
    sl = _radix_scratch("sl", (rows, 1), torch.int32, dev)  # avoid modulo-by-zero
    torch.clamp(seq_lens_rows.view(rows, 1), min=1, out=sl)
    fill = _radix_scratch("fill", (rows, k), torch.int32, dev)
    torch.remainder(slot_off.view(1, k).expand(rows, k), sl, out=fill)
    neg = _radix_scratch("neg", (rows, k), torch.bool, dev)
    torch.lt(topk_idx, 0, out=neg)
    oob = _radix_scratch("oob", (rows, k), torch.bool, dev)
    torch.ge(topk_idx, seq_lens_rows.view(rows, 1), out=oob)
    torch.logical_or(neg, oob, out=neg)
    torch.where(neg, fill, topk_idx, out=res)
    out3d = res.view(num_reqs, H_kv, k)
    if idx_out is not None:
        idx_out[:num_reqs, :, :k].copy_(out3d)
        return idx_out[:num_reqs, :, :k]
    return out3d


def jssa_score_topk(
    proj_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_fac: int,
    head_size: int,
    cs_h: int,
    scores_out: torch.Tensor | None = None,
    top_idx_out: torch.Tensor | None = None,
    top_scores_out: torch.Tensor | None = None,
    use_radix: bool = True,
    projk_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score -> top-B. Returns ``(top_idx, top_scores)`` shaped
    ``(num_reqs, H_kv, k)`` with ``k = min(n_fac, max_kv_len)``. With
    pre-allocated buffers, ``top_idx_out`` may be int32 (radix, or torch.topk
    copied in) or int64 (torch.topk). Without buffers, int32 indices."""
    scores = jssa_score(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        head_size,
        cs_h,
        scores_out=scores_out,
        projk_cache=projk_cache,
    )
    k = min(n_fac, scores.shape[-1])
    use_radix_eff = use_radix and _radix_topk_available()

    if top_idx_out is not None and top_scores_out is not None:
        num_reqs, H_kv, _ = scores.shape
        assert top_scores_out.dtype == torch.float32
        assert (
            top_idx_out.shape[0] >= num_reqs
            and top_idx_out.shape[1] == H_kv
            and top_idx_out.shape[-1] >= k
        ), (
            f"top_idx_out shape {tuple(top_idx_out.shape)} too small for "
            f"({num_reqs}, {H_kv}, {k})"
        )
        if use_radix_eff and top_idx_out.dtype == torch.int32:
            idx_view = _radix_topk(scores, seq_lens, k, idx_out=top_idx_out)
            score_view = top_scores_out[:num_reqs, :, :k]
            torch.gather(scores, -1, idx_view.to(torch.int64), out=score_view)
            return idx_view, score_view
        if top_idx_out.dtype == torch.int32:
            # int32 buffer without the radix binding: torch.topk(out=) needs
            # int64, so compute and copy in.
            ts, ti = torch.topk(scores, k=k, dim=-1)
            idx_view = top_idx_out[:num_reqs, :, :k]
            score_view = top_scores_out[:num_reqs, :, :k]
            idx_view.copy_(ti.to(torch.int32))
            score_view.copy_(ts)
            return idx_view, score_view
        assert top_idx_out.dtype == torch.int64, (
            "top_idx_out must be int32 or int64"
        )
        idx_view = top_idx_out[:num_reqs, :, :k]
        score_view = top_scores_out[:num_reqs, :, :k]
        torch.topk(scores, k=k, dim=-1, out=(score_view, idx_view))
        return idx_view, score_view

    if use_radix_eff:
        top_idx = _radix_topk(scores, seq_lens, k)
        top_scores = torch.gather(scores, -1, top_idx.to(torch.int64))
        return top_idx, top_scores

    top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
    return top_idx.to(torch.int32), top_scores
