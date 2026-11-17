"""Fixed-budget full ranking versus coarse-to-fine ranking on paired 32K RULER inputs.

This uses the no-sink/no-local fixed-budget inputs from
adablock_ruler_3090_free_budget_20261113_full_01. All ranking variants share
GPU KV gather and PyTorch SDPA. It does not measure autonomous S/B or reproduce
JSSA's indexed-attention timing. Run from the AdaBlock repository root.
"""

from __future__ import annotations

import argparse
import importlib.util
import gc
import hashlib
import json
import re
import random
import statistics
import sys
import time
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01"
DEFAULT_RANKER = REPO / "results/s_no_fixed_20261116/ranker/best.pt"
DEFAULT_RULER = REPO / "experiments/runs/adablock_ruler_harder_20261022_01"
DEFAULT_DIGEST = REPO / "experiments/server_snapshots/ruler_autonomous_20261022_01"
EXPECTED_SELECTED_SHA256 = "35eb364d354c7230c57282d3d5cbbe906941495c57946abd6597c9aac1c8f0ec"
EXPECTED_RANKER_SHA256 = "4184515e1a289a837ee302847d6f30078942110d69031b23e043682e46e6ac4a"


def method_names(args) -> list[str]:
    available = (["k_mass", "k_value", "kv_value"] if args.new_value_models else
                 ["mlp", "cobs", "diag_cobs", "dual32", "dual64"])
    if args.methods:
        if len(set(args.methods)) != len(args.methods) or not set(args.methods).issubset(available):
            raise ValueError(f"Methods must be unique and drawn from {available}")
        return args.methods
    return available


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def stable_topk_indices(scores, count, torch):
    """Top-K with the same score-descending, low-index-first ties as stable sort."""
    n = scores.shape[-1]
    # Map IEEE float32 bit patterns to monotonically ordered unsigned integers.
    # The secondary key prefers the original lower candidate index on ties.
    bits = scores.float().contiguous().view(torch.int32)
    ordered = torch.where(bits < 0, (~bits).long() & 0xffffffff,
                          bits.long() | 0x80000000)
    ordered = torch.where(scores == 0, 0x80000000, ordered)
    ids = torch.arange(n, device=scores.device, dtype=torch.long)
    keys = ordered * (n + 1) + (n - 1 - ids)
    return keys.topk(count, dim=-1, sorted=True).indices


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def config(args: argparse.Namespace) -> dict:
    selected = args.source_run / "selected.json"
    return {
        "selected": str(selected.resolve()),
        "selected_sha256": digest(selected),
        "source_run": str(args.source_run.resolve()),
        "ranker_checkpoint": str(args.ranker_checkpoint.resolve()),
        "ranker_sha256": digest(args.ranker_checkpoint),
        "ruler_dir": str(args.ruler_dir.resolve()),
        "digest_dir": str(args.digest_dir.resolve()),
        "model": str(args.model.resolve()),
        "budget_history_tokens": args.budget,
        "block_size": 16,
        "coarse_candidates": args.coarse_m,
        "methods": method_names(args),
        "topk_tie_break": ("unspecified for equal scores; torch.topk" if args.fast_topk else
                            "score descending, original candidate index ascending"),
        "fixed": "current visible incomplete block only; no sink/local",
        "attention": "same GPU K/V gather and PyTorch SDPA with GQA for all methods",
        "quality": "natural greedy decode, shared full prefill and first token, official task score",
        "audit": "same initial query on first decode step; selected-block set overlap",
        "timing": "same forced 16 input tokens, three unprofiled repeats and separate CUDA-event pass",
        "scope": ("trained exact K/K+V decode selectors" if args.new_value_models else
                  "frozen S MLP and K-summary selectors, same sparse backend"),
        "cobs_execution": "batched_all_kv_heads",
        "diag_cobs_execution": "one_batched_mean_variance_gemm",
        "gqa_backend": "native_sdpa" if args.native_gqa else "explicit_kv_expansion",
        "graph_score": args.graph_score,
        "shared_graph": args.shared_graph,
        "graph_build_scope": ("captured after index build, included in index_build_seconds"
                              if args.graph_score else None),
        "selector_dir": str(args.selector_dir.resolve()),
        "selector_checkpoint_sha256": {
            name: digest(args.diag_checkpoint if name == "diag_cobs" else
                         args.selector_dir / f"{name}_best.pt")
            for name in method_names(args) if name != "mlp"},
        "topk": "unsorted torch.topk" if args.fast_topk else "stable bit-key topk",
        "variance": ("exact population K/V diagonal variance" if args.new_value_models else
                     "diagonal of rank-4 covariance, matching pilot training"),
        "index_build_scope": "each method builds an isolated index and reports its own index bytes, build time, and peak allocated/reserved GPU memory",
        "online_timing_scope": "same sparse KV gather and SDPA within the run; online scores are method-specific",
    }


def check_inputs(args: argparse.Namespace) -> list[dict]:
    if args.budget <= 0 or args.budget % 16:
        raise ValueError("--budget must be a positive multiple of 16")
    if args.coarse_m < args.budget // 16:
        raise ValueError("--coarse-m must be at least budget/16")
    if args.limit < 1 or args.limit > 100:
        raise ValueError("--limit must be between 1 and 100")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard ID")
    needed = [
        args.source_run / "selected.json",
        args.source_run / "quality_000.json",
        args.ranker_checkpoint,
        *(args.diag_checkpoint if name == "diag_cobs" else args.selector_dir / f"{name}_best.pt"
          for name in method_names(args) if name != "mlp"),
        args.ruler_dir / "RULER/scripts/synthetic.yaml",
        args.ruler_dir / "RULER/scripts/eval/synthetic/constants.py",
        args.digest_dir / "cache_buffer.py",
        args.digest_dir / "fast_digest.py",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing: " + ", ".join(missing))
    if digest(args.source_run / "selected.json") != EXPECTED_SELECTED_SHA256:
        raise ValueError("Selected input hash differs from the completed 100-case run")
    if args.ranker_checkpoint.resolve() == DEFAULT_RANKER.resolve():
        if digest(args.ranker_checkpoint) != EXPECTED_RANKER_SHA256:
            raise ValueError("B checkpoint hash differs from the completed 100-case run")
    else:
        import torch
        payload = torch.load(args.ranker_checkpoint, map_location="cpu", weights_only=True)
        meta = payload.get("data_metadata", {})
        if payload.get("format") != "adablock_ranker_v1" or (meta.get("sink_blocks"), meta.get("local_window_blocks")) != (0, 0):
            raise ValueError("Custom checkpoint must be a no-sink/no-local AdaBlock ranker")
    if args.model is None or not (args.model / "config.json").is_file():
        raise FileNotFoundError("Pass --model with the local Llama-3.1-8B-Instruct weights directory")
    rows = json.loads((args.source_run / "selected.json").read_text())
    if len(rows) < args.limit:
        raise ValueError(f"Selected input has only {len(rows)} rows")
    for index in range(args.limit):
        path = args.source_run / f"quality_{index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        old = json.loads(path.read_text())
        if rows[index]["input_sha256"] != old["input_sha256"]:
            raise ValueError(f"Source hash mismatch at case {index}")
        if rows[index]["task"] != old["task"]:
            raise ValueError(f"Source task mismatch at case {index}")
    return rows


def prepare_output(args: argparse.Namespace) -> None:
    protocol = config(args)
    target = args.output / "protocol.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous != protocol:
            if any(args.output.glob("case_*.json")):
                raise ValueError("Output protocol differs from completed cases; choose a new --output")
            write_json(target, protocol)  # Reuse a directory containing only a failed preflight.
    else:
        write_json(target, protocol)


def run(args: argparse.Namespace) -> None:
    rows = check_inputs(args)
    prepare_output(args)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(args.digest_dir))
    from cache_buffer import BufferCache, reset_registry
    from fast_digest import compact_digest
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from models.block_ranker import BlockRanker, BlockRankerConfig
    from models.cacheable_selectors import (SimpleCOBS, DiagonalCOBS, DualEncoder, DualEncoderConfig, BlockEmbeddingCache,
                                             rank4_diagonal_summary, exact_block_summary, group_probabilities)
    from utils.ranking_batch import forward_packed

    scorer_spec = importlib.util.spec_from_file_location("ruler_official_local", REPO / "experiments/server_snapshots/ruler_harder_20261022_01/official.py")
    official = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(official)
    official.PACKAGE = args.ruler_dir

    @torch.inference_mode()
    def prefill(model, ids):
        reset_registry()
        cache = BufferCache(len(ids) + 256)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(ids), 1024):
            output = model.model(input_ids=torch.tensor([ids[offset:offset + 1024]], device="cuda"),
                                 past_key_values=cache, use_cache=True)
        logits = model.lm_head(output.last_hidden_state[:, -1:])[0, -1].float()
        torch.cuda.synchronize()
        return tuple(cache.to_legacy_cache()), logits, time.perf_counter() - start

    class Meter:
        def __init__(self, enabled: bool = False):
            self.enabled = enabled
            self.events = []

        def call(self, name, function):
            if not self.enabled:
                return function()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = function()
            end.record()
            self.events.append((name, start, end))
            return output

        def totals(self, steps):
            values = {}
            for name, start, end in self.events:
                values[name] = values.get(name, 0.0) + start.elapsed_time(end) / steps
            return values

    class DiagScoreGraph:
        """Static masked scan/top-32 over cache capacity; complete count is dynamic."""
        def __init__(self, selector, summary, position, complete, copy_summary=False):
            self.copy_summary = copy_summary
            self.query = torch.zeros((summary.shape[0], 4, summary.shape[-1]//2),
                                     device=summary.device, dtype=torch.float32)
            self.position = torch.tensor(float(position), device=summary.device)
            self.complete = torch.tensor(complete, device=summary.device)
            self.ids = torch.arange(summary.shape[1],device=summary.device).expand(summary.shape[0],-1)
            self.counts = torch.full_like(self.ids,16,dtype=torch.float32)
            self.summary = torch.zeros_like(summary) if copy_summary else summary
            self.selector = selector
            def compute():
                logits = selector(self.query,self.summary,self.ids,self.counts,self.position)
                logits = logits.masked_fill(self.ids[:,None,:]>=self.complete,float('-inf'))
                scores = group_probabilities(logits)
                return scores.topk(32,dim=-1,sorted=False).indices
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3): self.output=compute()
            torch.cuda.current_stream().wait_stream(warmup)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output=compute()

        def run(self, query, position, complete, summary=None):
            if self.copy_summary:
                self.summary.copy_(summary)
            self.query.copy_(query)
            self.position.fill_(position)
            self.complete.fill_(complete)
            self.graph.replay()
            return self.output

    class CobsScoreGraph:
        """Captured rank-4 COBS, GQA normalization and top-32 for one layer."""
        def __init__(self, selector, data, heads, group, dim, position, complete):
            capacity=data["mean"].shape[1]
            self.query=torch.zeros((heads,group,dim),device="cuda",dtype=torch.float32)
            self.position=torch.tensor(float(position),device="cuda")
            self.complete=torch.tensor(complete,device="cuda")
            self.ids=torch.arange(capacity,device="cuda").expand(heads,-1)
            self.data=data
            def compute():
                logits=selector(self.query,data["mean"],data["basis"],
                                data["eigenvalues"],self.ids,
                                data["counts"].expand(heads,-1),self.position)
                logits=logits.masked_fill(self.ids[:,None,:]>=self.complete,float('-inf'))
                scores=group_probabilities(logits)
                return scores.topk(32,dim=-1,sorted=False).indices
            warmup=torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):self.output=compute()
            torch.cuda.current_stream().wait_stream(warmup)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=compute()

        def run(self,query,position,complete):
            self.query.copy_(query)
            self.position.fill_(position)
            self.complete.fill_(complete)
            self.graph.replay()
            return self.output

    class Sparse:
        def __init__(self, model, cpu_pairs, capacity):
            self.model = model
            self.prefix = cpu_pairs[0][0].shape[2]
            self.h = model.config.num_key_value_heads
            self.d = model.config.hidden_size // model.config.num_attention_heads
            self.g = model.config.num_attention_heads // self.h
            self.cache = []
            for key, value in cpu_pairs:
                key_buffer = torch.empty((self.h, capacity, self.d), device="cuda", dtype=torch.bfloat16)
                value_buffer = torch.empty_like(key_buffer)
                key_buffer[:, :self.prefix].copy_(key[0].to("cuda"))
                value_buffer[:, :self.prefix].copy_(value[0].to("cuda"))
                self.cache.append((key_buffer, value_buffer))
            self.original = []
            self.offsets = torch.arange(16, device="cuda")
            self.heads = torch.arange(self.h, device="cuda")
            self.capture_selection = False
            self.selected_blocks = []

        def install(self):
            for index, layer in enumerate(self.model.model.layers):
                module = layer.self_attn
                self.original.append(module.forward)
                def forward(module, hidden_states, position_embeddings, _layer=index, **kwargs):
                    return self.forward(module, _layer, hidden_states, position_embeddings)
                module.forward = types.MethodType(forward, module)

        def uninstall(self):
            for layer, original in zip(self.model.model.layers, self.original):
                layer.self_attn.forward = original
            self.original = []

        def init_digest(self, mode):
            self.mode = mode
            n = self.prefix // 16
            self.digests = []
            is_diag = mode == "diag_cobs"
            is_dual = mode not in ("mlp", "cobs", "diag_cobs")
            self.embeddings = {mode: []} if is_dual else {}
            self.diag_summaries = []
            capacity_blocks = (self.cache[0][0].shape[1] + 15) // 16
            for key, value in self.cache:
                buffers = {}
                if is_diag or (is_dual and args.new_value_models):
                    counts = torch.full((capacity_blocks,), 16., device="cuda")
                    buffers["counts"] = counts
                    kblocks = key[:, :n*16].reshape(self.h,n,16,self.d)
                    vblocks = value[:, :n*16].reshape(self.h,n,16,self.d)
                    summary = exact_block_summary(kblocks, vblocks if mode == "kv_value" else None)
                else:
                    source = compact_digest(key[:, :n * 16])
                    fields = ("counts",) if is_dual else tuple(source)
                    for name in fields:
                        tensor = source[name]
                        axis = 0 if name == "counts" else 1
                        shape = list(tensor.shape)
                        shape[axis] = capacity_blocks
                        dest = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
                        dest.narrow(axis, 0, n).copy_(tensor)
                        buffers[name] = dest
                    if is_dual:
                        summary = rank4_diagonal_summary(source["mean"], source["basis"],
                                                          source["eigenvalues"])
                self.digests.append(buffers)
                if is_diag:
                    packed = torch.zeros((self.h,capacity_blocks,2*self.d),device="cuda",dtype=torch.float16)
                    packed[:,:n].copy_(summary.half())
                    self.diag_summaries.append(packed)
                if is_dual:
                    cache = BlockEmbeddingCache(selectors[mode], capacity_blocks, self.h,
                                                torch.device("cuda"), torch.float16)
                    cache.append_summary(summary)
                    self.embeddings[mode].append(cache)
            self.graphs = []
            if args.graph_score and mode == "diag_cobs":
                if args.shared_graph:
                    shared = DiagScoreGraph(selectors["diag_cobs"],self.diag_summaries[0],
                                            self.prefix,n,copy_summary=True)
                    self.graphs = [shared]*len(self.diag_summaries)
                else:
                    self.graphs = [DiagScoreGraph(selectors["diag_cobs"],item,self.prefix,n)
                                   for item in self.diag_summaries]
            elif args.graph_score and mode == "cobs":
                self.graphs = [CobsScoreGraph(selectors["cobs"],data,self.h,self.g,self.d,
                                              self.prefix,n) for data in self.digests]

        def index_bytes(self):
            stored = sum(t.numel()*t.element_size() for layer in self.digests
                         for t in layer.values())
            stored += sum(cache.data.numel()*cache.data.element_size()
                          for caches in self.embeddings.values() for cache in caches)
            stored += sum(t.numel()*t.element_size() for t in self.diag_summaries)
            return stored

        def update(self, layer, key, value):
            key_buffer, value_buffer = self.cache[layer]
            key_buffer[:, self.position].copy_(key[0])
            value_buffer[:, self.position].copy_(value[0])

        def update_digest(self, layer, block):
            key_buffer, value_buffer = self.cache[layer]
            if self.mode == "diag_cobs" or (self.mode not in ("mlp", "cobs") and args.new_value_models):
                k = key_buffer[:, block*16:(block+1)*16].reshape(self.h,1,16,self.d)
                v = value_buffer[:, block*16:(block+1)*16].reshape(self.h,1,16,self.d)
                summary = exact_block_summary(k, v if self.mode == "kv_value" else None)
                if self.mode == "diag_cobs":
                    self.diag_summaries[layer][:,block:block+1].copy_(summary.half())
                else:
                    self.embeddings[self.mode][layer].write_summary(block,summary)
                return
            source = compact_digest(key_buffer[:, block * 16:(block + 1) * 16])
            for name in self.digests[layer]:
                tensor = source[name]
                axis = 0 if name == "counts" else 1
                self.digests[layer][name].narrow(axis, block, 1).copy_(tensor)
            if self.mode not in ("mlp", "cobs", "diag_cobs"):
                summary = rank4_diagonal_summary(source["mean"], source["basis"],
                                                  source["eigenvalues"])
                self.embeddings[self.mode][layer].write_summary(block,summary)

        def choose(self, layer, q, complete, scale):
            grouped = q.view(self.h, self.g, self.d).float() * scale
            data = self.digests[layer]
            if self.mode in ("diag_cobs","cobs") and args.graph_score:
                blocks = self.meter.call("graph_score_topk", lambda: (
                    self.graphs[layer].run(grouped,self.position,complete,self.diag_summaries[layer])
                    if self.mode == "diag_cobs" else
                    self.graphs[layer].run(grouped,self.position,complete)))
                if self.capture_selection:
                    self.selected_blocks.append(blocks.detach().cpu().tolist())
                return (blocks[...,None]*16+self.offsets).view(1,self.h,-1)
            ids = torch.arange(complete, device="cuda").expand(self.h, -1)
            count = complete
            if self.mode == "mlp":
                def fine_score():
                    packed = {name: data[name][self.heads[:, None], ids].flatten(0, 1)
                              for name in ("mean", "basis", "eigenvalues")}
                    packed.update(query=grouped, counts=data["counts"][ids].reshape(-1),
                                  block_ids=ids.flatten(), scene_ids=self.heads.repeat_interleave(count),
                                  token_position=torch.full((self.h,), self.position, device="cuda"),
                                  layer_id=torch.full((self.h,), layer, device="cuda"), kv_head_id=self.heads,
                                  block_size=torch.full((self.h,), 16, device="cuda"))
                    return forward_packed(self.ranker, packed).view(self.h, count)
                scores = self.meter.call("ranker_score", fine_score)
            elif self.mode == "diag_cobs":
                def diag_score():
                    logits = selectors["diag_cobs"](grouped,self.diag_summaries[layer][:,:complete],
                         ids,data["counts"][:complete].expand(self.h,-1),self.position)
                    return group_probabilities(logits)
                scores = self.meter.call("online_scan_norm",diag_score)
            elif self.mode == "cobs":
                def cobs_score():
                    logits = selectors["cobs"](
                        grouped, data["mean"][:, :complete], data["basis"][:, :complete],
                        data["eigenvalues"][:, :complete], ids,
                        data["counts"][:complete].expand(self.h, -1), self.position)
                    return group_probabilities(logits)
                scores = self.meter.call("online_scan_norm", cobs_score)
            else:
                selector = selectors[self.mode]
                encoded = self.meter.call("query_encode", lambda: selector.encode_query(grouped))
                def dual_score():
                    logits = selector.score_encoded(encoded, self.embeddings[self.mode][layer].view()[:, :complete],
                                                    ids, data["counts"][:complete].expand(self.h,-1),
                                                    self.position)
                    return group_probabilities(logits)
                scores = self.meter.call("online_scan_norm", dual_score)
            take = self.budget // 16
            order = self.meter.call("sort_or_topk", lambda: (
                scores.topk(take,dim=-1,sorted=False).indices if args.fast_topk else
                stable_topk_indices(scores,take,torch)))
            blocks = ids.gather(1, order)
            if self.capture_selection:
                self.selected_blocks.append(blocks.detach().cpu().tolist())
            return (blocks[..., None] * 16 + self.offsets).view(1, self.h, -1)

        def forward(self, module, layer, hidden, embeddings):
            shape = (1, 1, -1, self.d)
            q = module.q_proj(hidden).view(shape).transpose(1, 2)
            key = module.k_proj(hidden).view(shape).transpose(1, 2)
            value = module.v_proj(hidden).view(shape).transpose(1, 2)
            q, key = apply_rotary_pos_emb(q, key, *embeddings)
            q = q[0, :, 0].unsqueeze(0).contiguous()
            self.meter.call("cache_update", lambda: self.update(layer, key[0, :, 0].unsqueeze(0).contiguous(),
                                                                 value[0, :, 0].unsqueeze(0).contiguous()))
            complete, offset = divmod(self.position, 16)
            history = self.choose(layer, q, complete, module.scaling)
            tail = (complete * 16 + self.offsets[:offset + 1]).view(1, 1, -1).expand(1, self.h, -1)
            indices = torch.cat((history, tail), dim=-1).long()
            def gather_kv():
                key_buffer, value_buffer = self.cache[layer]
                gather_index = indices[0, :, :, None].expand(self.h, indices.shape[-1], self.d)
                return (key_buffer.gather(1, gather_index)[None],
                        value_buffer.gather(1, gather_index)[None])
            selected_key, selected_value = self.meter.call("kv_gather", gather_kv)
            def attend():
                if args.native_gqa:
                    return F.scaled_dot_product_attention(
                        q.unsqueeze(2).contiguous(),selected_key.contiguous(),selected_value.contiguous(),
                        dropout_p=0.0,is_causal=False,scale=float(module.scaling),enable_gqa=True)
                return F.scaled_dot_product_attention(
                    q.unsqueeze(2).contiguous(),
                    selected_key.repeat_interleave(self.g, dim=1).contiguous(),
                    selected_value.repeat_interleave(self.g, dim=1).contiguous(),
                    dropout_p=0.0, is_causal=False, scale=float(module.scaling))
            output = self.meter.call("sparse_sdpa", attend)
            if offset == 15:
                self.meter.call("digest_update", lambda: self.update_digest(layer, complete))
            return module.o_proj(output.reshape(1, 1, -1)), None

    def clear():
        reset_registry()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def decode(model, first, eos, engine, mode, budget, cap=128, forced=None, capture=False, profile=False):
        engine.mode = mode
        engine.budget = budget
        engine.ranker = ranker
        engine.meter = Meter(profile)
        engine.capture_selection = capture
        engine.selected_blocks = []
        prompt_len = engine.prefix
        ids = [int(first.argmax())]
        token = torch.empty((1, 1), device="cuda", dtype=torch.long)
        position = torch.empty((1,), device="cuda", dtype=torch.long)
        engine.install()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            for step in range(len(forced) if forced is not None else cap - 1):
                if forced is None and ids[-1] in eos:
                    break
                p = prompt_len + step
                token.fill_(forced[step] if forced is not None else ids[-1])
                position.fill_(p)
                engine.position = p
                output = model.model(input_ids=token, position_ids=position[None], cache_position=position,
                                     past_key_values=None, use_cache=False)
                logits = model.lm_head(output.last_hidden_state[:, -1:])[0, -1].float()
                ids.append(int(logits.argmax()))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            stages = engine.meter.totals(max(1, len(ids) - 1))
        finally:
            engine.uninstall()
        return {
            "ids": ids,
            "decode_steps": len(ids) - 1,
            "decode_seconds": seconds,
            "cap_reached": forced is None and len(ids) == cap and ids[-1] not in eos,
            "stages_ms_per_step": stages,
            "selected_blocks": engine.selected_blocks if capture else None,
        }

    def compare_blocks(left, right):
        assert len(left) == len(right)
        total = len(left) * len(left[0])
        equal = 0
        intersection = 0
        selected = 0
        for a, b in zip(left, right):
            for aa, bb in zip(a, b):
                sa, sb = set(aa), set(bb)
                equal += sa == sb
                intersection += len(sa & sb)
                selected += len(sa)
        return {"equal_heads": equal, "total_heads": total,
                "mean_recall_of_reference": intersection / selected}

    def evaluate_one(index, row, old, model, tokenizer, eos):
        clear()
        prefix, first, prefill_seconds = prefill(model, row["input_ids"])
        if int(first.argmax()) != old["methods"]["full"]["ids"][0]:
            raise ValueError(f"Full first-token mismatch on case {index}")
        cpu_prefix = [(key.cpu(), value.cpu()) for key, value in prefix]
        del prefix
        clear()
        names = method_names(args)
        rotation = index % len(names)
        order = names[rotation:] + names[:rotation]
        case = {"index": index, "task": row["task"], "input_sha256": row["input_sha256"],
                "prompt_tokens": len(row["input_ids"]), "prefill_seconds": prefill_seconds,
                "budget": args.budget, "methods": {},
                "reference_full_score": old["methods"]["full"].get("ruler_score")}
        audited = {}
        profile = {}
        for name in order:
            clear()
            torch.cuda.reset_peak_memory_stats()
            engine = Sparse(model, cpu_prefix, len(row["input_ids"]) + 128)
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.init_digest(name)
            torch.cuda.synchronize()
            build_seconds = time.perf_counter() - start
            index_mib = engine.index_bytes() / 1048576
            result = decode(model, first, eos, engine, name, args.budget)
            prediction = official.normalize_prediction(
                tokenizer.decode(result["ids"], skip_special_tokens=True), row["task"])
            result["prediction"] = prediction
            result["ruler_score"] = official.metric_for(row["task"])([prediction], [row["answers"]])
            result["hits"] = sum(bool(re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)",
                                                prediction, re.I)) for answer in row["answers"])
            result["all_hit"] = result["hits"] == len(row["answers"])
            result["index_build_seconds"] = build_seconds
            result["index_cache_mib"] = index_mib
            result["selected_blocks"] = None
            case["methods"][name] = result
            if index < args.audit_count:
                first_token = int(old["methods"]["full"]["ids"][0])
                audited[name] = decode(model, first, eos, engine, name, args.budget,
                                       forced=[first_token], capture=True)["selected_blocks"]
            if index < args.profile_count:
                old_ids = old["methods"]["full"]["ids"]
                forced = (old_ids[:-1] + row["input_ids"][-16:])[:16]
                decode(model, first, eos, engine, name, args.budget, forced=forced[:3])
                samples = [1000 * decode(model, first, eos, engine, name, args.budget,
                                         forced=forced)["decode_seconds"] / 16 for _ in range(3)]
                events = decode(model, first, eos, engine, name, args.budget,
                                forced=forced, profile=True)
                profile[name] = {"forced_ms_per_step": samples,
                                 "median_forced_ms_per_step": statistics.median(samples),
                                 "gpu_event_ms_per_step": events["stages_ms_per_step"]}
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1048576
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1048576
            del engine
            clear()
        if audited:
            case["selection_audit_first_step"] = {
                name: compare_blocks(audited[names[0]], audited[name]) for name in names[1:]
            }
        if profile:
            case["profile"] = profile
        del cpu_prefix
        clear()
        return case

    with torch.inference_mode():
        payload = torch.load(args.ranker_checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format") == "adablock_ranker_v1":
            ranker_config, ranker_state = payload["config"], payload["state_dict"]
        else:
            ranker_config, ranker_state = payload["ranker_config"], payload["ranker_state"]
        ranker = BlockRanker(BlockRankerConfig(**ranker_config)).to("cuda")
        ranker.load_state_dict(ranker_state)
        ranker.eval().requires_grad_(False)
        selectors = {}
        for name in method_names(args):
            if name == "mlp":
                continue
            checkpoint_path = args.diag_checkpoint if name == "diag_cobs" else args.selector_dir / f"{name}_best.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if name == "diag_cobs":
                if payload["model"] != "diag_cobs" or payload["input_summary"] != "exact K population mean and diagonal variance":
                    raise ValueError("Diagonal COBS checkpoint mismatch")
                selector = DiagonalCOBS()
            elif args.new_value_models:
                if payload["name"] != name or payload["input_summary"] != "exact K/V population mean and diagonal variance":
                    raise ValueError(f"Value selector checkpoint mismatch: {name}")
                selector = DualEncoder(DualEncoderConfig(**payload["config"]))
            else:
                if payload["model"] != name or payload["protocol"]["variance"] != "diagonal of rank-4 covariance approximation; same at train and inference":
                    raise ValueError(f"Selector checkpoint protocol mismatch: {name}")
                selector = SimpleCOBS() if name == "cobs" else DualEncoder(DualEncoderConfig(**payload["config"]))
            selector.load_state_dict(payload["state_dict"])
            selectors[name] = selector.to("cuda").eval().requires_grad_(False)
        model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        if model_config.rope_scaling and "type" not in model_config.rope_scaling:
            # Transformers 4.43 eagerly reads the legacy key even with rope_type.
            model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=model_config, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
            local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        eos_value = model.generation_config.eos_token_id
        eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
        for index in range(args.shard_id, args.limit, args.num_shards):
            if index < args.start_index:
                continue
            target = args.output / f"case_{index:03d}.json"
            if target.exists():
                print(f"skip {index}", flush=True)
                continue
            old = json.loads((args.source_run / f"quality_{index:03d}.json").read_text())
            case = evaluate_one(index, rows[index], old, model, tokenizer, eos)
            write_json(target, case)
            print(f"complete {index} {case['task']}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    protocol = json.loads((args.output / "protocol.json").read_text())
    names = protocol["methods"]
    missing = [index for index in range(args.limit)
               if not (args.output / f"case_{index:03d}.json").is_file()]
    if missing:
        raise SystemExit(f"Cannot summarize: {len(missing)} cases are incomplete; first missing case is {missing[0]:03d}")
    cases = [json.loads((args.output / f"case_{index:03d}.json").read_text())
             for index in range(args.limit)]
    output = {"n": len(cases), "budget": protocol["budget_history_tokens"],
              "coarse_m": protocol["coarse_candidates"], "groups": {}}
    def paired_interval(values):
        if not values:
            return None
        generator = random.Random(20261115)
        means = []
        for _ in range(2000):
            means.append(statistics.mean(values[generator.randrange(len(values))]
                                         for _ in values))
        means.sort()
        return {"mean": statistics.mean(values),
                "bootstrap_95pct": [means[49], means[1949]]}
    for group in ["all", "niah_multivalue", "vt"]:
        selected = [case for case in cases if group == "all" or case["task"] == group]
        if not selected:
            continue
        results = {}
        for name in names:
            rows = [case["methods"][name] for case in selected]
            steps = sum(row["decode_steps"] for row in rows)
            results[name] = {
                "score": statistics.mean(row["ruler_score"] for row in rows),
                "all_hit": sum(row["all_hit"] for row in rows),
                "cap_reached": sum(row["cap_reached"] for row in rows),
                "weighted_decode_ms_per_step": 1000 * sum(row["decode_seconds"] for row in rows) / max(steps, 1),
                "mean_index_build_seconds": statistics.mean(row["index_build_seconds"] for row in rows),
                "mean_index_cache_mib": statistics.mean(row["index_cache_mib"] for row in rows),
                "median_peak_allocated_mib": statistics.median(row["peak_allocated_mib"] for row in rows),
                "median_peak_reserved_mib": statistics.median(row["peak_reserved_mib"] for row in rows),
            }
            profiles = [case["profile"][name] for case in selected if "profile" in case]
            if profiles:
                results[name]["profile_n"] = len(profiles)
                results[name]["median_forced_ms_per_step"] = statistics.median(
                    item["median_forced_ms_per_step"] for item in profiles)
                results[name]["gpu_event_ms_per_step"] = {
                    stage: statistics.mean(item["gpu_event_ms_per_step"].get(stage, 0) for item in profiles)
                    for stage in sorted(set().union(*(item["gpu_event_ms_per_step"] for item in profiles)))
                }
        audits = [case["selection_audit_first_step"] for case in selected if "selection_audit_first_step" in case]
        overlap = {}
        for name in names[1:]:
            if audits:
                overlap[name] = {
                    "audited_cases": len(audits),
                    "equal_head_fraction": sum(item[name]["equal_heads"] for item in audits) /
                    sum(item[name]["total_heads"] for item in audits),
                    "mean_recall_of_reference": statistics.mean(
                        item[name].get("mean_recall_of_reference", item[name].get("mean_recall_of_mlp")) for item in audits),
                }
        pairs = {}
        possible = ((("k_mass", "k_value"), ("k_value", "kv_value"),
                     ("k_mass", "kv_value")) if args.new_value_models else
                    (("mlp", "cobs"), ("mlp", "diag_cobs"), ("cobs", "diag_cobs"),
                     ("mlp", "dual32"), ("mlp", "dual64"), ("dual32", "dual64")))
        comparisons = [(a,b) for a,b in possible if a in names and b in names]
        for left, right in comparisons:
            differences = [case["methods"][right]["ruler_score"] - case["methods"][left]["ruler_score"]
                           for case in selected]
            pairs[f"{right}_minus_{left}"] = {
                "score_points": paired_interval(differences),
                "all_hit_fraction": paired_interval([
                    float(case["methods"][right]["all_hit"]) -
                    float(case["methods"][left]["all_hit"]) for case in selected]),
            }
        output["groups"][group] = {"n": len(selected), "methods": results,
                                     "paired_differences": pairs,
                                     "selection_audit_first_step": overlap}
    write_json(args.output / f"summary_{args.limit}.json", output)
    print(json.dumps(output["groups"]["all"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "summarize"))
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ranker-checkpoint", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--selector-dir", type=Path, default=REPO / "results/value_aware_selector_20261116/pilot_cpu_1536")
    parser.add_argument("--new-value-models", action="store_true")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--diag-checkpoint", type=Path, default=REPO / "results/value_aware_selector_20261116/diag_cobs_training/diag_cobs_best.pt")
    parser.add_argument("--fast-topk", action="store_true")
    parser.add_argument("--native-gqa", action="store_true")
    parser.add_argument("--graph-score", action="store_true")
    parser.add_argument("--shared-graph", action="store_true")
    parser.add_argument("--ruler-dir", type=Path, default=DEFAULT_RULER)
    parser.add_argument("--digest-dir", type=Path, default=DEFAULT_DIGEST)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO / "results/value_aware_selector_20261116/selector_free")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--coarse-m", type=int, default=256)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--audit-count", type=int, default=20)
    parser.add_argument("--profile-count", type=int, default=10)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.shared_graph and not args.graph_score:
        parser.error("--shared-graph requires --graph-score")
    if args.graph_score and (not args.fast_topk or method_names(args) not in (["diag_cobs"],["cobs"])):
        parser.error("--graph-score requires --fast-topk and exactly one COBS method")
    if args.command == "check":
        rows = check_inputs(args)
        print(json.dumps({"ready_cases": args.limit, "available_cases": len(rows),
                          "source": str(args.source_run), "budget": args.budget,
                          "coarse_m": args.coarse_m}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
