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
DEFAULT_RANKER = REPO / "experiments/runs/adablock_sab90_20261019_01/branches/B/best_pair.pt"
DEFAULT_RULER = REPO / "experiments/runs/adablock_ruler_harder_20261022_01"
DEFAULT_DIGEST = REPO / "experiments/server_snapshots/ruler_autonomous_20261022_01"
EXPECTED_SELECTED_SHA256 = "35eb364d354c7230c57282d3d5cbbe906941495c57946abd6597c9aac1c8f0ec"
EXPECTED_RANKER_SHA256 = "01916c5cb0f17db8d79d8d800fabaff235a70ce5187d95ca772007911384b43f"


def method_names(args) -> list[str]:
    return ["full_sort"] if args.only_full_sort else ["full_sort", "full_topk", f"coarse_{args.coarse_m}"]


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
        "topk_tie_break": "score descending, original candidate index ascending",
        "fixed": "current visible incomplete block only; no sink/local",
        "attention": "same GPU K/V gather and PyTorch SDPA with GQA for all methods",
        "quality": "natural greedy decode, shared full prefill and first token, official task score",
        "audit": "same initial query on first decode step; selected-block set overlap",
        "timing": "same forced 16 input tokens, three unprofiled repeats and separate CUDA-event pass",
        "scope": ("frozen S no-fixed ranker; fixed budget" if args.only_full_sort else
                  "frozen B ranker; fixed budget; not autonomous S/B"),
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from models.block_ranker import BlockRanker, BlockRankerConfig
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

        def init_digest(self):
            n = self.prefix // 16
            self.digests = []
            for key, _ in self.cache:
                source = compact_digest(key[:, :n * 16])
                buffers = {}
                for name, tensor in source.items():
                    axis = 0 if name == "counts" else 1
                    shape = list(tensor.shape)
                    shape[axis] = (key.shape[1] + 15) // 16
                    dest = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
                    dest.narrow(axis, 0, n).copy_(tensor)
                    buffers[name] = dest
                self.digests.append(buffers)

        def update(self, layer, key, value):
            key_buffer, value_buffer = self.cache[layer]
            key_buffer[:, self.position].copy_(key[0])
            value_buffer[:, self.position].copy_(value[0])

        def update_digest(self, layer, block):
            key_buffer, _ = self.cache[layer]
            source = compact_digest(key_buffer[:, block * 16:(block + 1) * 16])
            for name, tensor in source.items():
                axis = 0 if name == "counts" else 1
                self.digests[layer][name].narrow(axis, block, 1).copy_(tensor)

        def choose(self, layer, q, complete, scale):
            grouped = q.view(self.h, self.g, self.d).float() * scale
            data = self.digests[layer]
            if self.mode.startswith("coarse_"):
                def shortlist():
                    score = grouped.mean(1, keepdim=True) @ data["mean"][:, :complete].transpose(-1, -2)
                    return score.squeeze(1).topk(min(args.coarse_m, complete), sorted=False).indices
                ids = self.meter.call("coarse_score_topk", shortlist)
            else:
                ids = torch.arange(complete, device="cuda").expand(self.h, -1)
            count = ids.shape[1]
            def fine_score():
                packed = {name: data[name][self.heads[:, None], ids].flatten(0, 1)
                          for name in ("mean", "basis", "eigenvalues")}
                packed.update(
                    query=grouped,
                    counts=data["counts"][ids].reshape(-1),
                    block_ids=ids.flatten(),
                    scene_ids=self.heads.repeat_interleave(count),
                    token_position=torch.full((self.h,), self.position, device="cuda"),
                    layer_id=torch.full((self.h,), layer, device="cuda"),
                    kv_head_id=self.heads,
                    block_size=torch.full((self.h,), 16, device="cuda"),
                )
                return forward_packed(self.ranker, packed).view(self.h, count)
            scores = self.meter.call("ranker_score", fine_score)
            take = self.budget // 16
            if self.mode == "full_sort":
                order = self.meter.call("sort_or_topk", lambda: scores.sort(dim=-1, descending=True, stable=True).indices[:, :take])
            elif self.mode == "full_topk":
                order = self.meter.call("sort_or_topk", lambda: stable_topk_indices(scores, take, torch))
            else:
                order = self.meter.call("sort_or_topk", lambda: scores.topk(take, sorted=False).indices)
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
            def gather_and_attend():
                key_buffer, value_buffer = self.cache[layer]
                gather_index = indices[0, :, :, None].expand(self.h, indices.shape[-1], self.d)
                selected_key = key_buffer.gather(1, gather_index)[None]
                selected_value = value_buffer.gather(1, gather_index)[None]
                return F.scaled_dot_product_attention(
                    q.unsqueeze(2).contiguous(), selected_key.contiguous(), selected_value.contiguous(),
                    dropout_p=0.0, is_causal=False, scale=float(module.scaling), enable_gqa=True)
            output = self.meter.call("kv_gather_and_sdpa", gather_and_attend)
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
                "mean_recall_of_full_sort": intersection / selected}

    def evaluate_one(index, row, old, model, tokenizer, eos):
        clear()
        torch.cuda.reset_peak_memory_stats()
        prefix, first, prefill_seconds = prefill(model, row["input_ids"])
        if int(first.argmax()) != old["methods"]["full"]["ids"][0]:
            raise ValueError(f"Full first-token mismatch on case {index}")
        cpu_prefix = [(key.cpu(), value.cpu()) for key, value in prefix]
        del prefix
        clear()
        engine = Sparse(model, cpu_prefix, len(row["input_ids"]) + 128)
        del cpu_prefix
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.init_digest()
        torch.cuda.synchronize()
        digest_build = time.perf_counter() - start
        names = method_names(args)
        # Rotate execution order across cases to reduce thermal/order bias.
        rotation = index % len(names)
        order = names[rotation:] + names[:rotation]
        case = {"index": index, "task": row["task"], "input_sha256": row["input_sha256"],
                "prompt_tokens": len(row["input_ids"]), "prefill_seconds": prefill_seconds,
                "digest_build_seconds": digest_build, "budget": args.budget,
                "coarse_m": args.coarse_m, "methods": {}, "reference_full_score": old["methods"]["full"].get("ruler_score")}
        for name in order:
            result = decode(model, first, eos, engine, name, args.budget)
            prediction = official.normalize_prediction(
                tokenizer.decode(result["ids"], skip_special_tokens=True), row["task"])
            result["prediction"] = prediction
            result["ruler_score"] = official.metric_for(row["task"])([prediction], [row["answers"]])
            result["hits"] = sum(bool(re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)",
                                                prediction, re.I)) for answer in row["answers"])
            result["all_hit"] = result["hits"] == len(row["answers"])
            result["selected_blocks"] = None
            case["methods"][name] = result
        if index < args.audit_count and len(names) > 1:
            first_token = int(old["methods"]["full"]["ids"][0])
            selected = {name: decode(model, first, eos, engine, name, args.budget,
                                     forced=[first_token], capture=True)["selected_blocks"] for name in names}
            case["selection_audit_first_step"] = {
                name: compare_blocks(selected["full_sort"], selected[name]) for name in names[1:]
            }
        if index < args.profile_count:
            old_ids = old["methods"]["full"]["ids"]
            forced = (old_ids[:-1] + row["input_ids"][-16:])[:16]
            profile = {}
            for name in order:
                decode(model, first, eos, engine, name, args.budget, forced=forced[:3])
                samples = [1000 * decode(model, first, eos, engine, name, args.budget,
                                         forced=forced)["decode_seconds"] / 16 for _ in range(3)]
                events = decode(model, first, eos, engine, name, args.budget, forced=forced, profile=True)
                profile[name] = {"forced_ms_per_step": samples,
                                 "median_forced_ms_per_step": statistics.median(samples),
                                 "gpu_event_ms_per_step": events["stages_ms_per_step"]}
            case["profile"] = profile
        del engine
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
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        eos_value = model.generation_config.eos_token_id
        eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
        for index in range(args.shard_id, args.limit, args.num_shards):
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
                    "mean_recall_of_full_sort": statistics.mean(
                        item[name]["mean_recall_of_full_sort"] for item in audits),
                }
        pairs = {}
        comparisons = ((names[0], names[1]), (names[1], names[2]), (names[0], names[2])) if len(names) == 3 else ()
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
    parser.add_argument("--ruler-dir", type=Path, default=DEFAULT_RULER)
    parser.add_argument("--digest-dir", type=Path, default=DEFAULT_DIGEST)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO / "results/ranker_sort_ablation_20261115")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--coarse-m", type=int, default=256)
    parser.add_argument("--only-full-sort", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--audit-count", type=int, default=20)
    parser.add_argument("--profile-count", type=int, default=10)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
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
