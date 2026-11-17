"""Causal attention-mass oracle: block size 16/8/4, or token-level 512.

Uses the same 100 paired 32K RULER prompts as the earlier JSSA comparison.
For each generated token, each layer and KV head scores its complete historical
blocks with full-query attention probability, selects the highest-mass blocks,
and then performs sparse attention. There are no fixed sink/local blocks. The
currently incomplete block is retained and reported separately from the stated
historical budget. Oracle scoring scans full KV, so measured latency is not a
servable sparse-attention latency result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
import time
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01"
RULER = REPO / "experiments/runs/adablock_ruler_harder_20261022_01"
BUFFER = REPO / "experiments/server_snapshots/ruler_autonomous_20261022_01"
OFFICIAL = REPO / "experiments/server_snapshots/ruler_harder_20261022_01/official.py"
OUTPUT = REPO / "results/value_aware_selector_20261116/oracle_free"
SOURCE_SHA = "35eb364d354c7230c57282d3d5cbbe906941495c57946abd6597c9aac1c8f0ec"
BLOCKS = (16,)
BUDGETS = (512,)
METHODS = ("mass", "value")


def methods_for(args: argparse.Namespace) -> tuple[str, ...]:
    return METHODS


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def oracle_block_indices(query, keys, values, position, block_size, history_budget,
                         scale, method, max_swaps, candidate_pool, torch):
    """One KV head receives one shared block set; each GQA head retains its Q."""
    from experiments.one_off.value_aware_20261116 import contributions, mass_oracle, reconstructed, swap_search
    complete = position // block_size
    take = history_budget // block_size
    if block_size != 16 or history_budget != 512 or take > complete:
        raise ValueError("This run fixes 16-token blocks and 512 historical tokens")
    mass, weighted_value, full_output = contributions(
        query, keys[:, :position+1], values[:, :position+1], scale, block_size)
    initial = mass_oracle(mass, take)
    selected = []
    swap_counts = []
    errors = []
    for head in range(keys.shape[0]):
        ids = initial[head]
        records = []
        if method == "value":
            ids, records = swap_search(mass[head], weighted_value[head], ids,
                                       max_swaps=max_swaps, candidate_pool=candidate_pool)
        selected.append(ids)
        swap_counts.append(len(records))
        y_sparse = reconstructed(mass[head], weighted_value[head], ids)
        errors.append((y_sparse-full_output[head]).square().sum(-1).mean())
    selected = torch.stack(selected)
    offsets = torch.arange(block_size, device=keys.device)
    history = (selected[..., None]*block_size + offsets).flatten(1)
    tail = torch.arange(complete*block_size, position+1, device=keys.device)
    indices = torch.cat((history, tail[None].expand(keys.shape[0], -1)), -1).sort(-1).values
    coverage = mass[..., :complete].mean(1).gather(1, selected).sum(-1) + mass[..., -1].mean(1)
    return indices.long(), coverage, torch.stack(errors), sum(swap_counts)


def check(args: argparse.Namespace) -> list[dict]:
    if not 1 <= args.limit <= 100:
        raise ValueError("--limit must be in 1..100")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard ID")
    if args.model is None or not (args.model / "config.json").is_file():
        raise FileNotFoundError("Pass --model with local Llama-3.1-8B-Instruct weights")
    required = [args.source / "selected.json", args.ruler / "RULER/scripts/synthetic.yaml",
                args.buffer_dir / "cache_buffer.py", args.official]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    if file_sha(args.source / "selected.json") != SOURCE_SHA:
        raise ValueError("Selected input hash differs from the completed JSSA comparison")
    rows = json.loads((args.source / "selected.json").read_text())
    for index in range(args.limit):
        old_path = args.source / f"quality_{index:03d}.json"
        if not old_path.is_file():
            raise FileNotFoundError(old_path)
        old = json.loads(old_path.read_text())
        if rows[index]["input_sha256"] != old["input_sha256"]:
            raise ValueError(f"Input hash mismatch at case {index}")
    return rows


def protocol(args: argparse.Namespace) -> dict:
    return {
        "source": str(args.source.resolve()), "source_sha256": SOURCE_SHA,
        "model": str(args.model.resolve()), "methods": list(METHODS),
        "block_size": 16, "historical_blocks_per_kv_head": 32,
        "fixed": "current visible incomplete block, no sink/local",
        "attention": "post-RoPE Q/K, module.scale exactly once; per-query-head softmax; GQA mean selects shared blocks",
        "selection": "mass top-32 or improving one-block swaps from mass top-32",
        "candidate_range": f"top {args.candidate_pool} complete blocks by GQA mean attention mass; selected blocks can leave and re-enter",
        "stop": f"no improvement greater than 1e-9 or {args.max_swaps} accepted swaps",
        "search_cost": "O((accepted_swaps+1)*32*candidate_pool*G*D) per KV head and layer; candidate_chunk=128",
        "trajectory": "each method recomputes oracle on its own generated Q/K/V",
        "generation": "shared full prefill and first token, then greedy decode to EOS or 128 tokens",
        "timing_scope": "oracle search scans full KV; latency is diagnostic, not selector speed",
        "sparse_backend": "PyTorch SDPA with explicit GQA K/V expansion",
    }


def prepare_output(args: argparse.Namespace) -> None:
    target = args.output / "protocol.json"
    planned = protocol(args)
    if target.exists() and json.loads(target.read_text()) != planned:
        raise ValueError("Output protocol differs; select a new --output directory")
    if not target.exists():
        save_json(target, planned)


def run(args: argparse.Namespace) -> None:
    rows = check(args)
    prepare_output(args)
    selected_methods = methods_for(args)
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for generation")
    sys.path.insert(0, str(args.buffer_dir))
    from cache_buffer import BufferCache, reset_registry
    torch.set_num_threads(4)
    torch.manual_seed(20261015)

    scorer_spec = importlib.util.spec_from_file_location("ruler_granularity_official", args.official)
    official = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(official)
    official.PACKAGE = args.ruler

    @torch.inference_mode()
    def prefill(model, ids):
        reset_registry()
        cache = BufferCache(len(ids) + 256)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(ids), 1024):
            result = model.model(input_ids=torch.tensor([ids[offset:offset + 1024]], device="cuda"),
                                 past_key_values=cache, use_cache=True)
        logits = model.lm_head(result.last_hidden_state[:, -1:])[0, -1].float()
        torch.cuda.synchronize()
        return tuple(cache.to_legacy_cache()), logits, time.perf_counter() - start

    class OracleSparse:
        def __init__(self, model, cpu_pairs, capacity):
            self.model = model
            self.prefix = cpu_pairs[0][0].shape[2]
            self.h = model.config.num_key_value_heads
            self.g = model.config.num_attention_heads // self.h
            self.d = model.config.hidden_size // model.config.num_attention_heads
            self.pairs = []
            for key, value in cpu_pairs:
                kbuf = torch.empty((self.h, capacity, self.d), device="cuda", dtype=torch.bfloat16)
                vbuf = torch.empty_like(kbuf)
                kbuf[:, :self.prefix].copy_(key[0].to("cuda"))
                vbuf[:, :self.prefix].copy_(value[0].to("cuda"))
                self.pairs.append((kbuf, vbuf))
            self.original = []
            self.coverages = []
            self.errors = []
            self.swap_counts = []

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

        def forward(self, module, layer, hidden, embeddings):
            shape = (1, 1, -1, self.d)
            q = module.q_proj(hidden).view(shape).transpose(1, 2)
            k = module.k_proj(hidden).view(shape).transpose(1, 2)
            v = module.v_proj(hidden).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *embeddings)
            q = q[0, :, 0].unsqueeze(0).contiguous()
            key, value = self.pairs[layer]
            key[:, self.position].copy_(k[0, :, 0])
            value[:, self.position].copy_(v[0, :, 0])
            grouped = q.view(self.h, self.g, self.d)
            indices, coverage, error, swaps = oracle_block_indices(
                grouped, key, value, self.position, self.block_size, self.budget,
                float(module.scaling), self.method, self.max_swaps, self.candidate_pool, torch)
            self.coverages.append(coverage.mean())
            self.errors.append(error.mean())
            self.swap_counts.append(swaps)
            index = indices[..., None].expand(self.h, indices.shape[-1], self.d)
            selected_key = key.gather(1, index)[None]
            selected_value = value.gather(1, index)[None]
            output = F.scaled_dot_product_attention(
                q.unsqueeze(2).contiguous(),
                selected_key.repeat_interleave(self.g, dim=1).contiguous(),
                selected_value.repeat_interleave(self.g, dim=1).contiguous(),
                dropout_p=0.0, is_causal=False, scale=float(module.scaling))
            return module.o_proj(output.reshape(1, 1, -1)), None

    @torch.inference_mode()
    def decode(model, first, eos, engine, block_size, budget):
        engine.block_size, engine.budget = block_size, budget
        engine.coverages = []
        engine.errors = []
        engine.swap_counts = []
        ids = [int(first.argmax())]
        token = torch.empty((1, 1), dtype=torch.long, device="cuda")
        position = torch.empty((1,), dtype=torch.long, device="cuda")
        engine.install()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            for step in range(127):
                if ids[-1] in eos:
                    break
                current = engine.prefix + step
                engine.position = current
                token.fill_(ids[-1])
                position.fill_(current)
                result = model.model(input_ids=token, position_ids=position[None], cache_position=position,
                                     past_key_values=None, use_cache=False)
                logits = model.lm_head(result.last_hidden_state[:, -1:])[0, -1].float()
                ids.append(int(logits.argmax()))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        finally:
            engine.uninstall()
        steps = len(ids) - 1
        read = [budget + (engine.prefix + step) % block_size + 1 for step in range(steps)]
        return {
            "ids": ids,
            "decode_steps": steps,
            "decode_seconds_oracle_included": seconds,
            "mean_selected_tokens_per_kv_head": statistics.mean(read) if read else None,
            "mean_attention_mass_coverage": float(torch.stack(engine.coverages).mean()) if engine.coverages else None,
            "mean_output_squared_error": float(torch.stack(engine.errors).mean()) if engine.errors else None,
            "mean_accepted_swaps_per_layer_step": statistics.mean(engine.swap_counts) if engine.swap_counts else None,
            "eos": ids[-1] in eos,
            "cap_reached": len(ids) == 128 and ids[-1] not in eos,
        }

    def clear():
        reset_registry()
        gc.collect()
        torch.cuda.empty_cache()

    with torch.inference_mode():
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
            target = args.output / f"case_{index:03d}.json"
            old = json.loads((args.source / f"quality_{index:03d}.json").read_text())
            existing = json.loads(target.read_text()) if target.exists() else None
            if existing and all(name in existing["methods"] for name in selected_methods):
                print(f"skip complete {index}", flush=True)
                continue
            clear()
            prefix, first, prefill_seconds = prefill(model, rows[index]["input_ids"])
            if int(first.argmax()) != old["methods"]["full"]["ids"][0]:
                raise ValueError(f"Full first token mismatch at case {index}")
            cpu_pairs = [(key.cpu(), value.cpu()) for key, value in prefix]
            del prefix
            clear()
            engine = OracleSparse(model, cpu_pairs, len(rows[index]["input_ids"]) + 128)
            del cpu_pairs
            case = existing or {
                "index": index,
                "task": rows[index]["task"],
                "input_sha256": rows[index]["input_sha256"],
                "prompt_tokens": len(rows[index]["input_ids"]),
                "prefill_seconds": prefill_seconds,
                "full_reference": {
                    "ruler_score": old["methods"]["full"]["ruler_score"],
                    "ids": old["methods"]["full"]["ids"],
                    "cap_reached": old["methods"]["full"]["cap_reached"],
                },
                "methods": {},
            }
            if case["input_sha256"] != rows[index]["input_sha256"]:
                raise ValueError(f"Resume input hash mismatch at case {index}")
            order = list(selected_methods)
            offset = index % len(order)
            order = order[offset:] + order[:offset]
            for method in order:
                if method in case["methods"]:
                    continue
                engine.method = method
                engine.max_swaps = args.max_swaps
                engine.candidate_pool = args.candidate_pool
                result = decode(model, first, eos, engine, 16, 512)
                prediction = official.normalize_prediction(
                    tokenizer.decode(result["ids"], skip_special_tokens=True), case["task"])
                result["prediction"] = prediction
                result["ruler_score"] = official.metric_for(case["task"])([prediction], [rows[index]["answers"]])
                result["hits"] = sum(bool(re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)",
                                                   prediction, re.I)) for answer in rows[index]["answers"])
                result["all_hit"] = result["hits"] == len(rows[index]["answers"])
                case["methods"][method] = result
                save_json(target, case)
                print(f"case {index:03d} {method} score {result['ruler_score']}", flush=True)
            del engine
            clear()


def confidence_interval(values: list[float]) -> dict:
    generator = random.Random(20261115)
    boot = sorted(statistics.mean(values[generator.randrange(len(values))] for _ in values)
                  for _ in range(2000))
    return {"mean": statistics.mean(values), "bootstrap_95pct": [boot[49], boot[1949]]}


def summarize(args: argparse.Namespace) -> None:
    selected_methods = methods_for(args)
    recorded_protocol = json.loads((args.output / "protocol.json").read_text())
    if tuple(recorded_protocol["methods"]) != selected_methods:
        raise SystemExit("Output methods differ from the requested experiment mode")
    missing = [index for index in range(args.limit)
               if not (args.output / f"case_{index:03d}.json").is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} cases; first is {missing[0]:03d}")
    cases = [json.loads((args.output / f"case_{index:03d}.json").read_text())
             for index in range(args.limit)]
    if any(set(case["methods"]) != set(selected_methods) for case in cases):
        raise SystemExit("Some cases have incomplete methods")
    report = {"n": len(cases), "groups": {}}
    for group in ("all", "niah_multivalue", "vt"):
        rows = [case for case in cases if group == "all" or case["task"] == group]
        if not rows:
            continue
        methods = {}
        for name in selected_methods:
            values = [row["methods"][name] for row in rows]
            methods[name] = {
                "score": statistics.mean(value["ruler_score"] for value in values),
                "all_hit": sum(value["all_hit"] for value in values),
                "cap_reached": sum(value["cap_reached"] for value in values),
                "mean_attention_mass_coverage": statistics.mean(
                    value["mean_attention_mass_coverage"] for value in values),
                "mean_selected_tokens_per_kv_head": statistics.mean(
                    value["mean_selected_tokens_per_kv_head"] for value in values),
                "mean_decode_seconds_oracle_included": statistics.mean(
                    value["decode_seconds_oracle_included"] for value in values),
                "mean_output_squared_error": statistics.mean(
                    value["mean_output_squared_error"] for value in values),
                "mean_accepted_swaps_per_layer_step": statistics.mean(
                    value["mean_accepted_swaps_per_layer_step"] for value in values),
            }
        delta = [row["methods"]["value"]["ruler_score"] -
                 row["methods"]["mass"]["ruler_score"] for row in rows]
        contrasts = {"value_minus_mass": confidence_interval(delta)}
        report["groups"][group] = {
            "n": len(rows),
            "full_reference_score": statistics.mean(row["full_reference"]["ruler_score"] for row in rows),
            "methods": methods,
            "paired_score_differences": contrasts,
        }
    save_json(args.output / f"summary_{args.limit}.json", report)
    print(json.dumps(report["groups"]["all"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "summarize"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--ruler", type=Path, default=RULER)
    parser.add_argument("--buffer-dir", type=Path, default=BUFFER)
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--candidate-pool", type=int, default=256)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.max_swaps < 0 or args.candidate_pool < 32:
        parser.error("max-swaps must be nonnegative and candidate-pool >= 32")
    sys.path.insert(0, str(REPO))
    if args.command == "check":
        rows = check(args)
        print(json.dumps({"ready_cases": args.limit, "available_cases": len(rows),
                          "methods": methods_for(args), "source": str(args.source)}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
