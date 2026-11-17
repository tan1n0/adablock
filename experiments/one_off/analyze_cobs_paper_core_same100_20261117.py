"""Pair fixed-coefficient COBS with the completed same-prompt Full/JSSA/learned COBS run."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def interval(values: list[float], seed: int = 20261117) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return [draws[49], draws[1949]]


def summarize(cases: list[dict]) -> dict:
    names = ("full", "jssa_512", "learned_cobs", "fixed_cobs")
    methods = {}
    for name in names:
        rows = [case["methods"][name] for case in cases]
        steps = sum(row["decode_steps"] for row in rows)
        methods[name] = {
            "ruler_score": statistics.mean(row["score"] for row in rows),
            "all_hit_count": sum(row["all_hit"] for row in rows),
            "cap_reached_count": sum(row["cap_reached"] for row in rows),
            "mean_generated_tokens": statistics.mean(row["decode_steps"] for row in rows),
            "natural_decode_ms_per_step": 1000 * sum(row["decode_seconds"] for row in rows) / steps,
            "forced_16_ms_per_step_median": statistics.median(row["forced_ms_per_step"] for row in rows),
            "postprefill_forced_16_seconds_mean": statistics.mean(row["postprefill16_seconds"] for row in rows),
            "index_build_seconds_mean": statistics.mean(row["index_seconds"] for row in rows),
            "index_cache_mib_mean": statistics.mean(row["index_cache_mib"] for row in rows),
            "peak_allocated_mib_median": statistics.median(row["peak_allocated_mib"] for row in rows),
            "peak_reserved_mib_median": statistics.median(row["peak_reserved_mib"] for row in rows),
        }
    comparisons = {}
    for name in names[:-1]:
        score = [case["methods"]["fixed_cobs"]["score"] - case["methods"][name]["score"] for case in cases]
        time = [case["methods"]["fixed_cobs"]["forced_ms_per_step"] - case["methods"][name]["forced_ms_per_step"] for case in cases]
        post = [case["methods"]["fixed_cobs"]["postprefill16_seconds"] - case["methods"][name]["postprefill16_seconds"] for case in cases]
        hit = [int(case["methods"]["fixed_cobs"]["all_hit"]) - int(case["methods"][name]["all_hit"]) for case in cases]
        comparisons[f"fixed_cobs_minus_{name}"] = {
            "ruler_score_mean": statistics.mean(score), "ruler_score_bootstrap95": interval(score),
            "score_wins": sum(x > 0 for x in score), "score_losses": sum(x < 0 for x in score),
            "all_hit_net_count": sum(hit), "all_hit_wins": sum(x > 0 for x in hit),
            "all_hit_losses": sum(x < 0 for x in hit),
            "forced_ms_per_step_mean": statistics.mean(time), "forced_ms_bootstrap95": interval(time),
            "postprefill16_seconds_mean": statistics.mean(post), "postprefill16_bootstrap95": interval(post),
        }
    return {"n": len(cases), "methods": methods, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed_protocol = json.loads((args.fixed / "protocol.json").read_text())
    learned_protocol = json.loads((args.matched / "cobs" / "protocol.json").read_text())
    if fixed_protocol["selected_sha256"] != learned_protocol["selected_sha256"]:
        raise ValueError("The selected prompts differ")
    for key in ("budget_history_tokens", "block_size", "fixed", "attention", "gqa_backend", "graph_score", "digest_chunk"):
        if fixed_protocol[key] != learned_protocol[key]:
            raise ValueError(f"COBS protocol mismatch: {key}")
    if fixed_protocol["selector_checkpoint_sha256"] == learned_protocol["selector_checkpoint_sha256"]:
        raise ValueError("Fixed and learned COBS checkpoints must differ")
    original = json.loads((args.matched / "paired_cases.json").read_text())
    if len(original) != 100:
        raise ValueError("Expected all 100 previously paired cases")
    cases = []
    for i, old in enumerate(original):
        case = json.loads((args.fixed / f"case_{i:03d}.json").read_text())
        if case["index"] != i or case["input_sha256"] != old["input_sha256"] or case["task"] != old["task"]:
            raise ValueError(f"Prompt or task mismatch for case {i}")
        if case["budget"] != 512 or case["profile"]["forced_input_ids"] != old["forced_input_ids"]:
            raise ValueError(f"Forced input or KV budget mismatch for case {i}")
        row = case["methods"]["cobs"]
        p = case["profile"]["cobs"]
        fixed = {
            "score": row["ruler_score"], "all_hit": row["all_hit"],
            "cap_reached": row["cap_reached"], "decode_steps": row["decode_steps"],
            "decode_seconds": row["decode_seconds"],
            "forced_ms_per_step": p["median_forced_ms_per_step"],
            "index_seconds": row["index_build_seconds"],
            "postprefill16_seconds": row["index_build_seconds"] + 16 * p["median_forced_ms_per_step"] / 1000,
            "index_cache_mib": row["index_cache_mib"],
            "peak_allocated_mib": row["peak_allocated_mib"],
            "peak_reserved_mib": row["peak_reserved_mib"],
        }
        cases.append({"index": i, "task": old["task"], "input_sha256": old["input_sha256"],
                      "methods": {"full": old["methods"]["full"],
                                  "jssa_512": old["methods"]["jssa_512"],
                                  "learned_cobs": old["methods"]["cobs"],
                                  "fixed_cobs": fixed}})
    report = {
        "scope": "Fixed paper-core second-order score on frozen Llama, not official full NSA/NoPE/subspace/FP4 COBS",
        "timing_note": "Fixed COBS measured in a later run; timing is paired by prompt and forced tokens, not contemporaneously interleaved with historical Full/JSSA/learned COBS",
        "selected_sha256": fixed_protocol["selected_sha256"],
        "fixed_selector_sha256": fixed_protocol["selector_checkpoint_sha256"]["cobs"],
        "learned_selector_sha256": learned_protocol["selector_checkpoint_sha256"]["cobs"],
        "all": summarize(cases),
        "niah_multivalue": summarize([case for case in cases if case["task"] == "niah_multivalue"]),
        "vt": summarize([case for case in cases if case["task"] == "vt"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output.parent / "paired_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    (args.output.parent / "COMPLETE.json").write_text(json.dumps({"status": "complete", "cases": 100,
                                                                    "summary": args.output.name}, indent=2))
    print(json.dumps(report["all"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
