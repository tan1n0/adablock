"""Combine the seven same-prompt RULER arms without conflating timing protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocols = {
        "full": read(args.matched / "full/protocol.json"),
        "jssa": read(args.matched / "jssa_512/protocol.json"),
        "rank4": read(args.matched / "cobs/protocol.json"),
        "oracle": read(args.oracle / "protocol.json"),
        "mlp": read(args.mlp / "protocol.json"),
        "fixed": read(args.fixed / "cobs/protocol.json"),
    }
    hashes = {
        protocols["full"]["selected_sha256"],
        protocols["jssa"]["selected_sha256"],
        protocols["rank4"]["selected_sha256"],
        protocols["oracle"]["source_sha256"],
        protocols["mlp"]["selected_sha256"],
        protocols["fixed"]["selected_sha256"],
    }
    if len(hashes) != 1:
        raise ValueError(f"The source prompt hashes differ: {hashes}")
    if protocols["rank4"]["selector_checkpoint_sha256"] == protocols["fixed"]["selector_checkpoint_sha256"]:
        raise ValueError("Fixed and trained COBS checkpoints must differ")
    if protocols["rank4"]["budget_history_tokens"] != 512 or protocols["fixed"]["budget_history_tokens"] != 512:
        raise ValueError("COBS history budget must be 512 tokens")
    if protocols["oracle"]["historical_blocks_per_kv_head"] != 32 or protocols["mlp"]["budget_history_tokens"] != 512:
        raise ValueError("Oracle/MLP history budget must be 512 tokens")
    if not protocols["rank4"]["graph_score"] or not protocols["fixed"]["graph_score"]:
        raise ValueError("Both rank-4 COBS arms must use the captured graph")

    matched = read(args.matched / "summary.json")["all"]["methods"]
    oracle = read(args.oracle / "summary_100.json")["groups"]["all"]["methods"]
    mlp = read(args.mlp / "summary_100.json")["groups"]["all"]["methods"]["mlp"]
    fixed = read(args.fixed / "paired_summary.json")["all"]["methods"]["fixed_cobs"]
    if any(read(path)["n"] != 100 for path in (
        args.oracle / "summary_100.json", args.mlp / "summary_100.json"
    )):
        raise ValueError("Oracle and MLP summaries must each have 100 cases")
    if read(args.matched / "summary.json")["all"]["n"] != 100 or read(args.fixed / "paired_summary.json")["all"]["n"] != 100:
        raise ValueError("Matched and fixed COBS summaries must each have 100 cases")

    oracle_steps = {name: 0 for name in ("mass", "value")}
    oracle_seconds = {name: 0.0 for name in ("mass", "value")}
    for i in range(100):
        cases = [read(path / f"case_{i:03d}.json") for path in (
            args.matched / "full", args.matched / "jssa_512", args.matched / "cobs",
            args.oracle, args.mlp, args.fixed / "cobs"
        )]
        case_hashes = {case["input_sha256"] for case in cases}
        if len(case_hashes) != 1 or any(case["index"] != i for case in cases):
            raise ValueError(f"Case {i} is not paired on the same prompt")
        for name in oracle_steps:
            row = cases[3]["methods"][name]
            oracle_steps[name] += row["decode_steps"]
            oracle_seconds[name] += row["decode_seconds_oracle_included"]

    methods = {
        "full": {"label": "Full", "score": matched["full"]["score"],
                 "all_hit": matched["full"]["all_hit"],
                 "decode_ms_per_token": matched["full"]["natural_decode_ms_per_step"],
                 "timing_scope": "matched natural generation"},
        "jssa_512": {"label": "JSSA-512", "score": matched["jssa_512"]["score"],
                     "all_hit": matched["jssa_512"]["all_hit"],
                     "decode_ms_per_token": matched["jssa_512"]["natural_decode_ms_per_step"],
                     "timing_scope": "matched natural generation"},
        "rank4_cobs": {"label": "rank-4 COBS, optimized", "score": matched["cobs"]["score"],
                       "all_hit": matched["cobs"]["all_hit"],
                       "decode_ms_per_token": matched["cobs"]["natural_decode_ms_per_step"],
                       "timing_scope": "matched natural generation; trained selector; CUDA Graph"},
        "mass_oracle": {"label": "Attention-mass Oracle", "score": oracle["mass"]["score"],
                        "all_hit": oracle["mass"]["all_hit"],
                        "decode_ms_per_token": 1000 * oracle_seconds["mass"] / oracle_steps["mass"],
                        "timing_scope": "diagnostic; full KV scan included"},
        "value_oracle": {"label": "Value-aware Oracle", "score": oracle["value"]["score"],
                         "all_hit": oracle["value"]["all_hit"],
                         "decode_ms_per_token": 1000 * oracle_seconds["value"] / oracle_steps["value"],
                         "timing_scope": "diagnostic; full KV scan and swap search included"},
        "old_mlp": {"label": "Legacy interaction MLP (S checkpoint)", "score": mlp["score"],
                    "all_hit": mlp["all_hit"],
                    "decode_ms_per_token": mlp["weighted_decode_ms_per_step"],
                    "timing_scope": "separate eager run; no CUDA Graph"},
        "fixed_cobs": {"label": "Fixed-formula COBS", "score": fixed["ruler_score"],
                       "all_hit": fixed["all_hit_count"],
                       "decode_ms_per_token": fixed["natural_decode_ms_per_step"],
                       "timing_scope": "separate same-prompt run; CUDA Graph"},
    }
    report = {
        "n": 100,
        "selected_sha256": hashes.pop(),
        "methods": methods,
        "protocol_note": "All arms share prompts and 512-token sparse history plus visible current block. Only Full/JSSA/rank-4 COBS were alternated by 10-case batch. Oracle timings include full-KV reference work. MLP is eager. Fixed COBS ran separately.",
        "full_historical_decode_ms_per_token": 38.99126201564129,
        "sources": {
            "matched": str(args.matched.resolve()),
            "oracle": str(args.oracle.resolve()),
            "mlp": str(args.mlp.resolve()),
            "fixed": str(args.fixed.resolve()),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "summary.json", report)
    lines = ["| Method | RULER score | All-hit / 100 | Natural decode ms/token | Timing scope |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in methods.values():
        lines.append(f"| {row['label']} | {row['score']:.2f} | {row['all_hit']} | "
                     f"{row['decode_ms_per_token']:.2f} | {row['timing_scope']} |")
    lines += ["", "Matched Full is 39.03 ms/token. The older Full run reported 38.99 ms/token.", ""]
    table = "\n".join(lines)
    (args.output / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
