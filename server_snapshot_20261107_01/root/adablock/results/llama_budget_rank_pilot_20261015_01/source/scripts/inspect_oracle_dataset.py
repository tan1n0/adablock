from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect AdaBlock oracle JSONL quality.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, int) or value is None or isinstance(value, str):
        return False
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    return False


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    total = 0
    bad_rows = 0
    budget_counts: Counter[int] = Counter()
    required_k_counts: Counter[int] = Counter()
    allocated_k_counts: Counter[int] = Counter()
    reuse_counts: Counter[int] = Counter()
    source_docs: Counter[str] = Counter()
    attention_modes: Counter[str] = Counter()
    attention_backends: Counter[str] = Counter()
    oracle_layer_configs: Counter[str] = Counter()
    transition_versions: Counter[str] = Counter()
    transition_gaps: Counter[int] = Counter()
    reuse_previous_selectors: Counter[str] = Counter()
    anchor_offset_modes: Counter[str] = Counter()
    anchor_offsets: Counter[int] = Counter()
    controller_gaps: Counter[int] = Counter()
    policy_input_modes: Counter[str] = Counter()
    causal_feature_rows = 0
    causal_index_violations = 0
    history_boundary_rows = 0
    self_attention_mass_sum = 0.0
    past_attention_mass_sum = 0.0
    attention_mass_rows = 0
    sink_blocks_sum = 0
    local_blocks_sum = 0
    middle_blocks_sum = 0
    required_k_sum = 0
    allocated_k_sum = 0
    allocation_rows = 0
    saturated_rows = 0
    hidden_size: int | None = None
    first_bad: tuple[int, dict[str, Any]] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.max_rows is not None and total >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if has_nonfinite(row):
                bad_rows += 1
                if first_bad is None:
                    first_bad = (line_no, row)
                continue

            budget_counts[int(row["budget_label"])] += 1
            required_k = int(row.get("required_k", -1))
            required_k_counts[required_k] += 1
            if required_k >= 0:
                required_k_sum += required_k
            if "allocated_k" in row:
                allocated_k = int(row["allocated_k"])
                allocated_k_counts[allocated_k] += 1
                allocated_k_sum += allocated_k
                allocation_rows += 1
                saturated_rows += int(bool(row.get("budget_saturated", False)))
            reuse_counts[int(row["reuse_label"])] += 1
            source_docs[str(row.get("source_doc_id", row.get("doc_id", -1)))] += 1
            attention_modes[str(row.get("oracle_attention_mode", "legacy"))] += 1
            attention_backends[str(row.get("oracle_attn_implementation", "unknown"))] += 1
            layer_config = row.get("oracle_layer_indices")
            oracle_layer_configs[
                ",".join(map(str, layer_config)) if layer_config else "all_or_unknown"
            ] += 1
            transition_versions[
                str(row.get("oracle_transition_version", "legacy_or_unknown"))
            ] += 1
            if "oracle_transition_gap" in row:
                transition_gaps[int(row["oracle_transition_gap"])] += 1
            reuse_previous_selectors[
                str(row.get("reuse_previous_selector", "legacy_or_unknown"))
            ] += 1
            if "oracle_anchor_offset" in row:
                anchor_offsets[int(row["oracle_anchor_offset"])] += 1
            anchor_offset_modes[
                str(row.get("oracle_anchor_offset_mode", "legacy_or_unknown"))
            ] += 1
            if bool(row.get("oracle_causal_features", False)):
                causal_feature_rows += 1
            policy_input_modes[
                str(row.get("oracle_policy_input_mode", "legacy_or_unknown"))
            ] += 1
            if (
                "attention_target_token_index" in row
                and "controller_token_index" in row
            ):
                target_t = int(row["attention_target_token_index"])
                controller_t = int(row["controller_token_index"])
                controller_gaps[target_t - controller_t] += 1
                causal_index_violations += int(
                    controller_t != target_t - 1
                    or row.get("oracle_policy_input_mode")
                    != "previous_hidden_only"
                    or int(row.get("score_query_token_index", -1))
                    != controller_t
                    or int(row.get("policy_feature_max_token_index", -1))
                    != controller_t
                )
            history_boundary_rows += int(bool(row.get("history_block_boundary", False)))
            if (
                "current_token_self_attention_mass" in row
                and "past_attention_mass" in row
            ):
                self_attention_mass_sum += float(
                    row["current_token_self_attention_mass"]
                )
                past_attention_mass_sum += float(row["past_attention_mass"])
                attention_mass_rows += 1
            hidden_size = hidden_size or len(row["hidden_state"])
            sink_blocks_sum += int(row.get("fixed_sink_blocks", 0))
            local_blocks_sum += int(row.get("fixed_local_blocks", 0))
            middle_blocks_sum += int(row.get("eligible_middle_blocks", 0))

    valid_rows = total - bad_rows

    summary = {
        "path": str(path),
        "total_rows": total,
        "valid_rows": valid_rows,
        "bad_rows": bad_rows,
        "hidden_size": hidden_size,
        "num_docs": len(source_docs),
        "oracle_attention_modes": dict(attention_modes),
        "oracle_attention_backends": dict(attention_backends),
        "oracle_layer_configs": dict(oracle_layer_configs),
        "oracle_transition_versions": dict(transition_versions),
        "oracle_transition_gaps": dict(sorted(transition_gaps.items())),
        "reuse_previous_selectors": dict(reuse_previous_selectors),
        "oracle_anchor_offset_modes": dict(anchor_offset_modes),
        "oracle_anchor_offset_top20": anchor_offsets.most_common(20),
        "causal_feature_rows": causal_feature_rows,
        "oracle_policy_input_modes": dict(policy_input_modes),
        "controller_gap_counts": dict(sorted(controller_gaps.items())),
        "causal_index_violations": causal_index_violations,
        "history_block_boundary_rate": history_boundary_rows / max(valid_rows, 1),
        "budget_label_counts": dict(sorted(budget_counts.items())),
        "required_k_top20": required_k_counts.most_common(20),
        "reuse_label_counts": dict(sorted(reuse_counts.items())),
        "avg_fixed_sink_blocks": sink_blocks_sum / max(valid_rows, 1),
        "avg_fixed_local_blocks": local_blocks_sum / max(valid_rows, 1),
        "avg_eligible_middle_blocks": middle_blocks_sum / max(valid_rows, 1),
        "first_bad_line": first_bad[0] if first_bad else None,
    }
    if attention_mass_rows:
        summary.update(
            {
                "avg_current_self_attention_mass": (
                    self_attention_mass_sum / attention_mass_rows
                ),
                "avg_past_attention_mass": (
                    past_attention_mass_sum / attention_mass_rows
                ),
                "avg_total_attention_mass": (
                    (self_attention_mass_sum + past_attention_mass_sum)
                    / attention_mass_rows
                ),
            }
        )
    if allocation_rows:
        summary.update(
            {
                "allocated_k_counts": dict(sorted(allocated_k_counts.items())),
                "avg_required_k": required_k_sum / allocation_rows,
                "avg_allocated_k": allocated_k_sum / allocation_rows,
                "budget_saturated_rows": saturated_rows,
                "budget_saturated_rate": saturated_rows / allocation_rows,
            }
        )
    print(summary)


if __name__ == "__main__":
    main()
