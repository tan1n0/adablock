from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import AdaBlockPolicy, AdaBlockPolicyConfig
from models.block_ranker import load_ranker
from models.layer_head_sparse_qwen import LayerHeadQKSparseController
from utils.qk_block_oracle import QK_SCORE_SELECTORS
from utils.block_oracle import (
    aggregate_block_mass,
    cosine_block_scores,
    fill_selection_to_total_budget,
    fixed_and_middle_blocks,
    make_block_ranges,
    quest_block_scores,
    arkvale_block_scores,
    score_summary_features,
    select_middle_blocks,
    top_blocks,
)
from utils.longbench_eval import (
    LONG_INPUT_SHORT_OUTPUT_TASKS,
    TASK_MAX_NEW_TOKENS,
    build_model_input_text,
    extract_answers,
    format_longbench_prompt,
    load_dtype,
    load_local_task,
    load_longbench_task,
    mean,
    normalize_device_map,
    score_prediction,
    truncate_middle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdaBlock-driven sparse decoding on LongBench.")
    parser.add_argument(
        "--policy-checkpoint",
        default=None,
        help=(
            "Required when --selection-mode is policy or "
            "policy_layer_head_qk."
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model only from the local Hugging Face cache; never query the Hub.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/longbench_adablock_sparse")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--longbench-cache-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume each task from an existing output JSONL. Existing rows must "
            "have contiguous idx values starting at zero."
        ),
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--local-window-blocks", type=int, default=32)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--ranker-checkpoint", default=None)
    parser.add_argument("--fixed-middle-blocks", type=int, default=None,
                        help="For learned_qk/fixed_qk: exact requested middle K; total cap is fixed blocks + K.")
    parser.add_argument("--ranker-untrained-layers", choices=["error", "cobs"], default="error",
                        help="Explicit fallback when a ranker covers only selected layers.")
    parser.add_argument("--ranker-profile", action="store_true",
                        help="Synchronize stage timing in the layer/head prototype (adds timing overhead).")
    parser.add_argument("--reuse-threshold", type=float, default=0.8)
    parser.add_argument(
        "--qk-score-selector",
        choices=list(QK_SCORE_SELECTORS),
        default="quest_aabb",
        help="Post-RoPE Q/K selector used by policy_qk mode.",
    )
    parser.add_argument(
        "--qk-cobs-rank",
        type=int,
        default=4,
        help="Truncated covariance rank used by the training-aligned COBS selector.",
    )
    parser.add_argument(
        "--qk-layer-indices",
        nargs="+",
        type=int,
        default=[12, 20, 27],
        help="Layers using training-aligned Q/K digests in policy_qk mode.",
    )
    parser.add_argument("--fill-budget-with-score", action="store_true")
    parser.add_argument(
        "--budget-calibration",
        choices=["none", "tail_ema", "three_signal_proxy", "tail_margin"],
        default="none",
        help=(
            "Online, gradient-free calibration of budget logits. "
            "tail_ema uses the normalized retrieval-score mass in the bottom "
            "25%% of blocks as a diffuse-score proxy. three_signal_proxy combines "
            "score-utilization, score-tail pressure, and score-boundary margin. "
            "tail_margin uses tail pressure for direction and boundary margin as a gate."
        ),
    )
    parser.add_argument(
        "--budget-calibration-target-tail",
        type=float,
        default=0.215,
        help="Target bottom-tail score mass for tail_ema calibration.",
    )
    parser.add_argument(
        "--budget-calibration-eta",
        type=float,
        default=0.25,
        help="Step size for the scalar budget logit bias.",
    )
    parser.add_argument(
        "--budget-calibration-beta",
        type=float,
        default=0.9,
        help="EMA coefficient for the tail-mass proxy.",
    )
    parser.add_argument(
        "--budget-calibration-max-shift",
        type=float,
        default=1.5,
        help="Absolute clamp for the scalar budget logit bias.",
    )
    parser.add_argument(
        "--feedback-target-utilization",
        type=float,
        default=0.5,
        help="Target selected-block utilization for three_signal_proxy.",
    )
    parser.add_argument(
        "--feedback-target-margin",
        type=float,
        default=0.05,
        help="Reference score boundary margin for three_signal_proxy.",
    )
    parser.add_argument(
        "--feedback-score-epsilon",
        type=float,
        default=0.01,
        help="Normalized score threshold used by three_signal_proxy utilization.",
    )
    parser.add_argument(
        "--feedback-tail-low",
        type=float,
        default=0.08,
        help="Low tail-pressure threshold for decreasing the next budget.",
    )
    parser.add_argument(
        "--feedback-tail-high",
        type=float,
        default=0.20,
        help="High tail-pressure threshold for increasing the next budget.",
    )
    parser.add_argument(
        "--feedback-margin-low",
        type=float,
        default=0.01,
        help="Small boundary margin below which an increase is permitted.",
    )
    parser.add_argument(
        "--feedback-margin-high",
        type=float,
        default=0.05,
        help="Large boundary margin above which a decrease is permitted.",
    )
    parser.add_argument(
        "--selection-mode",
        default="policy",
        choices=[
            "policy",
            "policy_layer_head_qk",
            "policy_qk",
            "layer_head_qk_topk",
            "learned_qk",
            "fixed_qk",
            "oracle_topk",
            "oracle_coverage",
            "score_topk",
            "quest_topk",
            "arkvale_topk",
        ],
        help="Block selection strategy for sparse decoding.",
    )
    parser.add_argument(
        "--attention-coverage-threshold",
        type=float,
        default=0.95,
        help=(
            "For oracle_coverage, select the minimum number of middle blocks "
            "whose dense-probe attention mass plus fixed self/sink/local mass "
            "reaches this fraction."
        ),
    )
    parser.add_argument(
        "--oracle-max-middle-blocks",
        type=int,
        default=None,
        help=(
            "Optional safety cap for oracle_coverage dynamic middle blocks. "
            "Unset means no cap; capped steps are reported as saturated."
        ),
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", None],
        help="Optional attention backend override. Leave unset to use the model default.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens-override", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--debug-compare-full",
        action="store_true",
        help="Collect token-by-token full-vs-sparse decoding traces for debugging.",
    )
    parser.add_argument(
        "--debug-max-steps",
        type=int,
        default=20,
        help="Maximum generated steps to retain in the debug token trace.",
    )
    parser.add_argument(
        "--debug-logits-topk",
        type=int,
        default=10,
        help="How many top logits to retain per debug step.",
    )
    parser.add_argument(
        "--debug-teacher-force-full-prefix",
        action="store_true",
        help="Run an additional sparse trace that always feeds the full-decoding prefix tokens.",
    )
    parser.add_argument(
        "--allow-legacy-policy-checkpoint",
        action="store_true",
        help=(
            "Allow a checkpoint without policy_input_mode=previous_hidden_only. "
            "Use only for an intentional legacy ablation."
        ),
    )
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        path = Path(args.local_data_dir) / f"{task}.jsonl"
        return load_local_task(path)
    return load_longbench_task(task, split=args.split, cache_dir=args.longbench_cache_dir)


def load_resume_records(path: Path, total_rows: int) -> list[dict[str, object]]:
    """Load and validate a contiguous prefix written by an interrupted run."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        raise ValueError(
            f"Cannot resume {path}: the final JSONL row is incomplete (missing newline)."
        )
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Cannot resume {path}: blank row at line {line_number}.")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Cannot resume {path}: invalid JSON at line {line_number}."
            ) from exc
        expected_idx = line_number - 1
        if record.get("idx") != expected_idx:
            raise ValueError(
                f"Cannot resume {path}: line {line_number} has idx={record.get('idx')!r}, "
                f"expected {expected_idx}."
            )
        records.append(record)
    if len(records) > total_rows:
        raise ValueError(
            f"Cannot resume {path}: it contains {len(records)} rows but this run has "
            f"only {total_rows} input samples."
        )
    return records


def load_policy(
    path: str | Path,
    device: torch.device,
    allow_legacy: bool = False,
) -> AdaBlockPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    config_dict = dict(checkpoint["config"])
    policy_input_mode = config_dict.get("policy_input_mode", "legacy_unspecified")
    if policy_input_mode != "previous_hidden_only" and not allow_legacy:
        raise ValueError(
            f"{path} is not marked as a previous_hidden_only causal checkpoint "
            f"(policy_input_mode={policy_input_mode!r}). Retrain on "
            "causal_next_token_v1 Oracle data, or pass "
            "--allow-legacy-policy-checkpoint for an intentional old-model ablation."
        )
    if "budget_buckets" in config_dict:
        config_dict["budget_buckets"] = tuple(config_dict["budget_buckets"])
    policy = AdaBlockPolicy(AdaBlockPolicyConfig(**config_dict))
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    policy.eval()
    return policy


def fill_selection_with_scores(
    selected: set[int],
    scores: torch.Tensor,
    budget: int,
    max_blocks: int,
    required_blocks: set[int] | None = None,
) -> set[int]:
    if scores.numel() == 0 or budget <= 0:
        return set()
    target = min(budget, max_blocks, scores.numel())
    required = {idx for idx in (required_blocks or set()) if 0 <= idx < scores.numel()}
    valid_selected = {idx for idx in selected if 0 <= idx < scores.numel()}
    ranked_selected = sorted(valid_selected, key=lambda idx: float(scores[idx]), reverse=True)
    ranked_global = torch.argsort(scores, descending=True).tolist()

    ordered: list[int] = []
    seen: set[int] = set()

    for idx in sorted(required):
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    for idx in ranked_selected:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    for idx in ranked_global:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
        if len(ordered) >= target:
            break
    return set(ordered[:target])


def enforce_local_window(selected_blocks: set[int], current_block: int, local_window_blocks: int) -> set[int]:
    start = max(0, current_block - local_window_blocks + 1)
    local_blocks = set(range(start, current_block + 1))
    return set(selected_blocks) | local_blocks


def blocks_to_token_indices(block_ids: set[int], block_ranges: list[tuple[int, int]]) -> list[int]:
    token_indices: list[int] = []
    for block_id in sorted(block_ids):
        if not (0 <= block_id < len(block_ranges)):
            continue
        start, end = block_ranges[block_id]
        token_indices.extend(range(start, end))
    return token_indices


def select_middle_for_attention_coverage(
    block_mass: torch.Tensor,
    fixed_blocks: set[int],
    middle_blocks: set[int],
    threshold: float,
    always_retained_mass: float | torch.Tensor = 0.0,
    max_middle_blocks: int | None = None,
) -> tuple[set[int], float, bool]:
    """Select the fewest highest-mass middle blocks reaching total coverage.

    ``block_mass`` contains dense attention probability aggregated over history
    blocks. ``always_retained_mass`` is normally the current token's self mass.
    Scores are used only to order blocks; achieved coverage is always measured
    against the original dense attention probability mass.
    """

    if block_mass.ndim != 1:
        raise ValueError("block_mass must be one-dimensional.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("attention coverage threshold must be in (0, 1].")
    if max_middle_blocks is not None and max_middle_blocks < 0:
        raise ValueError("max_middle_blocks must be non-negative when set.")

    always_mass = torch.as_tensor(
        always_retained_mass,
        dtype=block_mass.dtype,
        device=block_mass.device,
    )
    total_mass = (block_mass.sum() + always_mass).clamp_min(1e-8)
    valid_fixed = sorted(
        idx for idx in fixed_blocks if 0 <= idx < block_mass.numel()
    )
    retained_mass = always_mass + (
        block_mass[valid_fixed].sum()
        if valid_fixed
        else block_mass.new_tensor(0.0)
    )
    coverage_floor = threshold - 1e-7
    if float(retained_mass / total_mass) >= coverage_floor:
        return set(), float(retained_mass / total_mass), False

    valid_middle = sorted(
        idx for idx in middle_blocks if 0 <= idx < block_mass.numel()
    )
    if not valid_middle:
        achieved = float(retained_mass / total_mass)
        return set(), achieved, achieved < coverage_floor

    middle_index = torch.tensor(
        valid_middle, dtype=torch.long, device=block_mass.device
    )
    order = torch.argsort(
        block_mass.index_select(0, middle_index), descending=True
    )
    ordered_blocks = middle_index.index_select(0, order)
    ordered_mass = block_mass.index_select(0, ordered_blocks)
    cumulative_coverage = (retained_mass + ordered_mass.cumsum(dim=0)) / total_mass
    reached = torch.nonzero(
        cumulative_coverage >= coverage_floor, as_tuple=False
    )
    required_k = (
        int(reached[0, 0].item()) + 1
        if reached.numel()
        else int(ordered_blocks.numel())
    )
    selected_k = required_k
    if max_middle_blocks is not None:
        selected_k = min(selected_k, max_middle_blocks)
    selected = set(ordered_blocks[:selected_k].tolist())
    achieved = (
        float(retained_mass / total_mass)
        if selected_k == 0
        else float(cumulative_coverage[selected_k - 1])
    )
    return selected, achieved, achieved < coverage_floor


def normalize_past_key_values(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def to_model_cache(past_key_values):
    if past_key_values is None or DynamicCache is None:
        return past_key_values
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def get_layer_devices(model) -> list[torch.device]:
    transformer = getattr(model, "model", model)
    devices: list[torch.device] = []
    for layer in transformer.layers:
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            device = torch.device(getattr(model, "device", "cpu"))
        devices.append(device)
    return devices


def slice_past_key_values(
    past_key_values,
    token_indices: list[int],
    layer_devices: list[torch.device],
):
    index = torch.tensor(token_indices, dtype=torch.long)
    sliced = []
    for layer_idx, (key, value) in enumerate(past_key_values):
        layer_index = index.to(key.device)
        target_device = layer_devices[layer_idx] if layer_idx < len(layer_devices) else key.device
        sliced.append(
            (
                key.index_select(2, layer_index).to(target_device),
                value.index_select(2, layer_index).to(target_device),
            )
        )
    return tuple(sliced)


def append_last_kv(full_past_key_values, step_past_key_values):
    updated = []
    for (full_key, full_value), (step_key, step_value) in zip(full_past_key_values, step_past_key_values):
        new_key = torch.cat([full_key, step_key[:, :, -1:, :].detach().cpu()], dim=2)
        new_value = torch.cat([full_value, step_value[:, :, -1:, :].detach().cpu()], dim=2)
        updated.append((new_key, new_value))
    return tuple(updated)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices.gather(-1, next_idx).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_trace_entry(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def topk_logits_entries(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, object]]:
    topk = min(k, int(logits.shape[-1]))
    values, indices = torch.topk(logits.float(), k=topk)
    return [
        {
            "token_id": int(token_id),
            "token_text": decode_trace_entry(tokenizer, int(token_id)),
            "logit": float(logit),
        }
        for token_id, logit in zip(indices.tolist(), values.tolist())
    ]


def run_full_decode_trace(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, object]], list[int]]:
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    max_new_tokens = args.max_new_tokens_override
    if max_new_tokens is None:
        raise ValueError("run_full_decode_trace requires caller to pass task max_new_tokens via args.max_new_tokens_override")

    transformer = getattr(model, "model", model)
    with torch.no_grad():
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    generated_ids: list[int] = []
    trace: list[dict[str, object]] = []
    first_logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, -1]
    next_token_id = sample_next_token(first_logits, args.temperature, args.top_p)
    generated_ids.append(next_token_id)
    if len(trace) < args.debug_max_steps:
        trace.append(
            {
                "step": 0,
                "token_id": next_token_id,
                "token_text": decode_trace_entry(tokenizer, next_token_id),
                "topk_logits": topk_logits_entries(tokenizer, first_logits, args.debug_logits_topk),
            }
        )
    past_key_values = normalize_past_key_values(outputs.past_key_values)
    history_len = int(input_ids.shape[-1])

    for step in range(1, max_new_tokens):
        step_input_ids = torch.tensor([[generated_ids[-1]]], device=model.device)
        step_attention_mask = torch.ones((1, history_len + 1), dtype=torch.long, device=model.device)
        position_ids = torch.tensor([[history_len]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            step_outputs = transformer(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                position_ids=position_ids,
                past_key_values=to_model_cache(past_key_values),
                use_cache=True,
                output_hidden_states=False,
            )
        step_logits = model.lm_head(step_outputs.last_hidden_state[:, -1:, :])[0, -1]
        next_token_id = sample_next_token(step_logits, args.temperature, args.top_p)
        generated_ids.append(next_token_id)
        if len(trace) < args.debug_max_steps:
            trace.append(
                {
                    "step": step,
                    "token_id": next_token_id,
                    "token_text": decode_trace_entry(tokenizer, next_token_id),
                    "topk_logits": topk_logits_entries(tokenizer, step_logits, args.debug_logits_topk),
                }
            )
        if next_token_id == tokenizer.eos_token_id:
            break
        past_key_values = normalize_past_key_values(step_outputs.past_key_values)
        history_len += 1

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction, trace, generated_ids


def run_sparse_decode(
    model,
    tokenizer,
    policy: AdaBlockPolicy | None,
    policy_device: torch.device,
    prompt_text: str,
    args: argparse.Namespace,
    teacher_forced_ids: list[int] | None = None,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    max_new_tokens = args.max_new_tokens_override
    if max_new_tokens is None:
        raise ValueError("run_sparse_decode requires caller to pass task max_new_tokens via args.max_new_tokens_override")

    with torch.no_grad():
        transformer = getattr(model, "model", model)
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    full_past_key_values = tuple(
        (key.detach().cpu(), value.detach().cpu()) for key, value in normalize_past_key_values(outputs.past_key_values)
    )
    hidden_history = outputs.last_hidden_state[0].detach().cpu().float()
    last_logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, -1]
    next_token_id = sample_next_token(last_logits, args.temperature, args.top_p)
    generated_ids = [next_token_id]
    current_input_token = teacher_forced_ids[0] if teacher_forced_ids else generated_ids[-1]

    sink_blocks = (args.sink_tokens + args.block_size - 1) // args.block_size
    max_blocks = args.budget_tokens // args.block_size
    layer_devices = get_layer_devices(model)
    previous_hidden = (
        hidden_history[-2].clone() if hidden_history.shape[0] >= 2 else None
    )
    previous_middle_selection: set[int] = set()
    selected_blocks_sum = 0.0
    selected_middle_blocks_sum = 0.0
    selected_tokens_sum = 0.0
    selected_steps = 0
    reuse_steps = 0
    achieved_coverage_sum = 0.0
    coverage_steps = 0
    coverage_saturated_steps = 0
    trace: list[dict[str, object]] = []
    if args.debug_max_steps > 0:
        trace.append(
            {
                "step": 0,
                "token_id": next_token_id,
                "token_text": decode_trace_entry(tokenizer, next_token_id),
                "selected_blocks": [],
                "selected_block_count": 0,
                "selected_token_count": 0,
                "topk_logits": topk_logits_entries(tokenizer, last_logits, args.debug_logits_topk),
            }
        )

    for _ in range(max_new_tokens - 1):
        history_len = hidden_history.shape[0]
        current_token_index = history_len - 1
        current_block = current_token_index // args.block_size
        block_ranges = make_block_ranges(history_len, args.block_size)
        candidate_ranges = block_ranges[: current_block + 1]
        achieved_attention_coverage: float | None = None
        coverage_saturated = False
        selected_middle_count = 0
        if len(candidate_ranges) <= 1:
            token_indices = list(range(history_len))
            selected_blocks = set(range(len(candidate_ranges)))
            if args.selection_mode == "oracle_coverage":
                achieved_attention_coverage = 1.0
        else:
            local_blocks = set(range(max(0, current_block - args.local_window_blocks + 1), current_block + 1))
            if args.selection_mode in {"quest_topk", "arkvale_topk"}:
                score_fn = (
                    arkvale_block_scores
                    if args.selection_mode == "arkvale_topk"
                    else quest_block_scores
                )
                scores = score_fn(hidden_history, candidate_ranges, current_token_index)
            else:
                scores = cosine_block_scores(hidden_history, candidate_ranges, current_token_index)
            sink, local, middle = fixed_and_middle_blocks(
                num_blocks=len(candidate_ranges),
                sink_blocks=sink_blocks,
                local_window_blocks=args.local_window_blocks,
                current_block=current_block,
            )
            fixed_blocks = sink | local
            dynamic_capacity = max(0, max_blocks - len(fixed_blocks))
            if args.selection_mode in {"oracle_topk", "oracle_coverage"}:
                dense_token_indices = list(range(history_len))
                dense_past = slice_past_key_values(full_past_key_values, dense_token_indices, layer_devices)
                dense_step_input_ids = torch.tensor([[current_input_token]], device=model.device)
                dense_attention_mask = torch.ones((1, history_len + 1), dtype=torch.long, device=model.device)
                dense_position_ids = torch.tensor([[history_len]], dtype=torch.long, device=model.device)
                # Keep the long prefill and normal sparse decode on the requested
                # efficient backend.  Only this q_len=1 Oracle probe needs eager
                # attention so that Transformers returns the exact attention row.
                probe_config = transformer.config
                previous_attn_implementation = probe_config._attn_implementation
                probe_config._attn_implementation = "eager"
                try:
                    with torch.no_grad():
                        dense_outputs = transformer(
                            input_ids=dense_step_input_ids,
                            attention_mask=dense_attention_mask,
                            position_ids=dense_position_ids,
                            past_key_values=to_model_cache(dense_past),
                            use_cache=True,
                            output_attentions=True,
                            output_hidden_states=False,
                        )
                finally:
                    probe_config._attn_implementation = previous_attn_implementation
                if dense_outputs.attentions is None:
                    raise RuntimeError(
                        f"{args.selection_mode} selection requires attention weights, but "
                        "the model backend returned None. Use eager attention for Oracle evaluation."
                    )
                dense_attentions = torch.stack([attn[0].detach().cpu().float() for attn in dense_outputs.attentions])
                dense_mean_attention = dense_attentions.mean(dim=(0, 1))
                token_attention = dense_mean_attention[0, :history_len]
                block_mass = aggregate_block_mass(token_attention, candidate_ranges)
                self_attention_mass = dense_mean_attention[0, history_len]
                if args.selection_mode == "oracle_coverage":
                    dynamic_blocks, achieved_attention_coverage, coverage_saturated = (
                        select_middle_for_attention_coverage(
                            block_mass=block_mass,
                            fixed_blocks=fixed_blocks,
                            middle_blocks=middle,
                            threshold=args.attention_coverage_threshold,
                            always_retained_mass=self_attention_mass,
                            max_middle_blocks=args.oracle_max_middle_blocks,
                        )
                    )
                else:
                    dynamic_blocks = set(
                        top_blocks(
                            block_mass,
                            dynamic_capacity,
                            candidates=middle,
                        )
                    )
                selected_blocks = fixed_blocks | dynamic_blocks
                token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
                # The dense pass is an Oracle probe only.  It must not update
                # hidden state or the full KV trajectory; those come from the
                # sparse forward below.
                del (
                    dense_outputs,
                    dense_attentions,
                    dense_mean_attention,
                    dense_past,
                    dense_token_indices,
                    token_attention,
                    block_mass,
                    self_attention_mass,
                )
            elif args.selection_mode in {"score_topk", "quest_topk", "arkvale_topk"}:
                dynamic_blocks = select_middle_blocks(scores, middle, dynamic_capacity)
                selected_blocks = fixed_blocks | dynamic_blocks
                if args.fill_budget_with_score:
                    selected_blocks = fill_selection_to_total_budget(
                        selected=selected_blocks,
                        scores=scores,
                        total_budget_blocks=max_blocks,
                        required_blocks=fixed_blocks,
                    )
                token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
            else:
                if policy is None:
                    raise ValueError("--selection-mode=policy requires --policy-checkpoint")
                bucket_values = list(policy.config.budget_buckets)
                score_features = score_summary_features(scores).unsqueeze(0)
                if previous_hidden is None:
                    query_drift = torch.tensor([1.0])
                else:
                    drift_value = float(
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            hidden_history[current_token_index].unsqueeze(0),
                            previous_hidden.unsqueeze(0),
                            dim=-1,
                        ).item()
                    )
                    query_drift = torch.tensor(
                        [min(2.0, max(0.0, drift_value))]
                    )
                with torch.no_grad():
                    policy_outputs = policy(
                        hidden_state=hidden_history[current_token_index].unsqueeze(0).to(policy_device),
                        query_drift=query_drift.to(policy_device),
                        score_features=score_features.to(policy_device),
                    )
                bucket_idx = int(policy_outputs["budget_prob"].argmax(dim=-1).item())
                predicted_middle_k = min(bucket_values[bucket_idx], dynamic_capacity, len(middle))
                current_middle = select_middle_blocks(scores, middle, predicted_middle_k)
                reuse = float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold
                reusable_middle = previous_middle_selection & middle
                if reuse and reusable_middle:
                    dynamic_blocks = reusable_middle
                    reuse_steps += 1
                else:
                    dynamic_blocks = current_middle
                selected_blocks = fixed_blocks | dynamic_blocks
                if args.fill_budget_with_score:
                    selected_blocks = fill_selection_to_total_budget(
                        selected=selected_blocks,
                        scores=scores,
                        total_budget_blocks=max_blocks,
                        required_blocks=fixed_blocks,
                    )
                token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
                previous_middle_selection = selected_blocks - fixed_blocks
            selected_middle_count = len(selected_blocks - fixed_blocks)
        selected_blocks_sum += len(selected_blocks)
        selected_middle_blocks_sum += selected_middle_count
        selected_tokens_sum += len(token_indices)
        selected_steps += 1
        if achieved_attention_coverage is not None:
            achieved_coverage_sum += achieved_attention_coverage
            coverage_steps += 1
            coverage_saturated_steps += int(coverage_saturated)
        sparse_past = slice_past_key_values(full_past_key_values, token_indices, layer_devices)
        step_input_ids = torch.tensor([[current_input_token]], device=model.device)
        step_attention_mask = torch.ones((1, len(token_indices) + 1), dtype=torch.long, device=model.device)
        position_ids = torch.tensor([[history_len]], dtype=torch.long, device=model.device)

        with torch.no_grad():
            step_outputs = transformer(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                position_ids=position_ids,
                past_key_values=to_model_cache(sparse_past),
                use_cache=True,
                output_hidden_states=False,
            )

        step_past = normalize_past_key_values(step_outputs.past_key_values)
        full_past_key_values = append_last_kv(full_past_key_values, step_past)
        new_hidden = step_outputs.last_hidden_state[0, -1].detach().cpu().float().unsqueeze(0)
        hidden_history = torch.cat([hidden_history, new_hidden], dim=0)
        previous_hidden = hidden_history[-2]
        step_logits = model.lm_head(step_outputs.last_hidden_state[:, -1:, :])[0, -1]
        next_token_id = sample_next_token(step_logits, args.temperature, args.top_p)
        generated_ids.append(next_token_id)
        if len(trace) < args.debug_max_steps:
            trace.append(
                {
                    "step": len(generated_ids) - 1,
                    "token_id": next_token_id,
                    "token_text": decode_trace_entry(tokenizer, next_token_id),
                    "selected_blocks": sorted(selected_blocks),
                    "selected_block_count": len(selected_blocks),
                    "selected_token_count": len(token_indices),
                    "selected_middle_block_count": selected_middle_count,
                    **(
                        {"achieved_attention_coverage": achieved_attention_coverage}
                        if achieved_attention_coverage is not None
                        else {}
                    ),
                    **(
                        {"coverage_saturated": coverage_saturated}
                        if achieved_attention_coverage is not None
                        else {}
                    ),
                    "topk_logits": topk_logits_entries(tokenizer, step_logits, args.debug_logits_topk),
                }
            )
        next_teacher_index = len(generated_ids) - 1
        if teacher_forced_ids and next_teacher_index < len(teacher_forced_ids):
            current_input_token = teacher_forced_ids[next_teacher_index]
        else:
            current_input_token = next_token_id
        if next_token_id == tokenizer.eos_token_id and not teacher_forced_ids:
            break
        if teacher_forced_ids and next_teacher_index >= len(teacher_forced_ids):
            break
        del sparse_past, step_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stats = {
        "avg_selected_blocks": selected_blocks_sum / max(selected_steps, 1),
        "avg_selected_middle_blocks": selected_middle_blocks_sum / max(selected_steps, 1),
        "avg_selected_tokens": selected_tokens_sum / max(selected_steps, 1),
        "reuse_rate": reuse_steps / max(selected_steps, 1),
        "generated_tokens": len(generated_ids),
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "used_prompt_tokens": int(input_ids.shape[-1]),
    }
    if args.selection_mode == "oracle_coverage":
        stats.update(
            {
                "attention_coverage_threshold": args.attention_coverage_threshold,
                "avg_achieved_attention_coverage": (
                    achieved_coverage_sum / max(coverage_steps, 1)
                ),
                "coverage_saturation_rate": (
                    coverage_saturated_steps / max(coverage_steps, 1)
                ),
                "oracle_max_middle_blocks": args.oracle_max_middle_blocks,
                "oracle_attention_aggregation": "mean_all_layers_query_heads",
                "oracle_probe_attention_implementation": "eager",
                "generation_attention_implementation": args.attn_implementation,
                "oracle_dense_probe_state_used_for_generation": False,
            }
        )
    return prediction, stats, trace


def run_layer_head_qk_decode(
    model,
    tokenizer,
    policy: AdaBlockPolicy | None,
    policy_device: torch.device,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, float], list[dict[str, object]]]:
    """Decode with current-query, per-layer/per-KV-head Q/K selection.

    The optional policy remains causal and checkpoint-compatible: it consumes
    the previous final hidden state and the legacy hidden-cosine score features
    to decide only how many middle blocks may be retrieved.  The location of
    those blocks is chosen inside each attention layer from the current token's
    post-RoPE query and that layer/KV head's post-RoPE key centroids.
    """

    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    max_new_tokens = args.max_new_tokens_override
    if max_new_tokens is None:
        raise ValueError(
            "run_layer_head_qk_decode requires caller to pass task max_new_tokens "
            "via args.max_new_tokens_override"
        )

    transformer = getattr(model, "model", model)
    with torch.no_grad():
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    full_past_key_values = normalize_past_key_values(outputs.past_key_values)
    hidden_history = outputs.last_hidden_state[0].detach().cpu().float()
    last_logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, -1]
    next_token_id = sample_next_token(last_logits, args.temperature, args.top_p)
    generated_ids = [next_token_id]
    current_input_token = next_token_id

    sink_blocks = (args.sink_tokens + args.block_size - 1) // args.block_size
    max_blocks = args.budget_tokens // args.block_size
    if args.selection_mode in {"learned_qk", "fixed_qk"}:
        max_blocks = sink_blocks + args.local_window_blocks + args.fixed_middle_blocks
    previous_hidden = hidden_history[-2].clone() if hidden_history.shape[0] >= 2 else None
    selected_blocks_sum = 0.0
    selected_tokens_sum = 0.0
    union_blocks_sum = 0.0
    union_tokens_sum = 0.0
    selected_steps = 0
    budget_calibration_shift = 0.0
    budget_tail_ema: float | None = None
    budget_calibration_shift_sum = 0.0
    budget_tail_sum = 0.0
    budget_calibration_steps = 0
    requested_middle_k_sum = 0.0
    reused_heads_sum = 0.0
    total_heads_sum = 0.0
    trace: list[dict[str, object]] = []
    if args.debug_max_steps > 0:
        trace.append(
            {
                "step": 0,
                "token_id": next_token_id,
                "token_text": decode_trace_entry(tokenizer, next_token_id),
                "selected_blocks": [],
                "selected_block_count": 0,
                "selected_token_count": 0,
                "topk_logits": topk_logits_entries(tokenizer, last_logits, args.debug_logits_topk),
            }
        )

    ranker, ranker_payload = (None, None)
    if args.selection_mode == "learned_qk":
        ranker, ranker_payload = args._ranker_bundle
    controller = LayerHeadQKSparseController(
        model=model,
        full_past_key_values=full_past_key_values,
        block_size=args.block_size,
        sink_blocks=sink_blocks,
        local_window_blocks=args.local_window_blocks,
        qk_score_selector=(
            args.qk_score_selector
            if args.selection_mode in {"policy_qk", "fixed_qk"}
            else None
        ),
        cobs_rank=args.qk_cobs_rank,
        qk_layer_indices=None if args.selection_mode in {"learned_qk", "fixed_qk"} else args.qk_layer_indices,
        ranker=ranker,
        ranker_metadata=ranker_payload["data_metadata"] if ranker_payload else None,
        ranker_untrained_layers=args.ranker_untrained_layers,
        profile=args.ranker_profile,
    )
    # The controller owns a CPU copy of the complete KV history.  Release the
    # dense prefill cache on GPU before starting sparse decode.
    del outputs, full_past_key_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    controller.install()
    try:
        for _ in range(max_new_tokens - 1):
            history_len = int(hidden_history.shape[0])
            block_ranges = make_block_ranges(history_len, args.block_size)
            current_block = len(block_ranges) - 1
            sink, local, middle = fixed_and_middle_blocks(
                num_blocks=len(block_ranges),
                sink_blocks=sink_blocks,
                local_window_blocks=args.local_window_blocks,
                current_block=current_block,
            )
            fixed_blocks = sink | local
            dynamic_capacity = max(0, min(max_blocks, len(block_ranges)) - len(fixed_blocks))
            requested_middle_k: int | None = None

            reuse_previous = False
            if args.selection_mode in {"policy_layer_head_qk", "policy_qk"}:
                if policy is None:
                    raise ValueError(
                        "--selection-mode=policy_layer_head_qk/policy_qk requires "
                        "--policy-checkpoint"
                    )
                legacy_scores = cosine_block_scores(
                    hidden_history,
                    block_ranges,
                    history_len - 1,
                )
                score_features = score_summary_features(legacy_scores).unsqueeze(0)
                tail_mass = float(score_features[0, -1].item())
                if args.budget_calibration == "tail_ema":
                    if not 0.0 < args.budget_calibration_beta < 1.0:
                        raise ValueError(
                            "--budget-calibration-beta must be between 0 and 1."
                        )
                    budget_tail_ema = (
                        tail_mass
                        if budget_tail_ema is None
                        else args.budget_calibration_beta * budget_tail_ema
                        + (1.0 - args.budget_calibration_beta) * tail_mass
                    )
                    budget_calibration_shift = min(
                        args.budget_calibration_max_shift,
                        max(
                            -args.budget_calibration_max_shift,
                            budget_calibration_shift
                            + args.budget_calibration_eta
                            * (budget_tail_ema - args.budget_calibration_target_tail),
                        ),
                    )
                    budget_calibration_shift_sum += budget_calibration_shift
                    budget_tail_sum += tail_mass
                    budget_calibration_steps += 1
                if previous_hidden is None:
                    query_drift = torch.tensor([1.0])
                else:
                    drift_value = float(
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            hidden_history[-1].unsqueeze(0),
                            previous_hidden.unsqueeze(0),
                            dim=-1,
                        ).item()
                    )
                    query_drift = torch.tensor([min(2.0, max(0.0, drift_value))])
                with torch.no_grad():
                    policy_outputs = policy(
                        hidden_state=hidden_history[-1].unsqueeze(0).to(policy_device),
                        query_drift=query_drift.to(policy_device),
                        score_features=score_features.to(policy_device),
                    )
                reuse_previous = (
                    float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold
                )
                bucket_values = list(policy.config.budget_buckets)
                budget_logits = policy_outputs["budget_logits"]
                # Preliminary three-feedback validation.  The current sparse
                # controller does not expose per-block attention weights, so
                # these are explicitly score-based proxies rather than true
                # retrieved-attention utilization measurements.
                if args.budget_calibration == "three_signal_proxy":
                    middle_ids = sorted(middle)
                    if middle_ids:
                        middle_scores = legacy_scores[middle_ids].float()
                        middle_probs = torch.softmax(middle_scores, dim=0)
                        raw_idx = int(policy_outputs["budget_logits"].argmax(dim=-1).item())
                        raw_k = min(
                            int(bucket_values[raw_idx]),
                            dynamic_capacity,
                            len(middle_ids),
                        )
                        take_k = max(1, raw_k)
                        order = torch.argsort(middle_scores, descending=True)
                        selected_probs = middle_probs[order[:take_k]]
                        utilization = float(
                            (selected_probs > args.feedback_score_epsilon).float().mean().item()
                        )
                        tail_count = max(1, int(0.25 * take_k))
                        tail_pressure = float(selected_probs[-tail_count:].sum().item())
                        boundary_margin = float("inf")
                        if take_k < len(middle_ids):
                            boundary_margin = float(
                                middle_scores[order[take_k - 1]].item()
                                - middle_scores[order[take_k]].item()
                            )
                        margin_risk = 0.0 if boundary_margin == float("inf") else max(
                            0.0,
                            min(
                                1.0,
                                (args.feedback_target_margin - boundary_margin)
                                / max(args.feedback_target_margin, 1e-6),
                            ),
                        )
                        util_signal = max(
                            -1.0,
                            min(1.0, utilization - args.feedback_target_utilization),
                        )
                        tail_signal = max(-1.0, min(1.0, tail_pressure - args.budget_calibration_target_tail))
                        feedback_signal = 0.25 * util_signal + 0.35 * tail_signal + 0.40 * margin_risk
                        budget_calibration_shift = min(
                            args.budget_calibration_max_shift,
                            max(
                                -args.budget_calibration_max_shift,
                                budget_calibration_shift
                                + args.budget_calibration_eta * feedback_signal,
                            ),
                        )
                        budget_calibration_shift_sum += budget_calibration_shift
                        budget_tail_sum += tail_pressure
                        budget_calibration_steps += 1
                if args.budget_calibration in {"tail_ema", "three_signal_proxy"}:
                    bucket_values_tensor = budget_logits.new_tensor(bucket_values)
                    budget_rank = bucket_values_tensor / max(
                        float(max(bucket_values)), 1.0
                    )
                    budget_logits = budget_logits + budget_calibration_shift * budget_rank
                bucket_idx = int(budget_logits.argmax(dim=-1).item())
                requested_middle_k = min(
                    int(policy.config.budget_buckets[bucket_idx]),
                    dynamic_capacity,
                    len(middle),
                )
                requested_middle_k_sum += requested_middle_k

            if args.selection_mode in {"learned_qk", "fixed_qk"}:
                requested_middle_k = min(args.fixed_middle_blocks, len(middle), dynamic_capacity)
                requested_middle_k_sum += requested_middle_k

            controller.begin_step(
                history_len=history_len,
                max_blocks=max_blocks,
                requested_middle_k=requested_middle_k,
                fill_to_budget=args.fill_budget_with_score,
                reuse_previous=reuse_previous,
            )
            step_input_ids = torch.tensor([[current_input_token]], device=model.device)
            position_ids = torch.tensor([[history_len]], dtype=torch.long, device=model.device)
            cache_position = position_ids[0]
            step_attention_mask = torch.ones(
                (1, history_len + 1), dtype=torch.long, device=model.device
            )
            with torch.no_grad():
                step_outputs = transformer(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    output_hidden_states=False,
                )
            step_stats = controller.finish_step()

            new_hidden = step_outputs.last_hidden_state[0, -1].detach().cpu().float().unsqueeze(0)
            hidden_history = torch.cat([hidden_history, new_hidden], dim=0)
            previous_hidden = hidden_history[-2]
            step_logits = model.lm_head(step_outputs.last_hidden_state[:, -1:, :])[0, -1]
            next_token_id = sample_next_token(step_logits, args.temperature, args.top_p)
            generated_ids.append(next_token_id)
            current_input_token = next_token_id

            selected_blocks_sum += step_stats.avg_selected_blocks
            selected_tokens_sum += step_stats.avg_selected_tokens
            union_blocks_sum += step_stats.union_blocks
            union_tokens_sum += step_stats.union_tokens
            selected_steps += 1
            reused_heads_sum += step_stats.reused_heads
            total_heads_sum += step_stats.total_heads
            if len(trace) < args.debug_max_steps:
                trace.append(
                    {
                        "step": len(generated_ids) - 1,
                        "token_id": next_token_id,
                        "token_text": decode_trace_entry(tokenizer, next_token_id),
                        "selected_blocks": [],
                        "selected_block_count": step_stats.avg_selected_blocks,
                        "selected_token_count": step_stats.avg_selected_tokens,
                        "union_block_count": step_stats.union_blocks,
                        "union_token_count": step_stats.union_tokens,
                        "topk_logits": topk_logits_entries(
                            tokenizer, step_logits, args.debug_logits_topk
                        ),
                    }
                )
            if next_token_id == tokenizer.eos_token_id:
                break
            del step_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        controller.uninstall()

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stats = {
        "avg_selected_blocks": selected_blocks_sum / max(selected_steps, 1),
        "avg_selected_tokens": selected_tokens_sum / max(selected_steps, 1),
        "avg_union_blocks": union_blocks_sum / max(selected_steps, 1),
        "avg_union_tokens": union_tokens_sum / max(selected_steps, 1),
        "ranker_checkpoint": args.ranker_checkpoint if args.selection_mode == "learned_qk" else None,
        "ranker_trained_layers": sorted(controller.ranker_layers),
        "ranker_untrained_layers": args.ranker_untrained_layers if ranker else None,
        "profile_seconds": controller.timings,
        "effective_total_budget_tokens": max_blocks * args.block_size,
        "reuse_rate": reused_heads_sum / max(total_heads_sum, 1.0),
        "qk_score_selector": (
            args.qk_score_selector if args.selection_mode in {"policy_qk", "fixed_qk"} else None
        ),
        "avg_reused_heads": reused_heads_sum / max(selected_steps, 1),
        "avg_total_heads": total_heads_sum / max(selected_steps, 1),
        "budget_calibration_mode": args.budget_calibration,
        "budget_calibration_final_shift": budget_calibration_shift,
        "budget_calibration_mean_shift": (
            budget_calibration_shift_sum / max(budget_calibration_steps, 1)
        ),
        "budget_calibration_mean_tail_mass": (
            budget_tail_sum / max(budget_calibration_steps, 1)
        ),
        "budget_calibration_steps": budget_calibration_steps,
        "avg_requested_middle_blocks": requested_middle_k_sum / max(selected_steps, 1),
        "generated_tokens": len(generated_ids),
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "used_prompt_tokens": int(input_ids.shape[-1]),
    }
    return prediction, stats, trace


def main() -> None:
    args = parse_args()
    if args.selection_mode in {"learned_qk", "fixed_qk"}:
        if args.fixed_middle_blocks is None or args.fixed_middle_blocks < 0:
            raise SystemExit("--fixed-middle-blocks >= 0 is required for learned_qk/fixed_qk.")
        if args.fill_budget_with_score or args.resume:
            raise SystemExit("v1 fixed-middle evaluation requires a fresh run without fill-budget-with-score.")
        if args.local_window_blocks < 1:
            raise SystemExit("v1 requires local-window-blocks >= 1 to keep partial blocks fixed.")
        if args.selection_mode == "learned_qk":
            if not args.ranker_checkpoint:
                raise SystemExit("--ranker-checkpoint is required for learned_qk.")
            args._ranker_bundle = load_ranker(args.ranker_checkpoint, "cpu")
            if args._ranker_bundle[1]["data_metadata"]["model_name"] != args.model_name:
                raise SystemExit("Ranker and base model names must match the extraction metadata.")
    if not 0.0 < args.attention_coverage_threshold <= 1.0:
        raise SystemExit("--attention-coverage-threshold must be in (0, 1].")
    if args.oracle_max_middle_blocks is not None and args.oracle_max_middle_blocks < 0:
        raise SystemExit("--oracle-max-middle-blocks must be non-negative.")
    if args.selection_mode == "oracle_coverage" and args.fill_budget_with_score:
        raise SystemExit(
            "--fill-budget-with-score is incompatible with oracle_coverage; "
            "coverage must determine the selected block count."
        )
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.selection_mode in {"policy", "policy_layer_head_qk", "policy_qk"}:
        if not args.policy_checkpoint:
            raise SystemExit(
                "--policy-checkpoint is required when --selection-mode is policy "
                "or policy_layer_head_qk"
            )
        policy = load_policy(
            args.policy_checkpoint,
            policy_device,
            allow_legacy=args.allow_legacy_policy_checkpoint,
        )
    else:
        policy = None
    load_model_path = args.model_name
    if args.local_files_only:
        # Resolve to a local directory as well as setting local_files_only.
        # Transformers 4.55's custom-generate probe otherwise consults the Hub
        # for a repo ID even though this evaluator never calls generate().
        load_model_path = str(Path(cached_file(args.model_name, "config.json", local_files_only=True)).parent)
    tokenizer = AutoTokenizer.from_pretrained(
        load_model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_model_path,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {}

    for task in args.tasks:
        rows = list(load_rows(args, task))
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        task_scores: list[float] = []
        task_blocks: list[float] = []
        task_middle_blocks: list[float] = []
        task_tokens: list[float] = []
        task_union_blocks: list[float] = []
        task_union_tokens: list[float] = []
        task_reuse: list[float] = []
        task_achieved_coverage: list[float] = []
        task_coverage_saturation: list[float] = []
        pred_path = output_dir / f"{task}.jsonl"
        if args.selection_mode in {"learned_qk", "fixed_qk"} and pred_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {pred_path}")
        resume_records = (
            load_resume_records(pred_path, len(rows)) if args.resume else []
        )
        if args.selection_mode == "oracle_coverage":
            for record in resume_records:
                observed_threshold = record.get("attention_coverage_threshold")
                if (
                    observed_threshold is None
                    or abs(
                        float(observed_threshold)
                        - args.attention_coverage_threshold
                    )
                    > 1e-9
                ):
                    raise ValueError(
                        f"Cannot resume {pred_path}: stored coverage threshold "
                        f"{observed_threshold!r} does not match requested "
                        f"{args.attention_coverage_threshold}."
                    )
        for record in resume_records:
            task_scores.append(float(record["score"]))
            task_blocks.append(float(record["avg_selected_blocks"]))
            if "avg_selected_middle_blocks" in record:
                task_middle_blocks.append(
                    float(record["avg_selected_middle_blocks"])
                )
            task_tokens.append(float(record["avg_selected_tokens"]))
            if "avg_union_blocks" in record:
                task_union_blocks.append(float(record["avg_union_blocks"]))
            if "avg_union_tokens" in record:
                task_union_tokens.append(float(record["avg_union_tokens"]))
            task_reuse.append(float(record["reuse_rate"]))
            if "avg_achieved_attention_coverage" in record:
                task_achieved_coverage.append(
                    float(record["avg_achieved_attention_coverage"])
                )
            if "coverage_saturation_rate" in record:
                task_coverage_saturation.append(
                    float(record["coverage_saturation_rate"])
                )
        start_idx = len(resume_records)
        if args.resume:
            print(
                {
                    "event": "resume_task",
                    "task": task,
                    "completed_samples": start_idx,
                    "total_samples": len(rows),
                    "remaining_samples": len(rows) - start_idx,
                }
            )
        task_args = argparse.Namespace(**vars(args))
        task_args.max_new_tokens_override = args.max_new_tokens_override or TASK_MAX_NEW_TOKENS[task]
        output_mode = "a" if args.resume else "w"
        progress = tqdm(
            rows[start_idx:],
            desc=f"sparse:{task}",
            unit="sample",
            total=len(rows),
            initial=start_idx,
        )
        with pred_path.open(output_mode, encoding="utf-8") as out:
            for idx, row in enumerate(progress, start=start_idx):
                prompt = format_longbench_prompt(task, row)
                model_input_text = build_model_input_text(
                    tokenizer,
                    prompt,
                    use_chat_template=not args.no_chat_template,
                )
                if args.selection_mode in {
                    "layer_head_qk_topk",
                    "policy_layer_head_qk",
                    "policy_qk",
                    "learned_qk",
                    "fixed_qk",
                }:
                    prediction, stats, sparse_trace = run_layer_head_qk_decode(
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                        policy_device=policy_device,
                        prompt_text=model_input_text,
                        args=task_args,
                    )
                else:
                    prediction, stats, sparse_trace = run_sparse_decode(
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                        policy_device=policy_device,
                        prompt_text=model_input_text,
                        args=task_args,
                    )
                debug_payload = None
                if args.debug_compare_full:
                    full_prediction, full_trace, full_generated_ids = run_full_decode_trace(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=model_input_text,
                        args=task_args,
                    )
                    max_steps = min(len(sparse_trace), len(full_trace), args.debug_max_steps)
                    comparison = []
                    divergence_step = None
                    for step_idx in range(max_steps):
                        sparse_step = sparse_trace[step_idx]
                        full_step = full_trace[step_idx]
                        match = sparse_step["token_id"] == full_step["token_id"]
                        if divergence_step is None and not match:
                            divergence_step = step_idx
                        comparison.append(
                            {
                                "step": step_idx,
                                "match": match,
                                "sparse_token_id": sparse_step["token_id"],
                                "sparse_token_text": sparse_step["token_text"],
                                "full_token_id": full_step["token_id"],
                                "full_token_text": full_step["token_text"],
                                "selected_blocks": sparse_step["selected_blocks"],
                                "selected_block_count": sparse_step["selected_block_count"],
                                "selected_token_count": sparse_step["selected_token_count"],
                            }
                        )
                    debug_payload = {
                        "full_prediction": full_prediction,
                        "sparse_prediction": prediction,
                        "divergence_step": divergence_step,
                        "compared_steps": comparison,
                    }
                    if args.debug_teacher_force_full_prefix and args.selection_mode not in {
                        "layer_head_qk_topk",
                        "policy_layer_head_qk",
                        "policy_qk",
                        "learned_qk",
                        "fixed_qk",
                    }:
                        teacher_prediction, _teacher_stats, teacher_sparse_trace = run_sparse_decode(
                            model=model,
                            tokenizer=tokenizer,
                            policy=policy,
                            policy_device=policy_device,
                            prompt_text=model_input_text,
                            args=task_args,
                            teacher_forced_ids=full_generated_ids,
                        )
                        teacher_max_steps = min(len(teacher_sparse_trace), len(full_trace), args.debug_max_steps)
                        teacher_comparison = []
                        teacher_divergence_step = None
                        for step_idx in range(teacher_max_steps):
                            sparse_step = teacher_sparse_trace[step_idx]
                            full_step = full_trace[step_idx]
                            match = sparse_step["token_id"] == full_step["token_id"]
                            if teacher_divergence_step is None and not match:
                                teacher_divergence_step = step_idx
                            teacher_comparison.append(
                                {
                                    "step": step_idx,
                                    "match": match,
                                    "sparse_token_id": sparse_step["token_id"],
                                    "sparse_token_text": sparse_step["token_text"],
                                    "full_token_id": full_step["token_id"],
                                    "full_token_text": full_step["token_text"],
                                    "selected_blocks": sparse_step["selected_blocks"],
                                    "selected_block_count": sparse_step["selected_block_count"],
                                    "selected_token_count": sparse_step["selected_token_count"],
                                    "sparse_topk_logits": sparse_step["topk_logits"],
                                    "full_topk_logits": full_step["topk_logits"],
                                }
                            )
                        debug_payload["teacher_forced_sparse_prediction"] = teacher_prediction
                        debug_payload["teacher_forced_divergence_step"] = teacher_divergence_step
                        debug_payload["teacher_forced_compared_steps"] = teacher_comparison
                answers = extract_answers(row)
                score = score_prediction(task, prediction, answers)
                task_scores.append(score)
                task_blocks.append(stats["avg_selected_blocks"])
                if "avg_selected_middle_blocks" in stats:
                    task_middle_blocks.append(stats["avg_selected_middle_blocks"])
                task_tokens.append(stats["avg_selected_tokens"])
                if "avg_union_blocks" in stats:
                    task_union_blocks.append(stats["avg_union_blocks"])
                if "avg_union_tokens" in stats:
                    task_union_tokens.append(stats["avg_union_tokens"])
                task_reuse.append(stats["reuse_rate"])
                if "avg_achieved_attention_coverage" in stats:
                    task_achieved_coverage.append(
                        stats["avg_achieved_attention_coverage"]
                    )
                if "coverage_saturation_rate" in stats:
                    task_coverage_saturation.append(
                        stats["coverage_saturation_rate"]
                    )
                out.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "prediction": prediction,
                            "answers": answers,
                            "score": score,
                            **stats,
                            **({"debug_trace": debug_payload} if debug_payload is not None else {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
        summary[task] = mean(task_scores)
        print(
            {
                "task": task,
                "score": summary[task],
                "num_samples": len(task_scores),
                "avg_selected_blocks": mean(task_blocks),
                **(
                    {"avg_selected_middle_blocks": mean(task_middle_blocks)}
                    if task_middle_blocks
                    else {}
                ),
                "avg_selected_tokens": mean(task_tokens),
                **(
                    {"avg_union_blocks": mean(task_union_blocks)}
                    if task_union_blocks
                    else {}
                ),
                **(
                    {"avg_union_tokens": mean(task_union_tokens)}
                    if task_union_tokens
                    else {}
                ),
                "avg_reuse_rate": mean(task_reuse),
                **(
                    {
                        "avg_achieved_attention_coverage": mean(
                            task_achieved_coverage
                        )
                    }
                    if task_achieved_coverage
                    else {}
                ),
                **(
                    {
                        "coverage_saturation_rate": mean(
                            task_coverage_saturation
                        )
                    }
                    if task_coverage_saturation
                    else {}
                ),
            }
        )

    summary["average"] = mean(list(summary.values()))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
    print(summary)


if __name__ == "__main__":
    main()
