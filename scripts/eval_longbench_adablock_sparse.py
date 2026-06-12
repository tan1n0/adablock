from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import AdaBlockPolicy, AdaBlockPolicyConfig
from utils.block_oracle import (
    OracleConfig,
    aggregate_block_mass,
    block_categories,
    cosine_block_scores,
    make_block_ranges,
    score_summary_features,
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
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/longbench_adablock_sparse")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--longbench-cache-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--local-window-blocks", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--reuse-threshold", type=float, default=0.8)
    parser.add_argument("--fill-budget-with-score", action="store_true")
    parser.add_argument(
        "--selection-mode",
        default="policy",
        choices=["policy", "oracle_topk"],
        help="Block selection strategy for sparse decoding.",
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
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        path = Path(args.local_data_dir) / f"{task}.jsonl"
        return load_local_task(path)
    return load_longbench_task(task, split=args.split, cache_dir=args.longbench_cache_dir)


def load_policy(path: str | Path, device: torch.device) -> AdaBlockPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    config_dict = dict(checkpoint["config"])
    if "budget_buckets" in config_dict:
        config_dict["budget_buckets"] = tuple(config_dict["budget_buckets"])
    policy = AdaBlockPolicy(AdaBlockPolicyConfig(**config_dict))
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    policy.eval()
    return policy


def category_allocate(
    scores: torch.Tensor,
    categories: torch.Tensor,
    category_prob: torch.Tensor,
    budget: int,
    max_blocks: int,
) -> set[int]:
    if scores.numel() == 0 or budget <= 0:
        return set()
    budget = min(budget, max_blocks, scores.numel())
    raw_alloc = torch.floor(category_prob.cpu() * budget).long()
    remaining = budget - int(raw_alloc.sum().item())
    if remaining > 0:
        order = torch.argsort(category_prob.cpu(), descending=True).tolist()
        for idx in order[:remaining]:
            raw_alloc[idx] += 1

    selected: set[int] = set()
    for category_idx, k in enumerate(raw_alloc.tolist()):
        if k <= 0:
            continue
        candidates = torch.nonzero(categories == category_idx, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        candidate_scores = scores[candidates]
        take = min(k, candidates.numel())
        top_local = torch.topk(candidate_scores, k=take).indices
        selected.update(candidates[top_local].tolist())

    if len(selected) < budget:
        for idx in torch.argsort(scores, descending=True).tolist():
            selected.add(idx)
            if len(selected) >= budget:
                break
    return selected


def fill_selection_with_scores(
    selected: set[int],
    scores: torch.Tensor,
    budget: int,
    max_blocks: int,
) -> set[int]:
    if scores.numel() == 0 or budget <= 0:
        return set()
    target = min(budget, max_blocks, scores.numel())
    filled = {idx for idx in selected if 0 <= idx < scores.numel()}
    if len(filled) >= target:
        return set(sorted(filled)[:target])
    for idx in torch.argsort(scores, descending=True).tolist():
        filled.add(idx)
        if len(filled) >= target:
            break
    return filled


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


def run_sparse_decode(
    model,
    tokenizer,
    policy: AdaBlockPolicy,
    policy_device: torch.device,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, float]]:
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

    oracle_config = OracleConfig(block_size=args.block_size, local_window_blocks=args.local_window_blocks)
    bucket_values = list(policy.config.budget_buckets)
    max_blocks = args.budget_tokens // args.block_size
    layer_devices = get_layer_devices(model)
    previous_hidden = None
    previous_selection: set[int] = set()
    selected_blocks_sum = 0.0
    selected_steps = 0
    reuse_steps = 0

    for _ in range(max_new_tokens - 1):
        history_len = hidden_history.shape[0]
        current_token_index = history_len - 1
        current_block = current_token_index // args.block_size
        block_ranges = make_block_ranges(history_len, args.block_size)
        candidate_ranges = block_ranges[: current_block + 1]
        if len(candidate_ranges) <= 1:
            token_indices = list(range(history_len))
            selected_blocks = set(range(len(candidate_ranges)))
        else:
            scores = cosine_block_scores(hidden_history, candidate_ranges, current_token_index)
            if args.selection_mode == "oracle_topk":
                dense_token_indices = list(range(history_len))
                dense_past = slice_past_key_values(full_past_key_values, dense_token_indices, layer_devices)
                dense_step_input_ids = torch.tensor([[generated_ids[-1]]], device=model.device)
                dense_attention_mask = torch.ones((1, history_len + 1), dtype=torch.long, device=model.device)
                dense_position_ids = torch.tensor([[history_len]], dtype=torch.long, device=model.device)
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
                if dense_outputs.attentions is None:
                    raise RuntimeError(
                        "oracle_topk selection requires attention weights, but the model backend returned None. "
                        "Use eager attention for oracle_topk evaluation."
                    )
                dense_attentions = torch.stack([attn[0].detach().cpu().float() for attn in dense_outputs.attentions])
                dense_mean_attention = dense_attentions.mean(dim=(0, 1))
                token_attention = dense_mean_attention[0, :history_len]
                block_mass = aggregate_block_mass(token_attention, candidate_ranges)
                block_mass = block_mass / block_mass.sum().clamp_min(1e-8)
                selected_blocks = set(top_blocks(block_mass, max_blocks))
                dense_step_past = normalize_past_key_values(dense_outputs.past_key_values)
                full_past_key_values = append_last_kv(full_past_key_values, dense_step_past)
                new_hidden = dense_outputs.last_hidden_state[0, -1].detach().cpu().float().unsqueeze(0)
                hidden_history = torch.cat([hidden_history, new_hidden], dim=0)
                previous_hidden = hidden_history[-2]
                token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
            else:
                score_features = score_summary_features(scores).unsqueeze(0)
                if previous_hidden is None:
                    query_drift = torch.tensor([1.0])
                else:
                    query_drift = torch.tensor(
                        [
                            1.0
                            - torch.nn.functional.cosine_similarity(
                                hidden_history[current_token_index].unsqueeze(0), previous_hidden.unsqueeze(0), dim=-1
                            ).item()
                        ]
                    )
                high_hit_count = max(1, int(0.1 * len(candidate_ranges)))
                high_hit_blocks = set(torch.topk(scores, k=min(high_hit_count, scores.numel())).indices.tolist())
                categories = block_categories(
                    num_blocks=len(candidate_ranges),
                    prompt_blocks=len(block_ranges),
                    current_block=current_block,
                    high_hit_blocks=high_hit_blocks,
                    config=oracle_config,
                )
                with torch.no_grad():
                    policy_outputs = policy(
                        hidden_state=hidden_history[current_token_index].unsqueeze(0).to(policy_device),
                        query_drift=query_drift.to(policy_device),
                        score_features=score_features.to(policy_device),
                    )
                bucket_idx = int(policy_outputs["budget_prob"].argmax(dim=-1).item())
                predicted_k = min(bucket_values[bucket_idx], max_blocks)
                selected_blocks = category_allocate(
                    scores=scores,
                    categories=categories,
                    category_prob=policy_outputs["category_prob"][0].detach().cpu(),
                    budget=predicted_k,
                    max_blocks=max_blocks,
                )
                selected_blocks = enforce_local_window(selected_blocks, current_block, args.local_window_blocks)
                reuse = float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold
                if reuse and previous_selection:
                    selected_blocks = set(previous_selection)
                    reuse_steps += 1
                if args.fill_budget_with_score:
                    selected_blocks = fill_selection_with_scores(
                        selected_blocks,
                        scores,
                        budget=max_blocks,
                        max_blocks=max_blocks,
                    )
                token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
                previous_selection = set(selected_blocks)

        selected_blocks_sum += len(selected_blocks)
        selected_steps += 1
        sparse_past = slice_past_key_values(full_past_key_values, token_indices, layer_devices)
        step_input_ids = torch.tensor([[generated_ids[-1]]], device=model.device)
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

        if args.selection_mode == "policy":
            step_past = normalize_past_key_values(step_outputs.past_key_values)
            full_past_key_values = append_last_kv(full_past_key_values, step_past)
            new_hidden = step_outputs.last_hidden_state[0, -1].detach().cpu().float().unsqueeze(0)
            hidden_history = torch.cat([hidden_history, new_hidden], dim=0)
            previous_hidden = hidden_history[-2]
        step_logits = model.lm_head(step_outputs.last_hidden_state[:, -1:, :])[0, -1]
        next_token_id = sample_next_token(step_logits, args.temperature, args.top_p)
        generated_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break
        del sparse_past, step_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stats = {
        "avg_selected_blocks": selected_blocks_sum / max(selected_steps, 1),
        "avg_selected_tokens": selected_blocks_sum * args.block_size / max(selected_steps, 1),
        "reuse_rate": reuse_steps / max(selected_steps, 1),
        "generated_tokens": len(generated_ids),
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "used_prompt_tokens": int(input_ids.shape[-1]),
    }
    return prediction, stats


def main() -> None:
    args = parse_args()
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy_checkpoint, policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    attn_impl = args.attn_implementation
    if args.selection_mode == "oracle_topk" and attn_impl != "eager":
        print(
            {
                "event": "force_eager_attention",
                "reason": "oracle_topk needs explicit attention weights",
                "requested_attn_implementation": attn_impl,
                "used_attn_implementation": "eager",
            }
        )
        attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
        attn_implementation=attn_impl,
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {}

    for task in args.tasks:
        rows = load_rows(args, task)
        if args.max_samples is not None:
            rows = list(rows)[: args.max_samples]
        task_scores: list[float] = []
        task_blocks: list[float] = []
        task_tokens: list[float] = []
        task_reuse: list[float] = []
        pred_path = output_dir / f"{task}.jsonl"
        task_args = argparse.Namespace(**vars(args))
        task_args.max_new_tokens_override = args.max_new_tokens_override or TASK_MAX_NEW_TOKENS[task]
        with pred_path.open("w", encoding="utf-8") as out:
            for idx, row in enumerate(tqdm(rows, desc=f"sparse:{task}", unit="sample")):
                prompt = format_longbench_prompt(task, row)
                model_input_text = build_model_input_text(
                    tokenizer,
                    prompt,
                    use_chat_template=not args.no_chat_template,
                )
                prediction, stats = run_sparse_decode(
                    model=model,
                    tokenizer=tokenizer,
                    policy=policy,
                    policy_device=policy_device,
                    prompt_text=model_input_text,
                    args=task_args,
                )
                answers = extract_answers(row)
                score = score_prediction(task, prediction, answers)
                task_scores.append(score)
                task_blocks.append(stats["avg_selected_blocks"])
                task_tokens.append(stats["avg_selected_tokens"])
                task_reuse.append(stats["reuse_rate"])
                out.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "prediction": prediction,
                            "answers": answers,
                            "score": score,
                            **stats,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary[task] = mean(task_scores)
        print(
            {
                "task": task,
                "score": summary[task],
                "num_samples": len(task_scores),
                "avg_selected_blocks": mean(task_blocks),
                "avg_selected_tokens": mean(task_tokens),
                "avg_reuse_rate": mean(task_reuse),
            }
        )

    summary["average"] = mean(list(summary.values()))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
    print(summary)


if __name__ == "__main__":
    main()
