from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import AdaBlockPolicy, AdaBlockPolicyConfig
from utils.block_oracle import (
    OracleConfig,
    block_categories,
    category_target,
    cosine_block_scores,
    make_block_ranges,
    score_summary_features,
)
from utils.longbench_eval import (
    build_model_input_text,
    format_longbench_prompt,
    load_dtype,
    load_local_task,
    load_longbench_task,
    normalize_device_map,
    truncate_middle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AdaBlock block-selection coverage against full attention.")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["hotpotqa", "musique"])
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default="results/adablock_oracle_coverage.json")
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--reuse-threshold", type=float, default=0.5)
    parser.add_argument(
        "--fill-budget-with-score",
        action="store_true",
        help="After AdaBlock selects blocks, fill remaining budget with cosine-score top blocks.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-nonfinite-docs", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        return load_local_task(Path(args.local_data_dir) / f"{task}.jsonl")
    return load_longbench_task(task, split=args.split)


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


def topk_coverage(block_mass: torch.Tensor, k: int) -> tuple[float, set[int]]:
    if block_mass.numel() == 0 or k <= 0:
        return 0.0, set()
    k = min(k, block_mass.numel())
    selected = torch.topk(block_mass, k=k).indices.tolist()
    total = block_mass.sum().clamp_min(1e-8)
    return float((block_mass[selected].sum() / total).item()), set(selected)


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
        category_scores = scores[candidates]
        take = min(k, candidates.numel())
        top_local = torch.topk(category_scores, k=take).indices
        selected.update(candidates[top_local].tolist())

    if len(selected) < budget:
        sorted_global = torch.argsort(scores, descending=True).tolist()
        for idx in sorted_global:
            selected.add(idx)
            if len(selected) >= budget:
                break
    return selected


def coverage_for_selection(block_mass: torch.Tensor, selected: set[int]) -> float:
    if not selected:
        return 0.0
    valid = [idx for idx in selected if 0 <= idx < block_mass.numel()]
    if not valid:
        return 0.0
    total = block_mass.sum().clamp_min(1e-8)
    return float((block_mass[valid].sum() / total).item())


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
        return set(list(filled)[:target])
    for idx in torch.argsort(scores, descending=True).tolist():
        filled.add(idx)
        if len(filled) >= target:
            break
    return filled


def main() -> None:
    args = parse_args()
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy_checkpoint, policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
        attn_implementation="eager",
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    max_blocks = args.budget_tokens // args.block_size
    oracle_config = OracleConfig(block_size=args.block_size)
    bucket_values = list(policy.config.budget_buckets)
    results: dict[str, dict[str, float | int]] = {}

    for task in args.tasks:
        rows = load_rows(args, task)
        if args.max_samples is not None:
            rows = list(rows)[: args.max_samples]

        total_steps = 0
        skipped_docs = 0
        fixed_cov_sum = 0.0
        score_topk_cov_sum = 0.0
        adablock_cov_sum = 0.0
        reuse_cov_sum = 0.0
        reuse_steps = 0
        adablock_blocks_sum = 0.0
        adablock_filled_cov_sum = 0.0
        reuse_filled_cov_sum = 0.0
        adablock_filled_blocks_sum = 0.0
        reuse_filled_blocks_sum = 0.0

        for row in tqdm(rows, desc=f"coverage:{task}", unit="sample"):
            prompt = format_longbench_prompt(task, row)
            model_input_text = build_model_input_text(
                tokenizer,
                prompt,
                use_chat_template=not args.no_chat_template,
            )
            encoded = tokenizer(model_input_text, return_tensors="pt", truncation=False)
            input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)
            with torch.no_grad():
                transformer = getattr(model, "model", model)
                outputs = transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                )

            hidden = outputs.last_hidden_state[0].detach().cpu().float()
            attentions = torch.stack([attn[0].detach().cpu().float() for attn in outputs.attentions])
            del outputs
            if not torch.isfinite(hidden).all() or not torch.isfinite(attentions).all():
                if args.skip_nonfinite_docs:
                    skipped_docs += 1
                    continue
                raise FloatingPointError(f"Non-finite model outputs in task={task}")

            mean_attention = attentions.mean(dim=(0, 1))
            block_ranges = make_block_ranges(hidden.shape[0], args.block_size)
            num_blocks = len(block_ranges)
            global_hit_mass = torch.zeros(num_blocks)
            previous_hidden = None
            previous_adablock_selection: set[int] = set()

            for t in range(1, hidden.shape[0], args.sample_stride):
                current_block = t // args.block_size
                candidate_ranges = block_ranges[: current_block + 1]
                if len(candidate_ranges) <= 1:
                    continue
                block_mass = torch.stack(
                    [
                        mean_attention[t, start : min(end, t + 1)].sum()
                        for start, end in candidate_ranges
                    ]
                )
                total_mass = block_mass.sum().clamp_min(1e-8)
                block_mass = block_mass / total_mass
                global_hit_mass[: block_mass.numel()] += block_mass

                fixed_coverage, _ = topk_coverage(block_mass, max_blocks)
                scores = cosine_block_scores(hidden, candidate_ranges, t)
                score_topk_indices = set(torch.topk(scores, k=min(max_blocks, scores.numel())).indices.tolist())
                score_topk_coverage = coverage_for_selection(block_mass, score_topk_indices)
                score_features = score_summary_features(scores).unsqueeze(0)
                if previous_hidden is None:
                    query_drift = torch.tensor([1.0])
                else:
                    query_drift = torch.tensor(
                        [
                            1.0
                            - torch.nn.functional.cosine_similarity(
                                hidden[t].unsqueeze(0), previous_hidden.unsqueeze(0), dim=-1
                            ).item()
                        ]
                    )

                with torch.no_grad():
                    policy_outputs = policy(
                        hidden_state=hidden[t].unsqueeze(0).to(policy_device),
                        query_drift=query_drift.to(policy_device),
                        score_features=score_features.to(policy_device),
                    )
                bucket_idx = int(policy_outputs["budget_prob"].argmax(dim=-1).item())
                predicted_k = min(bucket_values[bucket_idx], max_blocks)
                high_hit_count = max(1, int(0.1 * num_blocks))
                high_hit_blocks = set(torch.topk(global_hit_mass, k=high_hit_count).indices.tolist())
                categories = block_categories(
                    num_blocks=block_mass.numel(),
                    prompt_blocks=num_blocks,
                    current_block=current_block,
                    high_hit_blocks=high_hit_blocks,
                    config=oracle_config,
                )
                current_selection = category_allocate(
                    scores=scores,
                    categories=categories,
                    category_prob=policy_outputs["category_prob"][0].detach().cpu(),
                    budget=predicted_k,
                    max_blocks=max_blocks,
                )
                reuse = float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold
                if reuse and previous_adablock_selection:
                    reused_selection = set(previous_adablock_selection)
                    reuse_steps += 1
                else:
                    reused_selection = current_selection
                if args.fill_budget_with_score:
                    filled_selection = fill_selection_with_scores(
                        current_selection,
                        scores,
                        budget=max_blocks,
                        max_blocks=max_blocks,
                    )
                    reused_filled_selection = fill_selection_with_scores(
                        reused_selection,
                        scores,
                        budget=max_blocks,
                        max_blocks=max_blocks,
                    )

                fixed_cov_sum += fixed_coverage
                score_topk_cov_sum += score_topk_coverage
                adablock_cov_sum += coverage_for_selection(block_mass, current_selection)
                reuse_cov_sum += coverage_for_selection(block_mass, reused_selection)
                adablock_blocks_sum += len(reused_selection)
                if args.fill_budget_with_score:
                    adablock_filled_cov_sum += coverage_for_selection(block_mass, filled_selection)
                    reuse_filled_cov_sum += coverage_for_selection(block_mass, reused_filled_selection)
                    adablock_filled_blocks_sum += len(filled_selection)
                    reuse_filled_blocks_sum += len(reused_filled_selection)
                total_steps += 1
                previous_hidden = hidden[t]
                previous_adablock_selection = current_selection

            del hidden, attentions, mean_attention
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[task] = {
            "num_steps": total_steps,
            "skipped_docs": skipped_docs,
            "fixed_oracle_topk_coverage": fixed_cov_sum / max(total_steps, 1),
            "score_topk_coverage": score_topk_cov_sum / max(total_steps, 1),
            "adablock_coverage": adablock_cov_sum / max(total_steps, 1),
            "adablock_reuse_coverage": reuse_cov_sum / max(total_steps, 1),
            "adablock_reuse_rate": reuse_steps / max(total_steps, 1),
            "avg_adablock_blocks": adablock_blocks_sum / max(total_steps, 1),
            "avg_adablock_tokens": adablock_blocks_sum * args.block_size / max(total_steps, 1),
        }
        if args.fill_budget_with_score:
            results[task].update(
                {
                    "adablock_filled_coverage": adablock_filled_cov_sum / max(total_steps, 1),
                    "adablock_reuse_filled_coverage": reuse_filled_cov_sum / max(total_steps, 1),
                    "avg_adablock_filled_blocks": adablock_filled_blocks_sum / max(total_steps, 1),
                    "avg_adablock_filled_tokens": adablock_filled_blocks_sum
                    * args.block_size
                    / max(total_steps, 1),
                    "avg_adablock_reuse_filled_blocks": reuse_filled_blocks_sum / max(total_steps, 1),
                    "avg_adablock_reuse_filled_tokens": reuse_filled_blocks_sum
                    * args.block_size
                    / max(total_steps, 1),
                }
            )
        print({"task": task, **results[task]})

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(results, out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
