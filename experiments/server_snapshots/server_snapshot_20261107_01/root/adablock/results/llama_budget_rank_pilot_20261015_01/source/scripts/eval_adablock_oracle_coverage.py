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
    cosine_block_scores,
    fill_selection_to_total_budget,
    fixed_and_middle_blocks,
    make_block_ranges,
    score_summary_features,
    select_middle_blocks,
    top_blocks,
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
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--local-window-blocks", type=int, default=4)
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


def coverage_for_selection(block_mass: torch.Tensor, selected: set[int]) -> float:
    if not selected:
        return 0.0
    valid = [idx for idx in selected if 0 <= idx < block_mass.numel()]
    if not valid:
        return 0.0
    total = block_mass.sum().clamp_min(1e-8)
    return float((block_mass[valid].sum() / total).item())


def overlap_stats(selected: set[int], oracle_selected: set[int]) -> dict[str, float]:
    if not selected and not oracle_selected:
        return {
            "overlap_count": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
        }
    intersection = len(selected & oracle_selected)
    union = len(selected | oracle_selected)
    return {
        "overlap_count": float(intersection),
        "precision": intersection / max(len(selected), 1),
        "recall": intersection / max(len(oracle_selected), 1),
        "jaccard": intersection / max(union, 1),
    }


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

    sink_blocks = (args.sink_tokens + args.block_size - 1) // args.block_size
    max_blocks = args.budget_tokens // args.block_size
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
        reuse_blocks_sum = 0.0
        adablock_filled_cov_sum = 0.0
        reuse_filled_cov_sum = 0.0
        adablock_filled_blocks_sum = 0.0
        reuse_filled_blocks_sum = 0.0
        score_overlap_sum = 0.0
        score_jaccard_sum = 0.0
        adablock_overlap_sum = 0.0
        adablock_jaccard_sum = 0.0
        reuse_overlap_sum = 0.0
        reuse_jaccard_sum = 0.0
        adablock_filled_overlap_sum = 0.0
        adablock_filled_jaccard_sum = 0.0
        reuse_filled_overlap_sum = 0.0
        reuse_filled_jaccard_sum = 0.0

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
            previous_hidden = None
            previous_middle_selection: set[int] = set()

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
                scores = cosine_block_scores(hidden, candidate_ranges, t)
                sink, local, middle = fixed_and_middle_blocks(
                    num_blocks=block_mass.numel(),
                    sink_blocks=sink_blocks,
                    local_window_blocks=args.local_window_blocks,
                    current_block=current_block,
                )
                fixed_blocks = sink | local
                dynamic_capacity = max(0, max_blocks - len(fixed_blocks))

                oracle_middle = set(top_blocks(block_mass, dynamic_capacity, candidates=middle))
                oracle_topk_indices = fixed_blocks | oracle_middle
                fixed_coverage = coverage_for_selection(block_mass, oracle_topk_indices)
                score_middle = select_middle_blocks(scores, middle, dynamic_capacity)
                score_topk_indices = fixed_blocks | score_middle
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
                predicted_middle_k = min(
                    bucket_values[bucket_idx], dynamic_capacity, len(middle)
                )
                current_middle = select_middle_blocks(scores, middle, predicted_middle_k)
                current_selection = fixed_blocks | current_middle
                reuse = float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold
                reusable_middle = previous_middle_selection & middle
                if reuse and reusable_middle:
                    reused_selection = fixed_blocks | reusable_middle
                    reuse_steps += 1
                else:
                    reused_selection = current_selection
                if args.fill_budget_with_score:
                    filled_selection = fill_selection_to_total_budget(
                        selected=current_selection,
                        scores=scores,
                        total_budget_blocks=max_blocks,
                        required_blocks=fixed_blocks,
                    )
                    reused_filled_selection = fill_selection_to_total_budget(
                        selected=reused_selection,
                        scores=scores,
                        total_budget_blocks=max_blocks,
                        required_blocks=fixed_blocks,
                    )

                fixed_cov_sum += fixed_coverage
                score_topk_cov_sum += score_topk_coverage
                adablock_cov_sum += coverage_for_selection(block_mass, current_selection)
                reuse_cov_sum += coverage_for_selection(block_mass, reused_selection)
                adablock_blocks_sum += len(current_selection)
                reuse_blocks_sum += len(reused_selection)
                score_overlap = overlap_stats(score_topk_indices, oracle_topk_indices)
                adablock_overlap = overlap_stats(current_selection, oracle_topk_indices)
                reuse_overlap = overlap_stats(reused_selection, oracle_topk_indices)
                score_overlap_sum += score_overlap["recall"]
                score_jaccard_sum += score_overlap["jaccard"]
                adablock_overlap_sum += adablock_overlap["recall"]
                adablock_jaccard_sum += adablock_overlap["jaccard"]
                reuse_overlap_sum += reuse_overlap["recall"]
                reuse_jaccard_sum += reuse_overlap["jaccard"]
                if args.fill_budget_with_score:
                    adablock_filled_cov_sum += coverage_for_selection(block_mass, filled_selection)
                    reuse_filled_cov_sum += coverage_for_selection(block_mass, reused_filled_selection)
                    adablock_filled_blocks_sum += len(filled_selection)
                    reuse_filled_blocks_sum += len(reused_filled_selection)
                    filled_overlap = overlap_stats(filled_selection, oracle_topk_indices)
                    reuse_filled_overlap = overlap_stats(reused_filled_selection, oracle_topk_indices)
                    adablock_filled_overlap_sum += filled_overlap["recall"]
                    adablock_filled_jaccard_sum += filled_overlap["jaccard"]
                    reuse_filled_overlap_sum += reuse_filled_overlap["recall"]
                    reuse_filled_jaccard_sum += reuse_filled_overlap["jaccard"]
                total_steps += 1
                previous_hidden = hidden[t]
                previous_middle_selection = current_selection - fixed_blocks

            del hidden, attentions, mean_attention
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[task] = {
            "num_steps": total_steps,
            "skipped_docs": skipped_docs,
            "fixed_oracle_topk_coverage": fixed_cov_sum / max(total_steps, 1),
            "score_topk_coverage": score_topk_cov_sum / max(total_steps, 1),
            "score_topk_oracle_recall": score_overlap_sum / max(total_steps, 1),
            "score_topk_oracle_jaccard": score_jaccard_sum / max(total_steps, 1),
            "adablock_coverage": adablock_cov_sum / max(total_steps, 1),
            "adablock_oracle_recall": adablock_overlap_sum / max(total_steps, 1),
            "adablock_oracle_jaccard": adablock_jaccard_sum / max(total_steps, 1),
            "adablock_reuse_coverage": reuse_cov_sum / max(total_steps, 1),
            "adablock_reuse_oracle_recall": reuse_overlap_sum / max(total_steps, 1),
            "adablock_reuse_oracle_jaccard": reuse_jaccard_sum / max(total_steps, 1),
            "adablock_reuse_rate": reuse_steps / max(total_steps, 1),
            "avg_adablock_blocks": adablock_blocks_sum / max(total_steps, 1),
            "avg_adablock_tokens": adablock_blocks_sum * args.block_size / max(total_steps, 1),
            "avg_adablock_reuse_blocks": reuse_blocks_sum / max(total_steps, 1),
            "avg_adablock_reuse_tokens": reuse_blocks_sum * args.block_size / max(total_steps, 1),
        }
        if args.fill_budget_with_score:
            results[task].update(
                {
                    "adablock_filled_coverage": adablock_filled_cov_sum / max(total_steps, 1),
                    "adablock_reuse_filled_coverage": reuse_filled_cov_sum / max(total_steps, 1),
                    "adablock_filled_oracle_recall": adablock_filled_overlap_sum / max(total_steps, 1),
                    "adablock_filled_oracle_jaccard": adablock_filled_jaccard_sum / max(total_steps, 1),
                    "adablock_reuse_filled_oracle_recall": reuse_filled_overlap_sum / max(total_steps, 1),
                    "adablock_reuse_filled_oracle_jaccard": reuse_filled_jaccard_sum / max(total_steps, 1),
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
