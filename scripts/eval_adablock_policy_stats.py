from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.adablock_policy import AdaBlockPolicy, AdaBlockPolicyConfig
from utils.block_oracle import cosine_block_scores, make_block_ranges, score_summary_features
from utils.longbench_eval import (
    LONG_INPUT_SHORT_OUTPUT_TASKS,
    format_longbench_prompt,
    load_dtype,
    load_local_task,
    load_longbench_task,
    normalize_device_map,
    truncate_middle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AdaBlock policy statistics on LongBench prompts.")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default="results/adablock_policy_stats.json")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--longbench-cache-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--reuse-threshold", type=float, default=0.5)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        return load_local_task(Path(args.local_data_dir) / f"{task}.jsonl")
    return load_longbench_task(task, split=args.split, cache_dir=args.longbench_cache_dir)


def load_policy(path: str | Path, device: torch.device) -> AdaBlockPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    config_dict = dict(checkpoint["config"])
    if "budget_buckets" in config_dict:
        config_dict["budget_buckets"] = tuple(config_dict["budget_buckets"])
    config = AdaBlockPolicyConfig(**config_dict)
    policy = AdaBlockPolicy(config)
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy_checkpoint, policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    max_blocks = args.budget_tokens // args.block_size
    results: dict[str, dict[str, object]] = {}
    bucket_values = list(policy.config.budget_buckets)

    for task in args.tasks:
        rows = load_rows(args, task)
        if args.max_samples is not None:
            rows = list(rows)[: args.max_samples]
        bucket_counter: Counter[int] = Counter()
        category_sum = torch.zeros(policy.config.num_categories)
        total_steps = 0
        reuse_steps = 0
        unclamped_k_sum = 0.0
        clamped_k_sum = 0.0

        for row in tqdm(rows, desc=f"policy-stats:{task}", unit="sample"):
            prompt = format_longbench_prompt(task, row)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = outputs.hidden_states[-1][0].detach().cpu().float()
            del outputs
            block_ranges = make_block_ranges(hidden.shape[0], args.block_size)
            previous_hidden = None
            for t in range(1, hidden.shape[0], args.sample_stride):
                current_block = t // args.block_size
                candidate_ranges = block_ranges[: current_block + 1]
                if len(candidate_ranges) <= 1:
                    continue
                score_features = score_summary_features(
                    cosine_block_scores(hidden, candidate_ranges, t)
                ).unsqueeze(0)
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
                predicted_k = bucket_values[bucket_idx]
                clamped_k = min(predicted_k, max_blocks)
                reuse = float(policy_outputs["reuse_prob"].item()) > args.reuse_threshold

                bucket_counter[predicted_k] += 1
                category_sum += policy_outputs["category_prob"][0].detach().cpu()
                total_steps += 1
                reuse_steps += int(reuse)
                unclamped_k_sum += predicted_k
                clamped_k_sum += clamped_k
                previous_hidden = hidden[t]

        results[task] = {
            "num_policy_steps": total_steps,
            "avg_predicted_blocks": unclamped_k_sum / max(total_steps, 1),
            "avg_clamped_blocks": clamped_k_sum / max(total_steps, 1),
            "avg_clamped_tokens": clamped_k_sum * args.block_size / max(total_steps, 1),
            "reuse_rate": reuse_steps / max(total_steps, 1),
            "budget_bucket_counts": dict(sorted(bucket_counter.items())),
            "category_mean": (category_sum / max(total_steps, 1)).tolist(),
        }
        print({"task": task, **results[task]})

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(results, out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
