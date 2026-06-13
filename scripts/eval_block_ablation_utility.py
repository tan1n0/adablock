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

from utils.block_oracle import aggregate_block_mass, cosine_block_scores, make_block_ranges, top_blocks
from utils.longbench_eval import (
    LONG_INPUT_SHORT_OUTPUT_TASKS,
    build_model_input_text,
    extract_answers,
    format_longbench_prompt,
    load_dtype,
    load_local_task,
    load_longbench_task,
    normalize_device_map,
    truncate_middle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate block utility via single-block ablation on LongBench.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["hotpotqa"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--longbench-cache-dir", default=None)
    parser.add_argument("--output-jsonl", default="results/block_ablation_utility.jsonl")
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-steps-per-sample", type=int, default=4)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--candidate-mode",
        default="score_topk",
        choices=["score_topk", "attention_topk", "last_blocks"],
        help="How to choose candidate blocks to ablate before measuring utility.",
    )
    parser.add_argument("--candidate-blocks", type=int, default=16)
    parser.add_argument("--compare-topk", type=int, default=8)
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        return load_local_task(Path(args.local_data_dir) / f"{task}.jsonl")
    return load_longbench_task(task, split=args.split, cache_dir=args.longbench_cache_dir)


def choose_candidate_blocks(
    mode: str,
    hidden_history: torch.Tensor,
    mean_attention: torch.Tensor,
    current_index: int,
    block_ranges: list[tuple[int, int]],
    candidate_blocks: int,
) -> list[int]:
    if not block_ranges:
        return []
    if mode == "last_blocks":
        return list(range(max(0, len(block_ranges) - candidate_blocks), len(block_ranges)))

    if mode == "score_topk":
        scores = cosine_block_scores(hidden_history, block_ranges, current_index)
        k = min(candidate_blocks, scores.numel())
        return torch.topk(scores, k=k).indices.tolist()

    token_attention = mean_attention[current_index, : current_index + 1]
    block_mass = aggregate_block_mass(token_attention, block_ranges)
    return top_blocks(block_mass, candidate_blocks)


def ablate_history_tokens(
    history_ids: torch.Tensor,
    block_ranges: list[tuple[int, int]],
    remove_block: int,
) -> torch.Tensor:
    if not (0 <= remove_block < len(block_ranges)):
        return history_ids
    start, end = block_ranges[remove_block]
    keep_ids = torch.cat([history_ids[:, :start], history_ids[:, end:]], dim=-1)
    return keep_ids


def run_full_step(
    model,
    input_ids: torch.Tensor,
    target_token_id: int,
    need_attentions: bool,
    need_hidden: bool,
) -> dict[str, object]:
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=need_attentions,
            output_hidden_states=need_hidden,
            use_cache=False,
        )
    last_logits = outputs.logits[0, -1].detach().cpu().float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    target_logit = float(last_logits[target_token_id].item())
    target_loss = float((-log_probs[target_token_id]).item())
    result: dict[str, object] = {
        "target_logit": target_logit,
        "target_loss": target_loss,
        "top_prediction_id": int(torch.argmax(last_logits).item()),
    }
    if need_hidden:
        result["hidden_history"] = outputs.hidden_states[-1][0].detach().cpu().float()
    if need_attentions:
        attentions = torch.stack([attn[0].detach().cpu().float() for attn in outputs.attentions])
        result["mean_attention"] = attentions.mean(dim=(0, 1))
    return result


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
        attn_implementation=args.attn_implementation,
    )
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for task in args.tasks:
            rows = load_rows(args, task)
            if args.max_samples is not None:
                rows = list(rows)[: args.max_samples]

            for sample_idx, row in enumerate(tqdm(rows, desc=f"ablation:{task}", unit="sample")):
                prompt = format_longbench_prompt(task, row)
                model_input_text = build_model_input_text(
                    tokenizer,
                    prompt,
                    use_chat_template=not args.no_chat_template,
                )
                answers = extract_answers(row)
                if not answers:
                    continue
                teacher_answer = str(answers[0]).strip()
                if not teacher_answer:
                    continue
                answer_ids = tokenizer(teacher_answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
                if answer_ids.numel() == 0:
                    continue

                prompt_ids = tokenizer(model_input_text, return_tensors="pt", truncation=False)["input_ids"]
                max_steps = min(args.max_steps_per_sample, int(answer_ids.shape[-1]))

                for answer_step in range(0, max_steps, args.step_stride):
                    history_ids = torch.cat([prompt_ids, answer_ids[:, :answer_step]], dim=-1)
                    history_ids = truncate_middle(history_ids, args.max_input_length).to(model.device)
                    target_token_id = int(answer_ids[0, answer_step].item())

                    full_step = run_full_step(
                        model=model,
                        input_ids=history_ids,
                        target_token_id=target_token_id,
                        need_attentions=args.candidate_mode == "attention_topk",
                        need_hidden=True,
                    )
                    hidden_history = full_step["hidden_history"]
                    mean_attention = full_step.get("mean_attention")
                    current_index = int(history_ids.shape[-1] - 1)
                    block_ranges = make_block_ranges(int(history_ids.shape[-1]), args.block_size)
                    candidate_ids = choose_candidate_blocks(
                        mode=args.candidate_mode,
                        hidden_history=hidden_history,
                        mean_attention=mean_attention if isinstance(mean_attention, torch.Tensor) else torch.empty(0),
                        current_index=current_index,
                        block_ranges=block_ranges,
                        candidate_blocks=args.candidate_blocks,
                    )

                    utilities: list[dict[str, object]] = []
                    for block_id in candidate_ids:
                        ablated_ids = ablate_history_tokens(history_ids, block_ranges, block_id)
                        ablated_step = run_full_step(
                            model=model,
                            input_ids=ablated_ids,
                            target_token_id=target_token_id,
                            need_attentions=False,
                            need_hidden=False,
                        )
                        utilities.append(
                            {
                                "block_id": block_id,
                                "block_token_start": block_ranges[block_id][0],
                                "block_token_end": block_ranges[block_id][1],
                                "loss_delta": ablated_step["target_loss"] - full_step["target_loss"],
                                "logit_delta": full_step["target_logit"] - ablated_step["target_logit"],
                            }
                        )

                    utilities.sort(key=lambda item: item["loss_delta"], reverse=True)
                    top_utility_blocks = [int(item["block_id"]) for item in utilities[: args.compare_topk]]

                    attention_top_blocks: list[int] = []
                    if isinstance(mean_attention, torch.Tensor):
                        token_attention = mean_attention[current_index, : current_index + 1]
                        block_mass = aggregate_block_mass(token_attention, block_ranges)
                        attention_top_blocks = top_blocks(block_mass, args.compare_topk)

                    score_top_blocks = top_blocks(
                        cosine_block_scores(hidden_history, block_ranges, current_index),
                        args.compare_topk,
                    )

                    record = {
                        "task": task,
                        "sample_idx": sample_idx,
                        "answer_step": answer_step,
                        "prompt_tokens": int(prompt_ids.shape[-1]),
                        "used_history_tokens": int(history_ids.shape[-1]),
                        "target_token_id": target_token_id,
                        "target_token_text": token_text(tokenizer, target_token_id),
                        "full_target_logit": full_step["target_logit"],
                        "full_target_loss": full_step["target_loss"],
                        "full_top_prediction_id": full_step["top_prediction_id"],
                        "full_top_prediction_text": token_text(tokenizer, int(full_step["top_prediction_id"])),
                        "candidate_mode": args.candidate_mode,
                        "candidate_block_ids": candidate_ids,
                        "utility_top_blocks": top_utility_blocks,
                        "attention_top_blocks": attention_top_blocks,
                        "score_top_blocks": score_top_blocks,
                        "utilities": utilities,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print({"output": str(output_path), "tasks": args.tasks})


if __name__ == "__main__":
    main()
