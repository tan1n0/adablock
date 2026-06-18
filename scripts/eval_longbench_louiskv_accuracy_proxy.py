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

from scripts.eval_longbench_adablock_sparse import (  # noqa: E402
    append_last_kv,
    blocks_to_token_indices,
    get_layer_devices,
    normalize_past_key_values,
    quest_block_scores,
    sample_next_token,
    slice_past_key_values,
    to_model_cache,
)
from utils.block_oracle import cosine_block_scores, make_block_ranges  # noqa: E402
from utils.longbench_eval import (  # noqa: E402
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
    parser = argparse.ArgumentParser(
        description=(
            "Run a LouisKV-style accuracy proxy on LongBench. This script only "
            "models the precision-relevant visible-KV policy; it does not "
            "implement LouisKV offload, custom kernels, or heterogeneous "
            "per-layer cache lengths."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/longbench_louiskv_accuracy_proxy")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--longbench-cache-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--retrieval-budget-tokens",
        type=int,
        default=512,
        help="Semantic retrieval budget. Sink/local tokens are added outside this budget.",
    )
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=32,
        help="Always-visible prefix tokens.",
    )
    parser.add_argument(
        "--local-window-tokens",
        type=int,
        default=512,
        help="Always-visible recent tokens.",
    )
    parser.add_argument(
        "--selection-score",
        default="cosine",
        choices=["cosine", "quest"],
        help="Block scoring proxy used for semantic retrieval.",
    )
    parser.add_argument(
        "--refresh-stride",
        type=int,
        default=1,
        help="Recompute retrieved blocks every N decoding steps. 1 is an accuracy upper proxy.",
    )
    parser.add_argument(
        "--reuse-similarity-threshold",
        type=float,
        default=None,
        help=(
            "Optional query cosine threshold for segment reuse. If set, reuse the previous "
            "retrieved blocks while similarity stays at or above this value."
        ),
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", None],
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


def first_n_blocks(token_count: int, block_size: int, num_tokens: int) -> set[int]:
    if token_count <= 0 or num_tokens <= 0:
        return set()
    block_count = (min(token_count, num_tokens) + block_size - 1) // block_size
    return set(range(block_count))


def last_n_blocks(current_block: int, block_size: int, num_tokens: int) -> set[int]:
    if current_block < 0 or num_tokens <= 0:
        return set()
    block_count = (num_tokens + block_size - 1) // block_size
    start = max(0, current_block - block_count + 1)
    return set(range(start, current_block + 1))


def score_blocks(
    hidden_history: torch.Tensor,
    candidate_ranges: list[tuple[int, int]],
    current_token_index: int,
    selection_score: str,
) -> torch.Tensor:
    if selection_score == "quest":
        return quest_block_scores(hidden_history, candidate_ranges, current_token_index)
    return cosine_block_scores(hidden_history, candidate_ranges, current_token_index)


def select_louiskv_proxy_blocks(
    hidden_history: torch.Tensor,
    candidate_ranges: list[tuple[int, int]],
    current_token_index: int,
    args: argparse.Namespace,
    previous_query: torch.Tensor | None,
    previous_retrieved_blocks: set[int],
    step_idx: int,
) -> tuple[set[int], set[int], bool]:
    current_block = current_token_index // args.block_size
    sink_blocks = first_n_blocks(
        token_count=hidden_history.shape[0],
        block_size=args.block_size,
        num_tokens=args.sink_tokens,
    )
    local_blocks = last_n_blocks(
        current_block=current_block,
        block_size=args.block_size,
        num_tokens=args.local_window_tokens,
    )
    required_blocks = {
        block
        for block in (sink_blocks | local_blocks)
        if 0 <= block < len(candidate_ranges)
    }

    should_refresh = step_idx % max(args.refresh_stride, 1) == 0 or not previous_retrieved_blocks
    if args.reuse_similarity_threshold is not None and previous_query is not None:
        similarity = torch.nn.functional.cosine_similarity(
            hidden_history[current_token_index].float().unsqueeze(0),
            previous_query.float().unsqueeze(0),
            dim=-1,
        ).item()
        if similarity < args.reuse_similarity_threshold:
            should_refresh = True

    retrieval_budget_blocks = max(0, args.retrieval_budget_tokens // args.block_size)
    reused = not should_refresh
    if should_refresh or retrieval_budget_blocks <= 0:
        scores = score_blocks(
            hidden_history=hidden_history,
            candidate_ranges=candidate_ranges,
            current_token_index=current_token_index,
            selection_score=args.selection_score,
        )
        eligible_scores = scores.clone()
        for block in required_blocks:
            if 0 <= block < eligible_scores.numel():
                eligible_scores[block] = -torch.inf
        eligible_count = int(torch.isfinite(eligible_scores).sum().item())
        take = min(retrieval_budget_blocks, eligible_count)
        if take > 0:
            retrieved_blocks = set(torch.topk(eligible_scores, k=take).indices.tolist())
        else:
            retrieved_blocks = set()
        reused = False
    else:
        retrieved_blocks = {
            block
            for block in previous_retrieved_blocks
            if 0 <= block < len(candidate_ranges)
        }

    selected_blocks = required_blocks | retrieved_blocks
    return selected_blocks, retrieved_blocks, reused


def run_louiskv_accuracy_proxy_decode(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, float]]:
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    max_new_tokens = args.max_new_tokens_override
    if max_new_tokens is None:
        raise ValueError("run_louiskv_accuracy_proxy_decode requires task max_new_tokens")

    transformer = getattr(model, "model", model)
    with torch.no_grad():
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    full_past_key_values = tuple(
        (key.detach().cpu(), value.detach().cpu())
        for key, value in normalize_past_key_values(outputs.past_key_values)
    )
    hidden_history = outputs.last_hidden_state[0].detach().cpu().float()
    last_logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, -1]
    next_token_id = sample_next_token(last_logits, args.temperature, args.top_p)
    generated_ids = [next_token_id]
    current_input_token = generated_ids[-1]

    layer_devices = get_layer_devices(model)
    previous_query: torch.Tensor | None = None
    previous_retrieved_blocks: set[int] = set()
    selected_blocks_sum = 0.0
    selected_tokens_sum = 0.0
    retrieved_blocks_sum = 0.0
    selected_steps = 0
    reuse_steps = 0

    for step_idx in range(max_new_tokens - 1):
        history_len = hidden_history.shape[0]
        current_token_index = history_len - 1
        block_ranges = make_block_ranges(history_len, args.block_size)
        candidate_ranges = block_ranges[: current_token_index // args.block_size + 1]

        selected_blocks, retrieved_blocks, reused = select_louiskv_proxy_blocks(
            hidden_history=hidden_history,
            candidate_ranges=candidate_ranges,
            current_token_index=current_token_index,
            args=args,
            previous_query=previous_query,
            previous_retrieved_blocks=previous_retrieved_blocks,
            step_idx=step_idx,
        )
        token_indices = blocks_to_token_indices(selected_blocks, candidate_ranges)
        if not token_indices:
            token_indices = list(range(history_len))

        selected_steps += 1
        selected_blocks_sum += len(selected_blocks)
        selected_tokens_sum += len(token_indices)
        retrieved_blocks_sum += len(retrieved_blocks)
        reuse_steps += int(reused)

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
        previous_query = hidden_history[current_token_index]
        previous_retrieved_blocks = set(retrieved_blocks)
        hidden_history = torch.cat([hidden_history, new_hidden], dim=0)

        step_logits = model.lm_head(step_outputs.last_hidden_state[:, -1:, :])[0, -1]
        next_token_id = sample_next_token(step_logits, args.temperature, args.top_p)
        generated_ids.append(next_token_id)
        current_input_token = next_token_id
        if next_token_id == tokenizer.eos_token_id:
            break

        del sparse_past, step_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stats = {
        "avg_selected_blocks": selected_blocks_sum / max(selected_steps, 1),
        "avg_selected_tokens": selected_tokens_sum / max(selected_steps, 1),
        "avg_retrieved_blocks": retrieved_blocks_sum / max(selected_steps, 1),
        "avg_retrieved_tokens": retrieved_blocks_sum * args.block_size / max(selected_steps, 1),
        "reuse_rate": reuse_steps / max(selected_steps, 1),
        "generated_tokens": len(generated_ids),
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "used_prompt_tokens": int(input_ids.shape[-1]),
        "retrieval_budget_tokens": int(args.retrieval_budget_tokens),
        "sink_tokens": int(args.sink_tokens),
        "local_window_tokens": int(args.local_window_tokens),
    }
    return prediction, stats


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
        task_retrieved_tokens: list[float] = []
        task_reuse: list[float] = []
        pred_path = output_dir / f"{task}.jsonl"
        task_args = argparse.Namespace(**vars(args))
        task_args.max_new_tokens_override = args.max_new_tokens_override or TASK_MAX_NEW_TOKENS[task]
        with pred_path.open("w", encoding="utf-8") as out:
            for idx, row in enumerate(tqdm(rows, desc=f"louiskv_proxy:{task}", unit="sample")):
                prompt = format_longbench_prompt(task, row)
                model_input_text = build_model_input_text(
                    tokenizer,
                    prompt,
                    use_chat_template=not args.no_chat_template,
                )
                prediction, stats = run_louiskv_accuracy_proxy_decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=model_input_text,
                    args=task_args,
                )
                answers = extract_answers(row)
                score = score_prediction(task, prediction, answers)
                task_scores.append(score)
                task_blocks.append(stats["avg_selected_blocks"])
                task_tokens.append(stats["avg_selected_tokens"])
                task_retrieved_tokens.append(stats["avg_retrieved_tokens"])
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
                "avg_retrieved_tokens": mean(task_retrieved_tokens),
                "avg_reuse_rate": mean(task_reuse),
            }
        )

    summary["average"] = mean(list(summary.values()))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
    print(summary)


if __name__ == "__main__":
    main()
