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

from utils.longbench_eval import (
    LONG_INPUT_SHORT_OUTPUT_TASKS,
    TASK_MAX_NEW_TOKENS,
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
    parser = argparse.ArgumentParser(description="Run full-attention LongBench baseline.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(LONG_INPUT_SHORT_OUTPUT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/longbench_full")
    parser.add_argument("--local-data-dir", default=None)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def load_rows(args: argparse.Namespace, task: str):
    if args.local_data_dir:
        path = Path(args.local_data_dir) / f"{task}.jsonl"
        return load_local_task(path)
    return load_longbench_task(task, split=args.split)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=normalize_device_map(args.device_map),
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
        pred_path = output_dir / f"{task}.jsonl"
        with pred_path.open("w", encoding="utf-8") as out:
            for idx, row in enumerate(tqdm(rows, desc=task, unit="sample")):
                prompt = format_longbench_prompt(task, row)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
                input_ids = truncate_middle(encoded["input_ids"], args.max_input_length).to(model.device)
                attention_mask = torch.ones_like(input_ids, device=model.device)
                max_new_tokens = TASK_MAX_NEW_TOKENS[task]
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                prediction = tokenizer.decode(
                    generated[0, input_ids.shape[-1] :],
                    skip_special_tokens=True,
                ).strip()
                answers = extract_answers(row)
                score = score_prediction(task, prediction, answers)
                task_scores.append(score)
                out.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "prediction": prediction,
                            "answers": answers,
                            "score": score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary[task] = mean(task_scores)
        print({"task": task, "score": summary[task], "num_samples": len(task_scores)})

    summary["average"] = mean(list(summary.values()))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
    print(summary)


if __name__ == "__main__":
    main()
