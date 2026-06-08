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

from utils.block_oracle import (
    OracleConfig,
    aggregate_block_mass,
    block_categories,
    category_target,
    cosine_block_scores,
    make_block_ranges,
    oracle_budget_label,
    reuse_label_from_previous,
    score_summary_features,
    top_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--reuse-coverage", type=float, default=0.9)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--device",
        default=None,
        help="Fallback device when device_map does not dispatch the model, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU oracle generation. Without this flag the script fails if CUDA is unavailable.",
    )
    parser.add_argument(
        "--skip-nonfinite-docs",
        action="store_true",
        help="Skip documents whose hidden states or attentions contain NaN/Inf.",
    )
    parser.add_argument("--log-every-doc", action="store_true")
    return parser.parse_args()


def load_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def iter_texts(path: Path, field: str, max_docs: int | None):
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_docs is not None and idx >= max_docs:
                break
            row = json.loads(line)
            text = row[field] if isinstance(row, dict) else str(row)
            if text.strip():
                yield idx, text


def main() -> None:
    args = parse_args()
    config = OracleConfig(
        block_size=args.block_size,
        coverage_threshold=args.coverage,
        reuse_coverage_threshold=args.reuse_coverage,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    cuda_available = torch.cuda.is_available()
    print(
        {
            "event": "torch_device_check",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count(),
        },
        flush=True,
    )
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment, so oracle generation would run on CPU. "
            "Install a CUDA-enabled PyTorch build or pass --allow-cpu intentionally."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation="eager",
    )
    if getattr(model, "hf_device_map", None) is None:
        fallback_device = args.device or ("cuda:0" if cuda_available else "cpu")
        model.to(fallback_device)
    model.eval()
    print(
        {
            "event": "model_loaded",
            "model": args.model_name,
            "device_map": getattr(model, "hf_device_map", None),
            "model_device": str(model.device),
        },
        flush=True,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        text_iter = iter_texts(Path(args.input_jsonl), args.text_field, args.max_docs)
        for doc_id, text in tqdm(text_iter, desc="building oracle", unit="doc"):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            seq_len = int(input_ids.shape[-1])
            if seq_len < args.block_size * 2:
                continue
            if args.log_every_doc:
                print({"event": "forward_start", "doc_id": doc_id, "seq_len": seq_len}, flush=True)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
            if args.log_every_doc:
                print({"event": "forward_done", "doc_id": doc_id, "seq_len": seq_len}, flush=True)

            hidden = outputs.hidden_states[-1][0].detach().cpu().float()
            attentions = torch.stack([attn[0].detach().cpu().float() for attn in outputs.attentions])
            if not torch.isfinite(hidden).all() or not torch.isfinite(attentions).all():
                message = {
                    "event": "nonfinite_forward_output",
                    "doc_id": doc_id,
                    "seq_len": seq_len,
                    "hidden_finite": bool(torch.isfinite(hidden).all().item()),
                    "attentions_finite": bool(torch.isfinite(attentions).all().item()),
                }
                if args.skip_nonfinite_docs:
                    print(message | {"action": "skip"}, flush=True)
                    continue
                raise FloatingPointError(
                    f"Model output contains NaN/Inf: {message}. "
                    "Regenerate with CUDA, a smaller max_length, or dtype bfloat16/float32."
                )
            # Shape: layers, heads, seq, seq. Average layers and heads for oracle labels.
            mean_attention = attentions.mean(dim=(0, 1))

            block_ranges = make_block_ranges(seq_len, args.block_size)
            num_blocks = len(block_ranges)
            global_hit_mass = torch.zeros(num_blocks)
            previous_oracle_blocks: list[int] = []
            previous_hidden = None

            for t in range(1, seq_len, args.sample_stride):
                current_block = t // args.block_size
                candidate_ranges = block_ranges[: current_block + 1]
                if len(candidate_ranges) <= 1:
                    continue

                token_attention = mean_attention[t, : t + 1]
                block_mass = aggregate_block_mass(token_attention, candidate_ranges)
                global_hit_mass[: block_mass.numel()] += block_mass

                high_hit_count = max(1, int(config.high_hit_top_fraction * num_blocks))
                high_hit_blocks = set(torch.topk(global_hit_mass, k=high_hit_count).indices.tolist())
                categories = block_categories(
                    num_blocks=block_mass.numel(),
                    prompt_blocks=num_blocks,
                    current_block=current_block,
                    high_hit_blocks=high_hit_blocks,
                    config=config,
                )

                budget_label, required_k = oracle_budget_label(
                    block_mass, config.coverage_threshold, config.budget_buckets
                )
                reuse_label = reuse_label_from_previous(
                    block_mass,
                    previous_oracle_blocks,
                    config.reuse_coverage_threshold,
                )

                cheap_scores = cosine_block_scores(hidden, candidate_ranges, t)
                score_features = score_summary_features(cheap_scores)
                if previous_hidden is None:
                    query_drift = 1.0
                else:
                    query_drift = float(
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            hidden[t].unsqueeze(0), previous_hidden.unsqueeze(0), dim=-1
                        ).item()
                    )

                row = {
                    "doc_id": doc_id,
                    "token_index": t,
                    "hidden_state": hidden[t].tolist(),
                    "query_drift": query_drift,
                    "score_features": score_features.tolist(),
                    "prev_feedback": [0.0] * 8,
                    "budget_label": budget_label,
                    "required_k": required_k,
                    "reuse_label": reuse_label,
                    "category_target": category_target(block_mass, categories).tolist(),
                }
                out.write(json.dumps(row, allow_nan=False) + "\n")

                previous_oracle_blocks = top_blocks(block_mass, required_k)
                previous_hidden = hidden[t]
            if args.log_every_doc:
                print({"event": "doc_done", "doc_id": doc_id, "seq_len": seq_len}, flush=True)


if __name__ == "__main__":
    main()
