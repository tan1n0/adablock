"""Extract current-query COBS ranking shards using the existing sampled Oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_oracle_dataset import SampledAttentionCollector, anchor_offset, load_dtype, normalize_device_map
from utils.ranking_data import RankingWriter, document_identity, model_signature


def ranking_metadata(model, model_name, layers, block_size, sink_tokens, local_window_blocks, cobs_rank):
    config = model.config
    return {"model_name": model_name, "model_signature": model_signature(config),
            "head_dim": config.hidden_size // config.num_attention_heads,
            "num_layers": config.num_hidden_layers, "num_kv_heads": config.num_key_value_heads,
            "group_size": config.num_attention_heads // config.num_key_value_heads,
            "block_size": block_size, "sink_blocks": (sink_tokens + block_size - 1) // block_size,
            "local_window_blocks": local_window_blocks, "cobs_rank": cobs_rank,
            "layers": sorted(layers), "query_scaling": "attention_scale_applied_once",
            "rope": "post_rope", "gqa": "mean_of_per_query_softmax_block_mass",
            "feature_storage_dtype": "float32"}


@torch.no_grad()
def collect_document(model, input_ids, writer, identity, sample_stride=64, query_chunk_size=8):
    transformer = getattr(model, "model", model)
    seq_len = input_ids.shape[-1]
    offset = anchor_offset(identity["source"], identity["doc_id"], sample_stride, "document_hash", 0)
    positions = [t for t in range(offset, seq_len, sample_stride) if t > 0]
    writer.begin_document(identity)
    if not positions:
        return
    meta = writer.metadata
    collector = SampledAttentionCollector(
        transformer.layers, meta["layers"], positions, seq_len, meta["block_size"], query_chunk_size,
        sink_blocks=meta["sink_blocks"], local_window_blocks=meta["local_window_blocks"],
        cobs_rank=meta["cobs_rank"], ranking_writer=writer,
    )
    try:
        transformer(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        collector.finish()
    finally:
        collector.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--text-field", default="text")
    p.add_argument("--layers", nargs="+", type=int, default=[12, 20, 27])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--sink-tokens", type=int, default=32)
    p.add_argument("--local-window-blocks", type=int, default=32)
    p.add_argument("--cobs-rank", type=int, default=4)
    p.add_argument("--sample-stride", type=int, default=64)
    p.add_argument("--query-chunk-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-docs", type=int)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args()
    if min(args.block_size, args.local_window_blocks, args.sample_stride, args.query_chunk_size, args.max_length) < 1:
        p.error("Sizes/stride must be positive; v1 requires local-window-blocks >= 1.")
    if args.sink_tokens < 0 or args.cobs_rank < 0 or len(set(args.layers)) != len(args.layers):
        p.error("Invalid sink/rank or duplicate layers.")
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("CUDA unavailable; use --allow-cpu explicitly for a small CPU probe.")
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error("Use a new output directory; existing ranking data will not be overwritten.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    # Extraction needs only the backbone, not a generation head or remote
    # generate extension. This also saves the LM-head allocation on CPU probes.
    model = AutoModel.from_pretrained(
        args.model_name, torch_dtype=load_dtype(args.dtype), device_map=normalize_device_map(args.device_map),
        attn_implementation="sdpa", local_files_only=args.local_files_only,
    ).eval()
    if getattr(model, "hf_device_map", None) is None:
        model.to(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if model.config.model_type != "qwen2":
        raise ValueError("The shared sampled collector currently supports Qwen2/Qwen2.5 only.")
    metadata = ranking_metadata(model, args.model_name, args.layers, args.block_size, args.sink_tokens,
                                args.local_window_blocks, args.cobs_rank)
    metadata.update(sample_stride=args.sample_stride, attention_backend="sdpa", base_dtype=args.dtype)
    writer = RankingWriter(output, metadata)
    with open(args.input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_docs is not None and i >= args.max_docs:
                break
            row = json.loads(line)
            text = row[args.text_field]
            identity = document_identity(row, text)
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"]
            collect_document(model, ids.to(model.device), writer, identity, args.sample_stride, args.query_chunk_size)
            print(json.dumps({"document": i, "tokens": ids.numel(), "shards": len(writer.entries)}), flush=True)
    manifest = writer.finish()
    print(json.dumps({"output": str(output), "scenes": sum(e["scenes"] for e in manifest["shards"])}))


if __name__ == "__main__":
    main()
