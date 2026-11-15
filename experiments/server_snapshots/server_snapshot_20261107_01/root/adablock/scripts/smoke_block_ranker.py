"""Small RANDOM Qwen integration test. This is not a pretrained-model quality benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_ranking_dataset import collect_document, ranking_metadata
from scripts.train_block_ranker import train
from models.block_ranker import load_ranker
from utils.ranking_data import RankingWriter, RankingDataset, document_identity
from utils.ranking_eval import evaluate_ranker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(12)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                        eos_token_id=2, bos_token_id=1, pad_token_id=0)
    config._attn_implementation = "sdpa"
    base = Qwen2ForCausalLM(config).eval()
    base_dir = output / "tiny_random_base"
    base.save_pretrained(base_dir)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"w{i}": i for i in range(4, 64)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(base_dir)
    meta = ranking_metadata(base, str(base_dir), [0, 1], 4, 4, 2, 2)
    for split, number in (("train", 3), ("validation", 2)):
        writer = RankingWriter(output / split, meta)
        rows = []
        for i in range(number):
            ids = torch.randint(4, 64, (1, 64 + 8*i))
            text = " ".join(f"w{j}" for j in ids[0].tolist())
            row = {"doc_id": f"{split}-{i}", "source": "synthetic_smoke", "text": text}
            rows.append(row)
            collect_document(base, ids, writer, document_identity(row, text), 4, 4)
        writer.finish()
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    settings = argparse.Namespace(train_dir=str(output/"train"), val_dir=str(output/"validation"),
        output_dir=str(output/"checkpoint"), epochs=8, batch_size=8, lr=1e-3, weight_decay=.01,
        projection_size=16, hidden_size=32, identity_size=4, block_chunk_size=32, no_position=False,
        coverage=.8, max_positives=16, hard_negatives=4, random_negatives=4, positive_pairs=32,
        pair_delta=1e-6, eval_budgets=[1, 2, 4, 8], max_train_scenes=None, max_val_scenes=None,
        seed=42, device="cpu", cpu_threads=2)
    history = train(settings)
    ranker, _ = load_ranker(output/"checkpoint/best.pt")
    result = evaluate_ranker(ranker, RankingDataset(output/"validation"), budgets=[1, 2, 4, 8], coverage=.8)
    result["warning"] = "Synthetic random Qwen only: validates plumbing, NOT downstream quality."
    result["training_first_last_loss"] = [history[0]["train_loss"], history[-1]["train_loss"]]
    (output/"evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "type": "synthetic integration smoke", "scenes": result["scenes"]}))


if __name__ == "__main__":
    main()
