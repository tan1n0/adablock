"""Isolated Llama pilot: document split, native-attention check, and extraction."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
MODEL = '/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb'
SEED = 20261015


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def prepare():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    pools = {
        'train': '/root/adablock/data/train_strict_packed_tokens.jsonl',
        'validation': '/root/adablock/results/ranker_audit_20261003_v1/validation_selected.jsonl',
        'test': '/root/adablock/results/ranker_audit_20261003_v1/test_reserved.jsonl',
    }
    rows = {k: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
            for k, p in pools.items()}
    rng = random.Random(SEED)
    for values in rows.values():
        rng.shuffle(values)
    used_components, used_hashes = set(), set()
    chosen = {}
    for split, pool, per_length in [('rank_train', 'train', 4), ('budget_train', 'train', 2),
                                    ('validation', 'validation', 2), ('test', 'test', 2)]:
        selected = []
        for length in (8192, 4096, 2048):
            target_source = f'token_pack_{length * 2}'
            candidates = sorted(rows[pool], key=lambda r: r.get('source') != target_source)
            for row in candidates:
                components = set(row.get('component_doc_ids') or [f"{row['source']}::{row['doc_id']}"])
                digest = hashlib.sha256(row['text'].encode()).hexdigest()
                if components & used_components or digest in used_hashes:
                    continue
                ids = tokenizer(row['text'], add_special_tokens=True)['input_ids']
                if len(ids) < length:
                    continue
                selected.append(dict(row, pilot_split=split, pilot_max_tokens=length,
                                     pilot_input_ids=ids[:length], original_llama_tokens=len(ids)))
                used_components.update(components)
                used_hashes.add(digest)
                if sum(r['pilot_max_tokens'] == length for r in selected) == per_length:
                    break
            else:
                raise RuntimeError(f'Insufficient disjoint data for {split}, {length}')
        # Short first, so memory/performance checks precede long examples.
        selected.sort(key=lambda r: r['pilot_max_tokens'])
        path = ROOT / f'{split}.jsonl'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        chosen[split] = [{k: v for k, v in r.items() if k not in ('text', 'pilot_input_ids')}
                         for r in selected]
    isolation = []
    for a, b in itertools.combinations(chosen, 2):
        ac = {x for r in chosen[a] for x in r.get('component_doc_ids', [r['doc_id']])}
        bc = {x for r in chosen[b] for x in r.get('component_doc_ids', [r['doc_id']])}
        assert not ac & bc
        isolation.append({'splits': [a, b], 'component_overlap': len(ac & bc)})
    import torch, transformers
    write_json(ROOT / 'protocol.json', {
        'seed': SEED, 'model': MODEL, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'split_documents': {k: len(v) for k, v in chosen.items()}, 'selection': chosen,
        'source_pools': pools, 'isolation': isolation, 'lengths': [2048, 4096, 8192],
        'layers': list(range(32)), 'kv_heads': 8, 'block_size': 16, 'sink_tokens': 32,
        'local_window_blocks': 32, 'cobs_rank': 4, 'sample_stride': 256, 'coverage_target': .95,
        'budget_classes': [0, 4, 8, 16, 32, 64, 128, 256, 'ALL'],
        'ranker_epochs': 3, 'ranker_batch_size': 64, 'ranker_lr': .0002,
        'budget_input': 'current query, layer/head identity, candidate count and frozen ranker score summaries; no teacher attention features',
        'evaluation_scope': 'held-out teacher-forced attention scenes; not LongBench answer F1 or measured decoding speed',
        'tokenization': 'entire raw training text tokenized once with Llama tokenizer, then token truncation',
        'preservation': 'new isolated directory; old datasets, weights, evaluations unchanged',
    })
    print(json.dumps({'stage': 'prepared', 'counts': {k: len(v) for k, v in chosen.items()}}), flush=True)


def check_native(model, output, tiny=False):
    import torch
    from scripts.build_oracle_dataset import SampledAttentionCollector
    from scripts.build_ranking_dataset import ranking_metadata
    from utils.ranking_data import RankingWriter
    torch.manual_seed(SEED)
    seq = 128 if tiny else 768
    positions = [63, 111] if tiny else [575, 700]
    layer_ids = list(range(model.config.num_hidden_layers)) if tiny else [0, 15, 31]
    bs, sink, local = (8, 8, 2) if tiny else (16, 32, 32)
    meta = ranking_metadata(model, MODEL, layer_ids, bs, sink, local, 4)
    writer = RankingWriter(output / 'native_check_shards', meta)
    writer.begin_document({'source': 'native_check', 'doc_id': 'synthetic', 'components': ['native_check'], 'text_sha256': 'synthetic'})
    collector = SampledAttentionCollector(model.layers, layer_ids, positions, seq, bs, 2,
        sink_blocks=meta['sink_blocks'], local_window_blocks=local, cobs_rank=4, ranking_writer=writer)
    native = {}
    hooks = []
    def capture(layer):
        def hook(module, args, output):
            weights = output[1][0, :, positions, :].float()
            native[layer] = weights.view(model.config.num_key_value_heads, -1, len(positions), seq).mean(1).cpu()
        return hook
    for layer in layer_ids:
        hooks.append(model.layers[layer].self_attn.register_forward_hook(capture(layer)))
    old_impl = model.config._attn_implementation
    model.config._attn_implementation = 'eager'
    try:
        ids = torch.randint(10, min(1000, model.config.vocab_size), (1, seq), device=next(model.parameters()).device)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        collector.finish()
    finally:
        collector.remove()
        for hook in hooks:
            hook.remove()
        model.config._attn_implementation = old_impl
    manifest = writer.finish()
    errors = []
    for shard in manifest['shards']:
        saved = torch.load(writer.directory / shard['file'], weights_only=True)
        for record in saved['records']:
            index = positions.index(record['token_position'])
            block = native[shard['layer_id']][:, index].view(model.config.num_key_value_heads, -1, bs).sum(-1)
            reference = block[:, record['block_ids']]
            errors.append(float((reference - record['teacher_mass']).abs().max()))
            reference_fixed = native[shard['layer_id']][:, index].sum(-1) - reference.sum(-1)
            errors.append(float((reference_fixed - record['fixed_mass']).abs().max()))
    tolerance = 2e-6 if tiny else .003
    assert max(errors) < tolerance, (max(errors), tolerance)
    result = {'max_abs_mass_error': max(errors), 'tolerance': tolerance, 'dtype': str(next(model.parameters()).dtype),
              'layers': layer_ids, 'positions': positions, 'native_backend': 'eager',
              'note': 'Native BF16 probabilities are rounded; collector preserves FP32 softmax probabilities.'}
    write_json(output / 'native_check.json', result)
    print(json.dumps({'stage': 'native_check_passed', **result}), flush=True)


def collect(split):
    import torch
    from transformers import AutoModel
    from scripts.build_ranking_dataset import collect_document, ranking_metadata
    from utils.ranking_data import RankingWriter, document_identity
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                                      attn_implementation='sdpa', local_files_only=True).eval()
    assert model.config.model_type == 'llama'
    if split == 'rank_train':
        check_native(model, ROOT)
    metadata = ranking_metadata(model, MODEL, list(range(32)), 16, 32, 32, 4)
    metadata.update(sample_stride=256, attention_backend='sdpa', base_dtype='bfloat16')
    writer = RankingWriter(ROOT / 'data' / split, metadata)
    rows = [json.loads(line) for line in (ROOT / f'{split}.jsonl').read_text().splitlines()]
    start = time.monotonic()
    for i, row in enumerate(rows):
        ids = torch.tensor([row['pilot_input_ids']], dtype=torch.long, device='cuda:0')
        collect_document(model, ids, writer, document_identity(row, row['text']), sample_stride=256, query_chunk_size=8)
        print(json.dumps({'stage': 'extract', 'split': split, 'done': i + 1, 'total': len(rows),
                          'tokens': ids.shape[-1], 'seconds': time.monotonic() - start,
                          'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
    manifest = writer.finish()
    write_json(ROOT / f'extract_{split}_complete.json', {'scenes': sum(s['scenes'] for s in manifest['shards']),
                                                        'seconds': time.monotonic() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'collect', 'tiny_check'])
    parser.add_argument('--split')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'collect':
        collect(args.split)
    else:
        import torch
        from transformers import LlamaConfig, LlamaModel
        torch.set_num_threads(2)
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                          rope_theta=500000., attention_dropout=0.)
        cfg._attn_implementation = 'eager'
        folder = ROOT / 'tiny_check'
        folder.mkdir()
        check_native(LlamaModel(cfg).eval(), folder, tiny=True)
