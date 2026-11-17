"""Train COBS and K-only dual encoders on disjoint ranking documents.

Existing shards have GQA-mean teacher mass and rank-4 covariance only. We
normalize the historical target by (1-fixed_mass), then put fixed mass back
when measuring retained true mass. This is a prefill-only pilot; it cannot
substitute for the planned decode-query and value-aware data collection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.cacheable_selectors import (DualEncoder, DualEncoderConfig, SimpleCOBS,
                                         group_probabilities, rank4_diagonal_summary)
from utils.ranking_data import RankingDataset, assert_disjoint

DATA = ROOT / 'results/s_no_fixed_20261116/data'
OUTPUT = ROOT / 'results/value_aware_selector_20261116/pilot_training'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def scene_logits(model, scene):
    q = scene['query']
    if isinstance(model, SimpleCOBS):
        return model(q, scene['mean'], scene['basis'], scene['eigenvalues'],
                     scene['block_ids'], scene['counts'], scene['token_position'])
    summary = rank4_diagonal_summary(scene['mean'], scene['basis'], scene['eigenvalues'])
    return model(q, summary, scene['block_ids'], scene['counts'], scene['token_position'])


def scene_loss(model, scene):
    target_mass = scene['teacher_mass'].float()
    fixed = scene['fixed_mass'].float()
    target = target_mass / target_mass.sum().clamp_min(1e-8)
    predicted = group_probabilities(scene_logits(model, scene))
    loss = -(target * predicted.clamp_min(1e-9).log()).sum()
    # Fixed mass is a separate, unranked current block. The complete block
    # probabilities share precisely 1-fixed_mass of the full distribution.
    assert torch.allclose(target_mass.sum() + fixed, target_mass.new_tensor(1.), atol=2e-4)
    return loss


@torch.no_grad()
def evaluate(model, scenes):
    model.eval()
    rows = []
    for scene in scenes:
        mass = scene['teacher_mass'].float()
        fixed = float(scene['fixed_mass'])
        target = mass / mass.sum().clamp_min(1e-8)
        p = group_probabilities(scene_logits(model, scene))
        take = min(32, len(mass))
        selected = p.argsort(dim=-1, descending=True, stable=True)[:take]
        ideal = mass.argsort(dim=-1, descending=True, stable=True)[:take]
        rows.append({'coverage': fixed + float(mass[selected].sum()),
                     'ideal_coverage': fixed + float(mass[ideal].sum()),
                     'distribution_l1': float((p-target).abs().sum()),
                     'distribution_kl': float((target * (target.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum())})
    values = sorted(row['coverage'] for row in rows)
    return {'scenes': len(rows),
            'mean_coverage_at_32': sum(values)/len(values),
            'p10_coverage_at_32': values[max(0, int(.1*len(values))-1)],
            'mean_ideal_coverage_at_32': sum(r['ideal_coverage'] for r in rows)/len(rows),
            'mean_distribution_l1': sum(r['distribution_l1'] for r in rows)/len(rows),
            'mean_distribution_kl': sum(r['distribution_kl'] for r in rows)/len(rows)}


def sample_scenes(dataset, count, seed):
    if count <= 0 or count >= len(dataset):
        ids = list(range(len(dataset)))
    else:
        rng = random.Random(seed)
        ids = rng.sample(range(len(dataset)), count)
    return [dataset[i] for i in ids], ids


def run(args):
    torch.set_num_threads(args.cpu_threads)
    train_ds = RankingDataset(args.data/'train')
    valid_ds = RankingDataset(args.data/'validation')
    assert_disjoint(train_ds, valid_ds)
    meta = train_ds.metadata
    if (meta['block_size'], meta['sink_blocks'], meta['local_window_blocks'],
        meta['cobs_rank'], meta['gqa']) != (16, 0, 0, 4, 'mean_of_per_query_softmax_block_mass'):
        raise ValueError('Dataset protocol does not match this experiment')
    train, train_ids = sample_scenes(train_ds, args.train_scenes, args.seed)
    valid, valid_ids = sample_scenes(valid_ds, args.valid_scenes, args.seed+1)
    protocol = {'seed': args.seed, 'steps': args.steps, 'learning_rate': args.lr,
                'train_scenes': len(train), 'validation_scenes': len(valid),
                'train_indices': train_ids, 'validation_indices': valid_ids,
                'train_manifest_sha256': sha(args.data/'train/manifest.json'),
                'validation_manifest_sha256': sha(args.data/'validation/manifest.json'),
                'labels': 'GQA mean block attention mass; prefill queries, lengths 2048/4096/8192',
                'target_normalization': 'historical mass divided by 1-fixed_mass; fixed mass added for coverage',
                'variance': 'diagonal of rank-4 covariance approximation; same at train and inference',
                'selector_query': 'post-RoPE and already attention-scaled',
                'evaluation': 'independent validation documents; best mean coverage@32',
                'limitations': 'no decode queries, 16K/32K, V summaries or value-aware labels'}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Output protocol changed; use another directory')
    save_json(protocol_path, protocol)
    for name, model in [('cobs', SimpleCOBS()),
                        ('dual32', DualEncoder(DualEncoderConfig(embedding_dim=32))),
                        ('dual64', DualEncoder(DualEncoderConfig(embedding_dim=64)))]:
        result_path = args.output/f'{name}_result.json'
        if result_path.exists():
            print(name, 'skip complete', flush=True)
            continue
        torch.manual_seed(args.seed)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        baseline = evaluate(model, valid)
        best = None
        history = []
        for step in range(1, args.steps+1):
            scene = train[(step-1) % len(train)]
            optimizer.zero_grad(set_to_none=True)
            loss = scene_loss(model, scene)
            loss.backward()
            optimizer.step()
            if step == args.steps or step % args.eval_every == 0:
                metrics = evaluate(model, valid)
                history.append({'step': step, 'train_loss': float(loss), **metrics})
                if best is None or metrics['mean_coverage_at_32'] > best['mean_coverage_at_32']:
                    best = {'step': step, **metrics}
                    payload = {'model': name, 'config': asdict(model.config) if isinstance(model,DualEncoder) else {},
                               'state_dict': model.state_dict(), 'protocol': protocol}
                    target = args.output/f'{name}_best.pt'
                    torch.save(payload, target.with_suffix('.tmp'))
                    target.with_suffix('.tmp').replace(target)
            model.train()
        save_json(result_path, {'model': name, 'baseline': baseline, 'best': best,
                                'history': history, 'checkpoint_sha256': sha(args.output/f'{name}_best.pt')})
        print(name, best, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--train-scenes', type=int, default=384)
    parser.add_argument('--valid-scenes', type=int, default=96)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--eval-every', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--cpu-threads', type=int, default=4)
    run(parser.parse_args())
