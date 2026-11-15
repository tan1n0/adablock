"""Frozen-ranker-conditioned CE budget pilot. No teacher information in features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from models.block_ranker import load_ranker
from utils.ranking_batch import BatchRankingDataset, pack_scenes, forward_packed
from utils.ranking_data import assert_disjoint

BUDGETS = [0, 4, 8, 16, 32, 64, 128, 256, 'ALL']
RHO = .95
SEED = 20261015
QUANTILES = [.5, .7, .8, .9, .95, .975, .99, .995, .999, 1.]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def checkpoint_hash():
    return hashlib.sha256((ROOT / 'ranker/best.pt').read_bytes()).hexdigest()


def online_features(scene, scores):
    """Only quantities available before attending to selected values.

    Scores remain ranking logits: their softmax is a feature, never a mass label.
    Only completed middle blocks are candidates; sinks/local/self are always fixed.
    """
    q = scene['query'].float().flatten()
    s = scores.detach().float().cpu()
    n = len(s)
    centered = s - s.mean()
    quantiles = torch.quantile(centered, torch.linspace(0, 1, 17))
    ordered = centered.sort(descending=True, stable=True).values
    probability = s.softmax(0).sort(descending=True).values.cumsum(0)
    boundaries = [min(k, n) - 1 for k in BUDGETS[1:-1]]
    gaps = torch.stack([ordered[0] - ordered[i] for i in boundaries])
    shares = torch.stack([probability[i] for i in boundaries])
    scalars = torch.tensor([np.log1p(n), np.log1p(scene['token_position']),
                            float(s.mean()), float(s.std(unbiased=False)), float(s.max() - s.min())])
    identities = torch.cat([F.one_hot(torch.tensor(scene['layer_id']), 32),
                            F.one_hot(torch.tensor(scene['kv_head_id']), 8)]).float()
    return torch.cat([q, quantiles, gaps, shares, scalars, identities]).float()


def labels_for_order(mass, fixed, scores, budgets=BUDGETS, rho=RHO):
    order = scores.argsort(descending=True, stable=True)
    cumulative = mass[order].float().cumsum(0)
    n = len(mass)
    actual = torch.tensor([n if k == 'ALL' else min(k, n) for k in budgets], dtype=torch.long)
    coverage = torch.tensor([float(fixed) + (float(cumulative[k-1]) if k else 0.) for k in actual])
    hits = coverage >= rho - 1e-7
    assert bool(hits.any()), 'Full candidate set does not recover target mass'
    label = int(torch.where(hits)[0][0])
    exact = 0 if float(fixed) >= rho - 1e-7 else int(torch.where(cumulative + fixed >= rho - 1e-7)[0][0]) + 1
    return label, coverage, actual, exact


@torch.no_grad()
def generate_labels(split):
    torch.set_num_threads(4)
    dataset = BatchRankingDataset(ROOT / 'data' / split)
    for other in ('rank_train', 'budget_train', 'validation', 'test'):
        if other != split:
            assert_disjoint(dataset, BatchRankingDataset(ROOT / 'data' / other))
    ranker, payload = load_ranker(ROOT / 'ranker/best.pt', 'cuda:0')
    ranker.requires_grad_(False)
    features, labels, coverage, actual, identities, exact, oracle_exact = [], [], [], [], [], [], []
    started = time.monotonic()
    for begin in range(0, len(dataset), 64):
        batch = [dataset[i] for i in range(begin, min(begin + 64, len(dataset)))]
        packed = pack_scenes(batch, 'cuda:0')
        scores = forward_packed(ranker, packed).float().cpu()
        for j, scene in enumerate(batch):
            score = scores[packed['offsets'][j]:packed['offsets'][j+1]]
            assert torch.isfinite(score).all()
            label, cov, cost, needed = labels_for_order(scene['teacher_mass'], scene['fixed_mass'], score)
            features.append(online_features(scene, score))
            labels.append(label); coverage.append(cov); actual.append(cost); exact.append(needed)
            oracle = scene['teacher_mass'].sort(descending=True).values.cumsum(0) + scene['fixed_mass']
            oracle_exact.append(0 if scene['fixed_mass'] >= RHO - 1e-7 else int(torch.where(oracle >= RHO - 1e-7)[0][0]) + 1)
            identities.append([begin + j, scene['document_index'], scene['layer_id'], scene['kv_head_id'],
                               scene['token_position'], len(score), float(scene['fixed_mass'])])
        if begin % 4096 == 0:
            print(json.dumps({'stage': 'labels', 'split': split, 'done': min(begin + 64, len(dataset)),
                              'total': len(dataset), 'seconds': time.monotonic() - started}), flush=True)
    result = {'features': torch.stack(features), 'labels': torch.tensor(labels), 'coverage': torch.stack(coverage),
              'actual_blocks': torch.stack(actual), 'identities': torch.tensor(identities, dtype=torch.float64),
              'required_exact': torch.tensor(exact), 'oracle_sort_required_exact': torch.tensor(oracle_exact),
              'ranker_sha256': checkpoint_hash(), 'budgets': BUDGETS, 'rho': RHO,
              'documents': dataset.manifest['documents'], 'split': split,
              'identity_columns': ['scene', 'document', 'layer', 'kv_head', 'position', 'candidates', 'fixed_mass']}
    assert torch.isfinite(result['features']).all()
    assert torch.all(result['coverage'].diff(dim=1) >= -1e-7)
    assert torch.allclose(result['coverage'][:, -1], torch.ones(len(dataset)), atol=2e-4, rtol=0)
    out = ROOT / 'budget_data'
    out.mkdir(exist_ok=True)
    if (out / f'{split}.pt').exists():
        raise FileExistsError(split)
    torch.save(result, out / f'{split}.pt')
    dump(out / f'{split}_summary.json', {'scenes': len(dataset), 'features': result['features'].shape[1],
         'ranker_sha256': result['ranker_sha256'], 'label_histogram': torch.bincount(result['labels'], minlength=len(BUDGETS)).tolist(),
         'mean_required_exact': float(result['required_exact'].float().mean()),
         'mean_oracle_sort_required_exact': float(result['oracle_sort_required_exact'].float().mean()),
         'seconds': time.monotonic() - started})


class BudgetClassifier(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('std', torch.ones(features))
        self.layers = nn.Sequential(nn.Linear(features, 128), nn.GELU(), nn.Dropout(.1),
                                    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, len(BUDGETS)))

    def forward(self, x):
        return self.layers(((x - self.mean) / self.std).clamp(-8, 8))


def load_data(split):
    value = torch.load(ROOT / 'budget_data' / f'{split}.pt', weights_only=True)
    assert value['ranker_sha256'] == checkpoint_hash()
    return value


@torch.no_grad()
def probabilities(model, data):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(x.to(device)).softmax(-1).cpu() for x in data['features'].split(1024)])


def quantile_actions(probability, q):
    if q == 1.:
        return torch.full((len(probability),), len(BUDGETS)-1, dtype=torch.long)
    return (probability.cumsum(-1) < q).sum(-1).clamp_max(len(BUDGETS)-1)


def metrics(data, actions, mask=None):
    if mask is None:
        mask = torch.ones(len(actions), dtype=torch.bool)
    ids = torch.where(mask)[0]
    a = actions[ids]
    cov = data['coverage'][ids, a]
    cost = data['actual_blocks'][ids, a].float()
    positions = data['identities'][ids, 4].float()
    candidates = data['identities'][ids, 5].float()
    fixed_tokens = positions + 1 - candidates * 16
    return {'scenes': len(ids), 'mean_coverage': float(cov.mean()), 'hit_95': float((cov >= RHO - 1e-7).float().mean()),
            'mean_middle_blocks': float(cost.mean()), 'mean_attention_token_fraction': float(((fixed_tokens + cost * 16) / (positions + 1)).mean()),
            'mean_deficit': float((RHO - cov).clamp_min(0).mean()), 'p05_coverage': float(torch.quantile(cov, .05)),
            'full_class_rate': float((a == len(BUDGETS)-1).float().mean()),
            'all_candidates_read_rate': float((cost == candidates).float().mean()),
            'underbudget_rate': float((a < data['labels'][ids]).float().mean())}


def train():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    data, val = load_data('budget_train'), load_data('validation')
    model = BudgetClassifier(data['features'].shape[1]).cuda()
    model.mean.copy_(data['features'].mean(0))
    model.std.copy_(data['features'].std(0, unbiased=False).clamp_min(.001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x, y = data['features'].cuda(), data['labels'].cuda()
    vx, vy = val['features'].cuda(), val['labels'].cuda()
    out = ROOT / 'budget'
    out.mkdir()
    config = {'objective': 'unweighted cross entropy only', 'epochs': 40, 'batch_size': 512, 'lr': .001,
              'seed': SEED, 'feature_size': x.shape[1], 'parameters': sum(p.numel() for p in model.parameters()),
              'selection': 'minimum document-held-out validation cross entropy',
              'calibration': 'lowest validation mean block cost among fixed probability quantiles with >=95% scenes reaching 95% mass',
              'quantiles': QUANTILES, 'ranker_sha256': checkpoint_hash(), 'budget_classes': BUDGETS,
              'normalization': 'budget training feature statistics only, standard deviation floor .001, clip [-8,8]',
              'deployment_note': 'budget sees ranking scores after all candidate scores are computed; sparse attention read budget only'}
    dump(out / 'config.json', config)
    history, best = [], float('inf')
    for epoch in range(40):
        model.train()
        loss_sum = 0.
        for indices in torch.randperm(len(y), device='cuda').split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), y[indices])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(b) for b in vx.split(1024)])
            val_ce = float(F.cross_entropy(logits, vy))
            acc = float((logits.argmax(-1) == vy).float().mean())
        row = {'epoch': epoch + 1, 'train_ce': loss_sum / len(y), 'validation_ce': val_ce, 'validation_accuracy': acc}
        history.append(row)
        if val_ce < best:
            best = val_ce
            torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch + 1, 'validation_ce': val_ce}, out / 'best.pt')
        print(json.dumps(row), flush=True)
    torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': 40}, out / 'last.pt')
    dump(out / 'history.json', history)
    saved = torch.load(out / 'best.pt', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    p = probabilities(model, val)
    candidates = [{'quantile': q, **metrics(val, quantile_actions(p, q))} for q in QUANTILES]
    eligible = [r for r in candidates if r['hit_95'] >= .95]
    selected = min(eligible, key=lambda r: (r['mean_middle_blocks'], r['quantile']))
    dump(out / 'calibration.json', {'selected_quantile': selected['quantile'], 'validation_only': True,
                                   'validation_argmax': metrics(val, p.argmax(-1)), 'candidates': candidates})


def fixed_interpolation(methods, target_cost):
    points = sorted([(m['mean_middle_blocks'], name, m) for name, m in methods.items() if name.startswith('fixed_')])
    for left, right in zip(points, points[1:]):
        if left[0] <= target_cost <= right[0]:
            weight = (target_cost - left[0]) / max(right[0] - left[0], 1e-9)
            return {'lower': left[1], 'upper': right[1], 'upper_weight': weight,
                    'mean_middle_blocks': target_cost,
                    **{metric: (1-weight)*left[2][metric] + weight*right[2][metric]
                       for metric in ('mean_coverage', 'hit_95', 'mean_attention_token_fraction', 'mean_deficit')},
                    'note': 'Analytical linear interpolation at measured attention block count; not a separately run policy or latency result.'}
    return None


def evaluate():
    torch.set_num_threads(4)
    payload = torch.load(ROOT / 'budget/best.pt', weights_only=True)
    model = BudgetClassifier(payload['config']['feature_size']).cuda()
    model.load_state_dict(payload['state_dict'])
    q = json.loads((ROOT / 'budget/calibration.json').read_text())['selected_quantile']
    report = {'model': 'Meta-Llama-3.1-8B-Instruct', 'ranker_sha256': checkpoint_hash(),
              'selected_budget_epoch': payload['epoch'], 'calibrated_quantile': q, 'rho': RHO,
              'scope': 'teacher-forced sampled attention rows, not answer F1 or speed', 'splits': {}}
    for split in ('validation', 'test'):
        data = load_data(split)
        p = probabilities(model, data)
        methods = {f'fixed_{k}': torch.full_like(data['labels'], i) for i, k in enumerate(BUDGETS)}
        methods.update(adaptive_argmax=p.argmax(-1), adaptive_calibrated=quantile_actions(p, q),
                       oracle_budget_for_frozen_ranker=data['labels'])
        groups = {'overall': torch.ones(len(p), dtype=torch.bool), 'fixed_below_95': data['identities'][:, 6] < RHO - 1e-7}
        for layer in range(32):
            groups[f'layer_{layer}'] = data['identities'][:, 2] == layer
        for doc in range(len(data['documents'])):
            groups[f'document_{doc}'] = data['identities'][:, 1] == doc
        for lo, hi in [(0, 2048), (2048, 4096), (4096, 8192)]:
            groups[f'position_{lo}_{hi}'] = (data['identities'][:, 4] >= lo) & (data['identities'][:, 4] < hi)
        summary = {group: {name: metrics(data, action, mask) for name, action in methods.items()}
                   for group, mask in groups.items() if bool(mask.any())}
        overall = summary['overall']
        report['splits'][split] = {'groups': summary, 'matched_block_cost': {
            name: fixed_interpolation(overall, overall[name]['mean_middle_blocks'])
            for name in ('adaptive_argmax', 'adaptive_calibrated')},
            'label_histogram': torch.bincount(data['labels'], minlength=len(BUDGETS)).tolist(),
            'mean_oracle_sort_required_exact': float(data['oracle_sort_required_exact'].float().mean()),
            'mean_frozen_ranker_required_exact': float(data['required_exact'].float().mean())}
        torch.save({'probabilities': p, 'actions': methods, 'ranker_sha256': checkpoint_hash()}, ROOT / f'{split}_budget_predictions.pt')
        with (ROOT / f'{split}_comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['method', *next(iter(overall.values())).keys()])
            writer.writeheader()
            writer.writerows([dict(method=name, **m) for name, m in overall.items()])
        with (ROOT / f'{split}_per_scene.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(data['identity_columns'] + ['label', 'argmax_class', 'calibrated_class', 'argmax_coverage', 'calibrated_coverage', 'argmax_blocks', 'calibrated_blocks'])
            for i, identity in enumerate(data['identities'].tolist()):
                a, c = int(methods['adaptive_argmax'][i]), int(methods['adaptive_calibrated'][i])
                writer.writerow(identity + [int(data['labels'][i]), a, c, float(data['coverage'][i,a]), float(data['coverage'][i,c]),
                                            int(data['actual_blocks'][i,a]), int(data['actual_blocks'][i,c])])
        print(json.dumps({'split': split, 'overall': overall}), flush=True)
    dump(ROOT / 'results.json', report)


def self_test():
    # Actual rank order deliberately reverses the best block: its label must differ.
    mass, fixed = torch.tensor([.6, .1, .1]), torch.tensor(.2)
    wrong = labels_for_order(mass, fixed, torch.tensor([0., 2., 1.]), [0, 1, 2, 'ALL'], .85)
    ideal = labels_for_order(mass, fixed, mass, [0, 1, 2, 'ALL'], .85)
    assert wrong[0] == 3 and ideal[0] == 2
    assert labels_for_order(torch.tensor([.02]), torch.tensor(.98), torch.tensor([1.]))[0] == 0
    assert labels_for_order(torch.tensor([.5]), torch.tensor(.5), torch.tensor([1.]))[2].tolist() == [0,1,1,1,1,1,1,1,1]
    scene = {'query': torch.randn(4,128), 'layer_id': 31, 'kv_head_id': 7, 'token_position': 777}
    scores = torch.randn(10)
    before = online_features(scene, scores)
    scene.update(teacher_mass=torch.full((10,), float('nan')), fixed_mass=float('nan'))
    assert torch.equal(before, online_features(scene, scores))
    assert quantile_actions(torch.tensor([[.2,.3,.5]]), .8).item() == 2
    print('PASS: actual-ranking labels, fixed-only coverage, candidate clipping, teacher-free features, quantile selection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['labels', 'train', 'evaluate', 'self_test'])
    parser.add_argument('--split')
    args = parser.parse_args()
    if args.stage == 'labels':
        generate_labels(args.split)
    else:
        globals()[args.stage]()
