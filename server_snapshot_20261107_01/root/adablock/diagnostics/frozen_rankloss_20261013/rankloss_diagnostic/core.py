"""Metrics use real teacher mass, never a softmax over selector scores."""
import math

import torch


METHODS = ('RankLoss', 'COBS', 'Quest', 'Oracle')
MASS_ATOL = 2e-4  # Existing RankingWriter tolerance, not middle renormalization.


def stable_order(scores, block_ids):
    scores = torch.as_tensor(scores).detach().double().cpu()
    ids = torch.as_tensor(block_ids).long().cpu()
    if scores.shape != ids.shape or scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned with candidate block_ids')
    if len(ids.unique()) != len(ids):
        raise ValueError('Duplicate candidate block IDs')
    canonical = torch.argsort(ids, stable=True)
    return canonical[torch.argsort(scores[canonical], descending=True, stable=True)]


def coverage_curve(mass, fixed, order):
    # Float64 accumulation preserves stored float32 teacher probabilities.
    return torch.cat((torch.tensor([float(fixed)], dtype=torch.float64),
                      float(fixed) + mass.double().cpu()[order].cumsum(0)))


def required_prefix(curve, target, budget_cap, tolerance=1e-7):
    if not 0 <= target <= 1 + MASS_ATOL or budget_cap < 0:
        raise ValueError('Invalid target or budget cap')
    hits = (curve >= target - tolerance).nonzero().flatten()
    required = int(hits[0]) if len(hits) else None
    return {'required_k': required, 'unreachable': required is None,
            'over_budget_cap': required is not None and required > budget_cap,
            'within_cap_k': required if required is not None and required <= budget_cap else None,
            'budget_cap': budget_cap, 'max_coverage': float(curve[-1])}


def audit_baseline_scores(sample):
    """Local smoke fallback: formulas documented in the existing source audit.

    Production uses utils.ranking_eval.baseline_scores. Q is already scaled.
    Each head's raw baseline score is averaged over the GQA group, matching
    the repository's legacy baseline convention (not teacher probabilities).
    """
    from recovered.block_ranker import cobs_terms
    first, second = cobs_terms(sample['query'], sample['mean'], sample['basis'], sample['eigenvalues'])
    cobs = (first + .5 * second + sample['counts'].float().log()).mean(0)
    q = sample['query'].float()
    quest = (torch.einsum('gd,bd->gb', q.clamp_min(0), sample['maximum'].float())
             + torch.einsum('gd,bd->gb', q.clamp_max(0), sample['minimum'].float())).mean(0)
    return {'COBS': cobs, 'Quest': quest}


def validate_scene(s, metadata):
    ids, mass = s['block_ids'], s['teacher_mass']
    n, t, bs = len(ids), int(s['token_position']), int(metadata['block_size'])
    if t < 0 or bs != int(s['block_size']):
        raise ValueError('Invalid position/block size')
    history = (t + bs - 1) // bs
    expected = list(range(min(int(metadata['sink_blocks']), history),
                          max(min(int(metadata['sink_blocks']), history),
                              history - int(metadata['local_window_blocks']))))
    if sorted(ids.tolist()) != expected or not bool(((ids + 1) * bs <= t).all()):
        raise ValueError('Candidates differ from configured sink/local/causal middle partition')
    d = int(metadata['head_dim'])
    rank = int(metadata['cobs_rank'])
    shapes = {'query': (7, d), 'mean': (n, d), 'basis': (n, rank, d),
              'eigenvalues': (n, rank), 'minimum': (n, d), 'maximum': (n, d),
              'counts': (n,), 'teacher_mass': (n,), 'block_ids': (n,)}
    for key, shape in shapes.items():
        if key not in s:
            raise ValueError(f'Missing offline field: {key}')
        if tuple(s[key].shape) != shape or not bool(torch.isfinite(s[key]).all()):
            raise ValueError(f'Invalid offline field {key}: expected {shape}')
    if ids.dtype != torch.long or not bool((s['counts'] == bs).all()):
        raise ValueError('Middle candidates must be complete blocks with int64 IDs')
    if not bool((s['minimum'] <= s['maximum']).all()) or not bool((s['eigenvalues'] >= 0).all()):
        raise ValueError('Invalid Quest extrema / covariance eigenvalues')
    fixed = float(s['fixed_mass'])
    if not math.isfinite(fixed) or fixed < 0 or bool((mass < 0).any()):
        raise ValueError('Invalid teacher mass')
    if abs(fixed + float(mass.double().sum()) - 1) > MASS_ATOL:
        raise ValueError('fixed + middle teacher mass must equal 1; do not renormalize middle')
    if int(s['layer_id']) not in metadata['layers'] or not 0 <= int(s['kv_head_id']) < 4:
        raise ValueError('Invalid scene layer/KV head')


def validate_attention(sample, payload):
    """Only a real per-head probability row and V can reconstruct AV."""
    if payload.get('format') != 'adablock_attention_values_v1':
        raise ValueError('Output error requires adablock_attention_values_v1 sidecar')
    a, v = payload['attention'].double(), payload['values'].double()
    t = int(sample['token_position'])
    if tuple(a.shape) != (7, t + 1) or tuple(v.shape) not in ((t + 1, 128), (7, t + 1, 128)):
        raise ValueError('Sidecar needs attention [7,t+1] and V [t+1,128] or [7,t+1,128], including self')
    if not torch.isfinite(a).all() or not torch.isfinite(v).all() or bool((a < 0).any()):
        raise ValueError('Nonfinite/negative attention or V')
    if not torch.allclose(a.sum(-1), torch.ones(7, dtype=a.dtype), atol=MASS_ATOL, rtol=0):
        raise ValueError('Each query head must independently causal-softmax including self')
    bs = int(sample['block_size'])
    masses = torch.tensor([a[:, int(b)*bs:(int(b)+1)*bs].sum(-1).mean() for b in sample['block_ids']], dtype=a.dtype)
    if not torch.allclose(masses, sample['teacher_mass'].double(), atol=MASS_ATOL, rtol=0):
        raise ValueError('Sidecar attention disagrees with teacher block mass')
    if abs(1 - float(masses.sum()) - float(sample['fixed_mass'])) > MASS_ATOL:
        raise ValueError('Sidecar fixed mass disagrees with teacher')
    return a, v


def attention_output_error(sample, validated, selected_ids):
    a, v = validated
    keep = torch.ones(a.shape[1], dtype=torch.bool)
    bs = int(sample['block_size'])
    chosen = set(map(int, selected_ids))
    for b in sample['block_ids'].tolist():
        if b not in chosen:
            keep[b * bs:(b + 1) * bs] = False
    sparse = a * keep
    denom = sparse.sum(-1, keepdim=True)
    if bool((denom <= 0).any()):
        raise ValueError('Selected tokens have zero probability for a query head; sparse AV undefined')
    sparse = sparse / denom  # Per-head selected attention renormalization, not teacher-middle mass.
    multiply = (lambda weights: weights @ v) if v.ndim == 2 else (lambda weights: torch.einsum('gt,gtd->gd', weights, v))
    full, approx = multiply(a), multiply(sparse)
    norm = float(torch.linalg.vector_norm(full))
    error = float(torch.linalg.vector_norm(approx - full))
    return {'output_error_status': 'available', 'output_abs_l2': error,
            'output_rmse': float((approx - full).square().mean().sqrt()),
            'output_relative_l2': error / norm if norm > 0 else None,
            'full_output_l2': norm}


class Means:
    """Scene mean and first-within-document-then-across-document mean."""
    def __init__(self):
        self.groups = {}

    def add(self, key, doc, row):
        group = self.groups.setdefault(tuple(key), {})
        state = group.setdefault(int(doc), {'scenes': 0, 'metrics': {}})
        state['scenes'] += 1
        for metric, value in row.items():
            if isinstance(value, (int, float, bool)) and value is not None:
                acc = state['metrics'].setdefault(metric, [0., 0])
                acc[0] += float(value)
                acc[1] += 1

    def rows(self, key_names):
        for key, docs in sorted(self.groups.items(), key=lambda item: str(item[0])):
            common = dict(zip(key_names, key))
            total_scenes = sum(d['scenes'] for d in docs.values())
            metrics = sorted({m for d in docs.values() for m in d['metrics']})
            for weighting in ('scene', 'document_equal'):
                row = dict(common, weighting=weighting, scenes=total_scenes, documents=len(docs))
                for m in metrics:
                    available = [d['metrics'][m] for d in docs.values() if m in d['metrics']]
                    row[m] = (sum(v[0] for v in available) / sum(v[1] for v in available)
                              if weighting == 'scene' else sum(v[0]/v[1] for v in available)/len(available))
                    row[m + '_available_scenes'] = sum(v[1] for v in available)
                    row[m + '_available_documents'] = len(available)
                yield row
