"""CPU-ready frozen RankLoss vs repository COBS/Quest and block-mass Oracle."""
import argparse
import csv
import json
import platform
import shlex
import sys
import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import torch

from .core import (METHODS, Means, attention_output_error, audit_baseline_scores,
                   coverage_curve, required_prefix, stable_order, validate_attention, validate_scene)
from .data import LocalAuditDataset, check_independent_test, read_json, scene_key, sha256
from .prepare import resolve_config


FROZEN_HASH = 'd76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae'
FEATURE_KEYS = ('query', 'mean', 'basis', 'eigenvalues', 'counts', 'block_ids',
                'token_position', 'block_size', 'layer_id', 'kv_head_id')


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def repository_baselines(raw):
    result = {}
    for method, aliases in [('COBS', ('cobs_legacy_mean_logits', 'cobs')),
                            ('Quest', ('quest_aabb', 'quest'))]:
        present = [key for key in aliases if key in raw]
        if len(present) != 1:
            raise ValueError(f'Expected one actual repository {method} baseline; got keys {list(raw)}')
        result[method] = raw[present[0]]
    return result


def load_runtime(config, local_audit, smoke, device):
    receipt = read_json(config['freeze_receipt'])
    checkpoint_hash = sha256(config['checkpoint'])
    if checkpoint_hash != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch; refuse to select or train another checkpoint')
    if not smoke:
        if receipt['checkpoint_sha256'] != FROZEN_HASH or receipt['step'] != 50000:
            raise ValueError('Formal diagnosis requires the audited frozen all28 50k checkpoint')
        if read_json(config['training_config']) != receipt['source_config']:
            raise ValueError('Actual training configuration differs from the audited frozen experiment')
        command = shlex.split(receipt['source_config']['production_command'])
        out = command[command.index('--out') + 1]
        if receipt['checkpoint_path'] != str(Path(out) / 'best.pt').replace('\\', '/'):
            raise ValueError('Checkpoint path does not match the actual training configuration')
    source_hashes = {}
    if local_audit:
        if not smoke:
            raise ValueError('Local audit formula fallback is limited to nonformal smoke')
        from recovered.block_ranker import load_ranker
        dataset = LocalAuditDataset(config['data'])
        baseline = audit_baseline_scores
        source_hashes['models/block_ranker.py'] = sha256(Path(__file__).resolve().parents[1] / 'recovered/block_ranker.py')
    else:
        root = Path(config['repo_root'])
        sys.path.insert(0, str(root))
        from models.block_ranker import load_ranker
        from utils.ranking_data import RankingDataset
        from utils.ranking_eval import baseline_scores
        dataset = RankingDataset(config['data'])
        baseline = lambda sample: repository_baselines(baseline_scores(sample))
        for filename in ('models/block_ranker.py', 'utils/ranking_data.py', 'utils/ranking_eval.py'):
            source_hashes[filename] = sha256(root / filename)
            expected = receipt['source_config']['source_sha256'][filename]
            if source_hashes[filename] != expected:
                raise ValueError(f'Repository source differs from frozen experiment: {filename}')
    model, payload = load_ranker(config['checkpoint'], device=device)
    model.float().eval().requires_grad_(False)
    if not smoke and (payload.get('training', {}).get('step') != 50000
                      or model.config.head_dim != 128 or model.config.num_layers != 28
                      or model.config.num_kv_heads != 4
                      or sum(p.numel() for p in model.parameters()) != 61313):
        raise ValueError('Checkpoint structure or training provenance does not match all28 50k')
    keys = ('schema', 'model_name', 'model_signature', 'block_size', 'sink_blocks',
            'local_window_blocks', 'cobs_rank', 'layers', 'query_scaling', 'rope', 'gqa')
    if any(payload['data_metadata'].get(k) != dataset.metadata.get(k) for k in keys):
        raise ValueError('Checkpoint/data metadata mismatch (old3layer test is not all28 test)')
    if (dataset.metadata['block_size'] != 16 or dataset.metadata['head_dim'] != 128
            or dataset.metadata['cobs_rank'] != 4 or dataset.metadata['local_window_blocks'] < 1):
        raise ValueError('Expected block16, D128, covariance rank4, fixed unfinished local block')
    return dataset, model, baseline, receipt, source_hashes, payload


def evaluate(config, output, device='cpu', smoke=False, local_audit=False, max_scenes=None,
             budgets=(4, 8, 16, 32, 64), targets=(.9, .95, .99), budget_cap=64,
             require_output_error=False, save_rankings=False, plot=False):
    if not budgets or min(budgets) < 0 or budget_cap < 0 or any(not 0 < t <= 1 for t in targets):
        raise ValueError('Invalid budget, cap, or target')
    budgets = sorted(set(budgets))
    if max_scenes is not None and max_scenes < 1:
        raise ValueError('max_scenes must be positive')
    if smoke:
        max_scenes = 16 if max_scenes is None else max_scenes
        if max_scenes > 128:
            raise ValueError('Smoke is limited to 128 CPU/offline scenes')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    start = time.perf_counter()
    dataset, model, baseline, receipt, source_hashes, checkpoint_payload = load_runtime(config, local_audit, smoke, device)
    manifest_path = Path(config['data']) / 'manifest.json'
    data_hash = sha256(manifest_path)
    split_hashes = ({'test': data_hash} if smoke else check_independent_test(config, dataset.manifest, receipt))
    attention_manifest = read_json(config['attention_manifest']) if config.get('attention_manifest') else None
    if attention_manifest:
        if (attention_manifest.get('format') != 'adablock_attention_index_v1'
                or attention_manifest.get('ranking_manifest_sha256') != data_hash):
            raise ValueError('Attention sidecar index format/data manifest mismatch')
    if require_output_error and attention_manifest is None:
        raise ValueError('Output error unavailable: original ranking v1 has no per-head attention or V')
    indices = range(len(dataset))
    if max_scenes is not None and max_scenes < len(dataset):
        # Deterministic spread, recorded exactly. No split generation/repartition.
        indices = torch.linspace(0, len(dataset)-1, max_scenes).long().unique().tolist()
    if not indices:
        raise ValueError('No dataset scenes')
    output.mkdir(parents=True)
    json_write(output / 'RUNNING.json', {'config': config, 'smoke': smoke, 'indices': list(indices) if max_scenes is not None else 'all'})
    coverage_means, quality_means = Means(), Means()
    documents, layers, heads, output_status = Counter(), Counter(), Counter(), Counter()
    seen_keys, shard_hashes, attention_hashes = set(), {}, {}
    skipped = 0
    scene_fields = ['scene_index', 'scene_key', 'document_index', 'layer_id', 'token_position',
                    'kv_head_id', 'method', 'budget', 'actual_k', 'candidate_count', 'fixed_mass',
                    'middle_mass', 'coverage', 'oracle_gap', 'selected_block_ids',
                    'output_error_status', 'output_abs_l2', 'output_rmse', 'output_relative_l2', 'full_output_l2']
    quality_fields = ['scene_index', 'scene_key', 'document_index', 'method', 'target_kind',
                      'reference_budget', 'target_mass', 'required_k', 'unreachable',
                      'over_budget_cap', 'within_cap_k', 'budget_cap', 'max_coverage']
    with (output / 'scenes.csv').open('w', newline='', encoding='utf-8') as scenes_file, \
         (output / 'quality_scenes.csv').open('w', newline='', encoding='utf-8') as quality_file, \
         (output / 'scenes.jsonl').open('w', encoding='utf-8') as diagnostic_file, torch.inference_mode():
        sw, qw = csv.DictWriter(scenes_file, scene_fields), csv.DictWriter(quality_file, quality_fields)
        sw.writeheader(); qw.writeheader()
        for index in indices:
            shard = dataset.shards[bisect_right(dataset.ends, index)]
            if shard['file'] not in shard_hashes:
                shard_hashes[shard['file']] = sha256(Path(config['data']) / shard['file'])
            sample = dataset[index]
            validate_scene(sample, dataset.metadata)
            key = scene_key(sample)
            if key in seen_keys:
                raise ValueError(f'Duplicate scene: {key}')
            seen_keys.add(key)
            if not len(sample['block_ids']):
                skipped += 1
                continue
            doc = int(sample['document_index'])
            documents[doc] += 1; layers[int(sample['layer_id'])] += 1; heads[int(sample['kv_head_id'])] += 1
            features = {k: sample[k].to(device) if isinstance(sample[k], torch.Tensor) else sample[k]
                        for k in FEATURE_KEYS}
            scores = baseline(sample)
            scores.update(RankLoss=model(features).detach().float().cpu(), Oracle=sample['teacher_mass'])
            orders = {method: stable_order(scores[method], sample['block_ids']) for method in METHODS}
            curves = {method: coverage_curve(sample['teacher_mass'], sample['fixed_mass'], order)
                      for method, order in orders.items()}
            sidecar = None
            status = 'unavailable_no_attention_and_V'
            if attention_manifest and key in attention_manifest['scenes']:
                entry = attention_manifest['scenes'][key]
                path = Path(config['attention_manifest']).parent / entry['file']
                digest = sha256(path)
                if digest != entry['sha256']:
                    raise ValueError(f'Attention sidecar hash mismatch: {key}')
                attention_hashes[key] = digest
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if payload.get('scene_key') != key:
                    raise ValueError('Attention sidecar scene identity mismatch')
                sidecar = validate_attention(sample, payload)
                status = 'available'
            elif require_output_error:
                raise ValueError(f'Output error unavailable for scene {key}: missing attention/V sidecar')
            output_status[status] += 1
            identity = {k: int(sample[k]) for k in ('document_index', 'layer_id', 'token_position', 'kv_head_id')}
            rows, quality_rows = [], []
            for method in METHODS:
                for budget in budgets:
                    take = min(budget, len(sample['block_ids']))
                    selected = sample['block_ids'][orders[method][:take]].tolist()
                    cov = float(curves[method][take])
                    error = attention_output_error(sample, sidecar, selected) if sidecar else {'output_error_status': status}
                    row = dict(identity, scene_index=index, scene_key=key, method=method,
                               budget=budget, actual_k=take, candidate_count=len(sample['block_ids']),
                               fixed_mass=float(sample['fixed_mass']), middle_mass=cov-float(sample['fixed_mass']),
                               coverage=cov, oracle_gap=max(0., float(curves['Oracle'][take])-cov),
                               selected_block_ids=json.dumps(selected), **error)
                    sw.writerow(row); rows.append(row)
                    values = {k: row[k] for k in ('actual_k', 'candidate_count', 'fixed_mass', 'middle_mass', 'coverage', 'oracle_gap')}
                    values.update({k: v for k, v in error.items() if k != 'output_error_status'})
                    coverage_means.add((method, budget), doc, values)
                requests = [('absolute', '', float(target)) for target in targets]
                requests += [('RankLoss_coverage', budget, float(curves['RankLoss'][min(budget, len(sample['block_ids']))]))
                             for budget in budgets]
                for kind, reference_budget, target in requests:
                    prefix = required_prefix(curves[method], target, budget_cap)
                    row = dict(scene_index=index, scene_key=key, document_index=doc, method=method,
                               target_kind=kind, reference_budget=reference_budget, target_mass=target, **prefix)
                    qw.writerow(row); quality_rows.append(row)
                    quality_means.add((method, kind, reference_budget, target if kind == 'absolute' else ''), doc,
                                      dict(target_mass=target, **prefix))
            diagnostic = dict(scene_index=index, scene_key=key, **identity, coverage=rows, quality=quality_rows)
            if save_rankings:
                diagnostic['ranked_block_ids'] = {m: sample['block_ids'][o].tolist() for m, o in orders.items()}
            diagnostic_file.write(json.dumps(diagnostic, allow_nan=False) + '\n')
    if not documents:
        raise ValueError('No scenes with middle candidates')
    if sha256(config['checkpoint']) != receipt['checkpoint_sha256'] or sha256(manifest_path) != data_hash:
        raise ValueError('Input checkpoint or manifest changed during evaluation')
    coverage_rows = list(coverage_means.rows(('method', 'budget')))
    quality_rows = list(quality_means.rows(('method', 'target_kind', 'reference_budget', 'absolute_target')))
    csv_write(output / 'coverage.csv', coverage_rows)
    csv_write(output / 'quality.csv', quality_rows)
    summary = {
        'status': 'complete', 'evaluation_label': 'NONFORMAL_SMOKE' if smoke else 'INDEPENDENT_TEST',
        'partial_dataset': len(indices) < len(dataset), 'device': str(device), 'dtype': 'float32 ranker; float64 mass accumulation',
        'evaluation_seconds': time.perf_counter()-start, 'config': config, 'metadata': dataset.metadata, 'budgets': budgets,
        'targets': list(targets), 'budget_cap': budget_cap, 'dataset_scenes': len(dataset),
        'selected_scene_indices': list(indices) if max_scenes is not None else 'all', 'scenes': sum(documents.values()), 'skipped_no_middle': skipped,
        'document_scene_counts': dict(documents), 'layer_scene_counts': dict(layers), 'kv_head_scene_counts': dict(heads),
        'documents': len(documents), 'document_unit': 'packed document/sequence from manifest; scenes are not independent documents',
        'checkpoint': {'path': config['checkpoint'], 'sha256': receipt['checkpoint_sha256'],
                       'original_path': receipt['checkpoint_path'], 'training': checkpoint_payload['training'],
                       'model_config': model.config.__dict__,
                       'parameters': sum(p.numel() for p in model.parameters()), 'frozen': all(not p.requires_grad for p in model.parameters())},
        'split_manifest_sha256': split_hashes, 'used_shard_sha256': shard_hashes,
        'freeze_receipt_sha256': sha256(config['freeze_receipt']),
        'freeze_source_config_path': receipt.get('source_config_path'),
        'training_config_sha256': sha256(config['training_config']) if config.get('training_config') else None,
        'data_smoke_provenance': dataset.manifest.get('smoke_provenance'),
        'document_identities': {doc: dataset.manifest['documents'][doc] for doc in documents},
        'source_sha256': source_hashes, 'used_attention_sha256': attention_hashes,
        'attention_manifest_sha256': sha256(config['attention_manifest']) if attention_manifest else None,
        'output_error_scene_status_counts': dict(output_status),
        'output_error_semantics': 'Per-head sparse attention renormalized on fixed+selected tokens, compared to full AV before output projection; relative L2 null for zero full output',
        'baseline_semantics': 'local audited formula fallback (unverified against current selector)' if local_audit else 'actual repository baseline_scores, legacy mean raw GQA scores',
        'teacher_semantics': 'independent causal softmax including self per query head; block sum then mean of 7 heads; raw middle mass unchanged',
        'tie_break': 'descending score then ascending original block ID',
        'oracle_scope': 'attention-mass upper bound among same-size middle selections; not task/output-error upper bound',
        'quality_semantics': 'exact shortest prefix of full ranking + fixed teacher mass; null when unreachable; cap reported separately; no interpolation',
        'aggregation_semantics': 'required_k/within_cap_k means conditional on available values; denominators reported per metric; unreachable/over-cap rates include all scenes',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__},
        'coverage': coverage_rows, 'quality': quality_rows,
    }
    json_write(output / 'summary.json', summary)
    if plot:
        write_svg(output / 'coverage.svg', coverage_rows)
    (output / 'RUNNING.json').unlink()
    return summary


def write_svg(path, rows):
    colors = {'RankLoss': '#2563eb', 'COBS': '#059669', 'Quest': '#d97706', 'Oracle': '#9333ea'}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430">',
             '<rect width="900" height="430" fill="white"/>']
    for offset, weighting in ((0, 'scene'), (450, 'document_equal')):
        panel = [r for r in rows if r['weighting'] == weighting]
        budgets = sorted({r['budget'] for r in panel})
        def x(b): return offset + 60 + 320 * budgets.index(b)/max(1, len(budgets)-1)
        def y(c): return 350 - 290*c
        parts += [f'<text x="{offset+60}" y="30" font-family="sans-serif">{weighting}: raw total coverage</text>',
                  f'<path d="M {offset+60} 60 V 350 H {offset+390}" fill="none" stroke="black"/>']
        for value in (0., .5, 1.):
            parts.append(f'<text x="{offset+25}" y="{y(value)+5}" font-size="12">{value}</text>')
        for b in budgets:
            parts.append(f'<text x="{x(b)-8}" y="370" font-size="12">{b}</text>')
        for j, method in enumerate(METHODS):
            data = sorted((r for r in panel if r['method'] == method), key=lambda r:r['budget'])
            points = ' '.join(f'{x(r["budget"]):.2f},{y(r["coverage"]):.2f}' for r in data)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="2"/>')
            parts.append(f'<text x="{offset+50+j*97}" y="400" fill="{colors[method]}" font-size="12">{method}</text>')
    parts.append('</svg>')
    Path(path).write_text('\n'.join(parts), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=1)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--local-audit', action='store_true')
    p.add_argument('--max-scenes', type=int); p.add_argument('--budgets', nargs='+', type=int, default=[4,8,16,32,64])
    p.add_argument('--targets', nargs='+', type=float, default=[.9,.95,.99]); p.add_argument('--budget-cap', type=int, default=64)
    p.add_argument('--require-output-error', action='store_true'); p.add_argument('--save-rankings', action='store_true')
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()
    if a.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    try:
        summary = evaluate(resolve_config(a.config), a.output, a.device, a.smoke, a.local_audit,
                           a.max_scenes, a.budgets, a.targets, a.budget_cap,
                           a.require_output_error, a.save_rankings, a.plot)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError) as error:
        p.exit(2, f'Diagnosis stopped: {error}\n')
    print(json.dumps({k: summary[k] for k in ('evaluation_label', 'device', 'scenes', 'documents', 'evaluation_seconds', 'output_error_scene_status_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
