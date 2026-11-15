"""Small CPU parity check against the unchanged repository evaluator."""
from pathlib import Path

import torch

from rankloss_diagnostic.core import audit_baseline_scores
from rankloss_diagnostic.data import read_json
from rankloss_diagnostic.prepare import resolve_config
from rankloss_diagnostic.run import json_write, load_runtime


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--result', required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    config = resolve_config(a.config)
    dataset, model, baseline, receipt, hashes, payload = load_runtime(config,False,True,'cpu')
    from utils.ranking_eval import evaluate_ranker
    new = read_json(Path(a.result)/'summary.json')
    original = evaluate_ranker(model,dataset,device='cpu',budgets=new['budgets'],max_scenes=len(new['selected_scene_indices']))
    aliases = {'RankLoss':'learned','COBS':'cobs_legacy_mean_logits','Quest':'quest_aabb','Oracle':'oracle'}
    delta = 0.
    for row in new['coverage']:
        if row['weighting'] == 'scene':
            legacy = original['groups']['overall']['methods'][aliases[row['method']]][f'coverage@{row["budget"]}']
            delta = max(delta,abs(row['coverage']-legacy))
    if delta > 1e-6:
        raise AssertionError(f'Coverage differs from existing evaluator: {delta}')
    score_delta = 0.
    for index in new['selected_scene_indices']:
        sample = dataset[index]
        actual, fallback = baseline(sample), audit_baseline_scores(sample)
        for method in ('COBS','Quest'):
            difference = float((actual[method]-fallback[method]).abs().max())
            score_delta = max(score_delta,difference)
            if not torch.allclose(actual[method],fallback[method],atol=2e-5,rtol=1e-5):
                raise AssertionError(f'Audit fallback formula differs from repository: {method}')
    result = {'status':'passed','device':'cpu','scenes':new['scenes'],
              'max_abs_coverage_delta_vs_repository':delta,
              'max_abs_fallback_score_delta_vs_repository':score_delta,
              'checkpoint_sha256':receipt['checkpoint_sha256'],'source_sha256':hashes,
              'nonformal_selection_validation_only':True}
    json_write(Path(a.result)/'repository_parity.json',result)
    print(result)


if __name__ == '__main__':
    main()
