"""Locate exact CPU/GPU prefix differences without changing the running assessment."""
import gzip
import json
from pathlib import Path
import sys
import time

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
sys.path.insert(0,str(root))
import torch
from rankloss_diagnostic.core import coverage_curve,required_prefix,stable_order
from rankloss_diagnostic.prepare import resolve_config
from rankloss_diagnostic.run import FEATURE_KEYS,json_write,load_runtime

torch.set_num_threads(1)
ds,model,baseline,receipt,hashes,payload=load_runtime(resolve_config(root/'gpu_full_validation_config.json'),False,False,'cpu')
pilot=root/'results/gpu_pilot_v1'; differences=[]; total=0; max_delta=0.
with torch.inference_mode():
    for path in sorted((pilot/'shards').glob('*/scenes.jsonl.gz')):
        with gzip.open(path,'rt') as stream:
            for line in stream:
                d=json.loads(line);s=ds[d['scene_index']];n=len(s['block_ids'])
                score=model({k:s[k] for k in FEATURE_KEYS});order=stable_order(score,s['block_ids'])
                curve=coverage_curve(s['teacher_mass'],s['fixed_mass'],order)
                deltas=[abs(float(curve[min(k,n)])-d['methods']['RankLoss']['coverage'][j]) for j,k in enumerate(d['budgets'])]
                max_delta=max(max_delta,max(deltas))
                for j,target in enumerate(d['quality_targets']):
                    cpu=required_prefix(curve,target,64)['required_k'];gpu=d['methods']['RankLoss']['quality'][j]['required_k']
                    if cpu!=gpu:
                        differences.append({'scene_index':d['scene_index'],'scene_key':d['scene_key'],'candidate_count':n,
                                            'target_index':j,'target':target,'CPU_required_k':cpu,'GPU_required_k':gpu,
                                            'CPU_mass_at_GPU_k':float(curve[gpu]) if gpu is not None else None,
                                            'CPU_mass_at_CPU_k':float(curve[cpu]) if cpu is not None else None,
                                            'coverage_deltas':deltas,'CPU_first_64_block_ids':s['block_ids'][order[:64]].tolist()})
                total+=1
result={'analysis_split':'validation_used_for_checkpoint_selection','scenes':total,'quality_comparisons':total*8,
        'coverage_comparisons':total*5,'max_coverage_delta':max_delta,'prefix_differences':differences,
        'status':'within_teacher_mass_tolerance_with_recorded_rank_difference' if max_delta<=2e-4 else 'exceeds_tolerance',
        'explanation':'Same FP32 checkpoint and metric formula; CPU/GPU scores may order near-equal blocks differently. Exact K is computed independently on each actual ranking; it is not forced to agree.'}
json_write(pilot/'numerical_difference_audit.json',result)
print(json.dumps(result))
