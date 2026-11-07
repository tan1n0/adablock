"""CPU recomputation of every selected pilot scene, including long late queries."""
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
dataset,model,baseline,receipt,hashes,payload=load_runtime(resolve_config(root/'gpu_full_validation_config.json'),False,False,'cpu')
start=time.perf_counter(); total=0; max_delta=0.; prefixes=0; candidate_min=None;candidate_max=0
pilot=root/'results/gpu_pilot_v1'
with torch.inference_mode():
    for p in sorted((pilot/'shards').glob('*/scenes.jsonl.gz')):
        with gzip.open(p,'rt') as stream:
            for line in stream:
                d=json.loads(line);s=dataset[d['scene_index']];n=len(s['block_ids'])
                scores=model({k:s[k] for k in FEATURE_KEYS})
                curve=coverage_curve(s['teacher_mass'],s['fixed_mass'],stable_order(scores,s['block_ids']))
                for j,k in enumerate(d['budgets']):
                    max_delta=max(max_delta,abs(float(curve[min(k,n)])-d['methods']['RankLoss']['coverage'][j]))
                for j,target in enumerate(d['quality_targets']):
                    required=required_prefix(curve,target,64)['required_k']
                    prefixes += required != d['methods']['RankLoss']['quality'][j]['required_k']
                candidate_min=n if candidate_min is None else min(candidate_min,n);candidate_max=max(candidate_max,n);total+=1
result={'status':'passed' if max_delta<=2e-4 and prefixes==0 else 'failed',
        'analysis_split':'validation_used_for_checkpoint_selection','scenes':total,
        'candidate_min':candidate_min,'candidate_max':candidate_max,
        'all_pilot_CPU_GPU_coverage_max_delta':max_delta,'all_pilot_prefix_disagreement_count':prefixes,
        'CPU_verification_seconds':time.perf_counter()-start}
json_write(pilot/'all_scene_cpu_gpu_parity.json',result)
print(json.dumps(result))
if result['status']!='passed':raise SystemExit(2)
