"""Strict FP32 recomputation of the single recorded pilot border case; no teacher changes."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
sys.path.insert(0,str(root))
import torch
from rankloss_diagnostic.core import coverage_curve,required_prefix,stable_order
from rankloss_diagnostic.prepare import resolve_config
from rankloss_diagnostic.run import FEATURE_KEYS,json_write,load_runtime

# Let the authorized full run release its worker contexts before using one card for this audit.
start=time.perf_counter()
while json.loads((root/'results/full_validation_gpu_v1/progress.json').read_text()).get('status')!='complete':
    if time.perf_counter()-start>180:raise RuntimeError('Full run still active; defer small precision audit')
    time.sleep(2)
status=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
gpu0=[list(map(int,line.split(','))) for line in status.splitlines() if line.strip().startswith('0,')][0]
if gpu0[1]>1024 or gpu0[2]>10:raise RuntimeError('GPU0 occupied by another job; precision audit deferred')
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
ds,cpu,baseline,receipt,hashes,payload=load_runtime(resolve_config(root/'gpu_full_validation_config.json'),False,False,'cpu')
from models.block_ranker import load_ranker
from utils.ranking_batch import pack_scenes,forward_packed
gpu,_=load_ranker(receipt['checkpoint_path'],'cuda:0');gpu.float().eval().requires_grad_(False)
records=[]
with gzip.open(root/'results/gpu_pilot_v1/shards/0000/scenes.jsonl.gz','rt') as stream:
    records=[json.loads(line) for line in stream]
j=next(i for i,d in enumerate(records) if d['scene_index']==98)
batch_records=records[(j//8)*8:(j//8+1)*8];samples=[ds[d['scene_index']] for d in batch_records]
sample=ds[98];offset=j%8
with torch.inference_mode():
    cp=pack_scenes(samples,'cpu');gp=pack_scenes(samples,'cuda:0')
    cpu_batch=forward_packed(cpu,cp)[cp['offsets'][offset]:cp['offsets'][offset+1]].float().cpu()
    gpu_batch=forward_packed(gpu,gp)[gp['offsets'][offset]:gp['offsets'][offset+1]].float().cpu()
    cpu_single=cpu({k:sample[k] for k in FEATURE_KEYS}).float().cpu()
    gpu_single=gpu({k:sample[k].to('cuda:0') if isinstance(sample[k],torch.Tensor) else sample[k] for k in FEATURE_KEYS}).float().cpu()
scores={'cpu_single':cpu_single,'cpu_batch':cpu_batch,'gpu_single':gpu_single,'gpu_batch':gpu_batch}
orders={name:stable_order(value,sample['block_ids']) for name,value in scores.items()}
target=records[j]['quality_targets'][7]
results={}
for name,order in orders.items():
    curve=coverage_curve(sample['teacher_mass'],sample['fixed_mass'],order)
    results[name]={'selected64':sample['block_ids'][order[:64]].tolist(),'coverage64':float(curve[64]),
                   'required_k_to_original_GPU_target':required_prefix(curve,target,64)['required_k']}
cpu_only=set(results['cpu_single']['selected64'])-set(results['gpu_batch']['selected64'])
gpu_only=set(results['gpu_batch']['selected64'])-set(results['cpu_single']['selected64'])
index={int(b):i for i,b in enumerate(sample['block_ids'])}
border=[]
for a in sorted(cpu_only):
    for b in sorted(gpu_only):
        border.append({'CPU_only_block':a,'GPU_only_block':b,
                       'teacher_CPU_only_mass':float(sample['teacher_mass'][index[a]]),
                       'teacher_GPU_only_mass':float(sample['teacher_mass'][index[b]]),
                       'teacher_gap_GPU_minus_CPU':float(sample['teacher_mass'][index[b]]-sample['teacher_mass'][index[a]]),
                       'score_gap_CPU_minus_GPU_block_by_mode':{name:float(value[index[a]]-value[index[b]]) for name,value in scores.items()},
                       'border_scores':{name:{str(a):float(value[index[a]]),str(b):float(value[index[b]])} for name,value in scores.items()}})
result={'analysis_split':'validation_used_for_checkpoint_selection','scene_index':98,'scene_key':records[j]['scene_key'],
        'checkpoint_sha256':receipt['checkpoint_sha256'],'dtype':'float32','TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'TF32_cudnn':torch.backends.cudnn.allow_tf32,'autocast_enabled':torch.is_autocast_enabled(),
        'prefix_tolerance_unchanged':1e-7,'teacher_unchanged':True,'original_GPU_target':target,
        'original_GPU_coverage64':records[j]['methods']['RankLoss']['coverage'][4],
        'reproduced_GPU_delta':abs(results['gpu_batch']['coverage64']-records[j]['methods']['RankLoss']['coverage'][4]),
        'batch_indices':[d['scene_index'] for d in batch_records],'border':border,'modes':results,
        'CPU_GPU_score_max_delta':float((cpu_single-gpu_batch).abs().max())}
assert result['reproduced_GPU_delta']<1e-10
json_write(root/'results/gpu_pilot_v1/precision_border_audit.json',result)
note=['# FP32 numerical audit','', 'validation_used_for_checkpoint_selection. This is a validation debug case, not an independent result.',
      'TF32 is disabled; no autocast or reduced precision. Teacher mass and prefix hit tolerance1e-7 remain unchanged.',
      'CPU single, CPU batched, GPU single and original-shape GPU batched scores were recomputed for the affected scene.',
      'The original GPU coverage is reproduced exactly. Near-equal FP32 scores order the border blocks differently; exact K is computed on each actual ranking.',
      '', json.dumps(result,indent=2), '',
      'COBS uses the actual repository cobs_legacy_mean_logits baseline; fixed sink/local/self counted once, raw middle mass retained.',
      'No scene-level independent confidence interval is claimed. All-scene pilot CPU recomputation is retained including its initial strict-agreement failed flag and the detailed numerical audit.']
(root/'results/gpu_pilot_v1/NUMERICAL_AUDIT.md').write_text('\n'.join(note),encoding='utf-8')
print(json.dumps(result))
