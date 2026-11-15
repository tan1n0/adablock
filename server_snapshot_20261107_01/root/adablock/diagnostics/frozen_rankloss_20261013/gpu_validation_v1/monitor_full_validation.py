"""Read-only bounded progress; ETA uses recent completed work, not a 16-scene extrapolation."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
out=root/'results/full_validation_gpu_v1'
launch=json.loads((root/'launch.json').read_text())
process_path=Path('/proc')/str(launch['pid'])
process_status=subprocess.run(['ps','-p',str(launch['pid']),'-o','pid=,etime=,pcpu=,rss=,stat='],capture_output=True,text=True).stdout.strip()
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15).stdout.strip()
result={'pid':launch['pid'],'process':process_status,'gpu':gpu,'output_directory':str(out)}
if (out/'progress.json').exists():
    progress=json.loads((out/'progress.json').read_text());result['progress']=progress
    if progress.get('completed_scenes') and progress.get('wall_seconds'):
        rate=progress['completed_scenes']/progress['wall_seconds']
        result['cumulative_scenes_per_second']=rate
        result['naive_remaining_seconds']=(414064-progress['completed_scenes'])/rate
work=[]
if (out/'work').exists():
    for p in sorted((out/'work').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:4]:
        work.append(json.loads(p.read_text()))
result['recent_worker_progress']=work
if (out/'summary.json').exists():
    d=json.loads((out/'summary.json').read_text());result['summary']={k:d[k] for k in ('status','scenes','shards','wall_seconds','output_bytes','max_worker_rss_kib','max_cuda_peak_allocated_bytes','cpu_device_coverage_max_delta','cpu_device_prefix_disagreement_count')}
log=root/'full_validation.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-3500));tail=stream.read().decode(errors='replace')
    result['log_tail']=tail.splitlines()[-6:]
print(json.dumps(result))
