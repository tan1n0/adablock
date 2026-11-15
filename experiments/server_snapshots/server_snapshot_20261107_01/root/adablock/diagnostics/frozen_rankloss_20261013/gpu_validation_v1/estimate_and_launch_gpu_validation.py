"""Use the completed representative pilot to launch exactly one authorized full run."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
pilot=root/'results/gpu_pilot_v1'
summary=json.loads((pilot/'summary.json').read_text())
if summary['status']!='complete' or summary['scenes']!=256:
    raise RuntimeError('Representative pilot is incomplete')
if summary['cpu_device_coverage_max_delta']>2e-4:
    raise RuntimeError('CPU/GPU metric parity did not pass')
manifest=json.loads(Path('/root/adablock/data/ranker_cobs_validation_all28_20261004/manifest.json').read_text())
rows=[json.loads(p.read_text()) for p in (pilot/'shards').glob('*/summary.json')]
details=[]
estimated_worker_seconds=0.
for representative,document_ids in ((0,range(0,4)),(4,range(4,8)),(8,range(8,12)),(12,range(12,16))):
    selected=[r for r in rows if r['document_scene_counts'].get(str(representative))]
    rate=sum(r['compute_write_seconds'] for r in selected)/sum(r['scenes'] for r in selected)
    overhead=sum(r['hash_seconds']+r['load_seconds'] for r in selected)/len(selected)
    entries=[s for s in manifest['shards'] if s['document_index'] in document_ids]
    scenes=sum(s['scenes'] for s in entries)
    predicted=rate*scenes+overhead*len(entries)
    estimated_worker_seconds+=predicted
    details.append({'representative_document':representative,'scenes_in_bin':scenes,
                    'compute_write_seconds_per_scene':rate,'hash_load_seconds_per_shard':overhead,
                    'predicted_worker_seconds':predicted})
estimate={'basis':'32 spread scenes at layers0/14 for one doc per length bin; includes hash/load/write costs',
          'pilot_wall_seconds':summary['wall_seconds'],'pilot_scenes':summary['scenes'],
          'predicted_four_worker_wall_seconds':estimated_worker_seconds/4,
          'remaining_range_seconds':[estimated_worker_seconds/4*.8,estimated_worker_seconds/4*1.8+30],
          'estimated_scene_output_bytes':sum(r['output_bytes'] for r in rows)/256*414064,
          'worker_max_rss_kib':summary['max_worker_rss_kib'],
          'gpu_peak_allocated_bytes':summary['max_cuda_peak_allocated_bytes'],
          'cpu_gpu_score_max_delta':summary['cpu_device_score_max_delta'],
          'cpu_gpu_coverage_max_delta':summary['cpu_device_coverage_max_delta'],
          'cpu_gpu_prefix_disagreements':summary['cpu_device_prefix_disagreement_count'],'bins':details}
(root/'pilot_estimate.json').write_text(json.dumps(estimate,indent=2))
if estimate['remaining_range_seconds'][1]>8*3600 or estimate['estimated_scene_output_bytes']>100*1024**3:
    print(json.dumps({'not_started':'Estimate exceeds bounded plan; needs review','estimate':estimate}));raise SystemExit(2)
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'rankloss_diagnostic.validation' in argv and Path(os.readlink(p/'cwd'))==root:
            raise RuntimeError(f'Existing diagnostic process{p.name}; do not start a duplicate')
    except (OSError,ValueError):pass
output=root/'results/full_validation_gpu_v1'
command=['/root/miniconda3/envs/ada/bin/python','-m','rankloss_diagnostic.validation',
         '--config','gpu_full_validation_config.json','--output','results/full_validation_gpu_v1',
         '--analysis-split','validation','--device','cuda','--gpus','0','1','2','3',
         '--workers','4','--batch-size','8']
if output.exists():command.append('--resume')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with (root/'full_validation.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'command':command,'output_directory':str(output),
        'analysis_split':'validation_used_for_checkpoint_selection','estimated_range_seconds':estimate['remaining_range_seconds'],
        'monitoring_clock':'elapsed durations only; remote wall clock is not presented as user time'}
path=root/'launch.json'
if path.exists():path=root/f'launch_{process.pid}.json'
path.write_text(json.dumps(record,indent=2))
print(json.dumps({'started':record,'estimate':estimate}))
