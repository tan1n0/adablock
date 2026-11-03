import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(name,value):(ROOT/name).write_text(json.dumps(value,indent=2))
dump('verification_status.json',dict(stage='production_parity_and_repeated_timing'))
jobs=[]
pending=[32768,16384,8192]
busy=set()
while pending:
    state=json.loads((ROOT/'status.json').read_text())['stage']
    if state=='failed':raise RuntimeError('Quality run failed; no timing jobs started')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for line in memory.splitlines():
        gpu,used=map(int,line.split(','))
        if gpu in busy or used>=100 or not pending:continue
        length=pending.pop(0);busy.add(gpu)
        with (ROOT/f'verification_{length}.log').open('w') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'verify_timing.py'),str(length)],stdout=f,stderr=subprocess.STDOUT,
                                         env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))))
        print('START',length,'GPU',gpu,flush=True)
    if pending:time.sleep(10)
codes=[p.wait() for p in jobs]
if any(codes):
    dump('verification_status.json',dict(stage='failed',codes=codes));raise RuntimeError(codes)
while json.loads((ROOT/'status.json').read_text())['stage']!='complete':
    if json.loads((ROOT/'status.json').read_text())['stage']=='failed':raise RuntimeError('Quality run failed')
    time.sleep(10)
sys.path.insert(0,'/root/adablock_ruler_harder_20261022_01/deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
values=[json.loads((ROOT/f'verification_{n}.json').read_text()) for n in [8192,16384,32768]]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for version in ['baseline','gpu','gpu_digest']:
    axes[0].plot([8,16,32],[v['index_seconds']['gpu_digest' if version=='gpu_digest' else 'baseline'] for v in values],marker='o',label=version)
    axes[1].plot([8,16,32],[v['timing'][version]['median_ms_per_step'] for v in values],marker='o',label=version)
axes[0].set_ylabel('Summary construction (seconds)')
axes[1].set_ylabel('Decode (ms/token, median of 3)')
for ax in axes:ax.set_xlabel('Context (K tokens)');ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'runtime_comparison.png',dpi=170);fig.savefig(ROOT/'runtime_comparison.pdf')
dump('verification_status.json',dict(stage='complete'))
