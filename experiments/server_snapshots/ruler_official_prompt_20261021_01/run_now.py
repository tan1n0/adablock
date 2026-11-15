"""User-authorized immediate run; restart this file to resume persisted methods."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_budget_20261021_01')
OUT=ROOT/'evaluation'
OUT.mkdir(exist_ok=True)

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

if not (OUT/'selected.json').exists():
    rows=json.loads((ROOT/'corrected_inputs.json').read_text())
    # Interleave tasks at each length so early results cover all three tasks.
    tasks=['niah_single_1','niah_multivalue','vt']
    rows.sort(key=lambda r:(r['length'],r['source_line'],tasks.index(r['task'])))
    for r in rows:r['max_new']=128
    dump(OUT/'selected.json',rows)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    manifest.update(n=450,selected_sha256=hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest(),
        max_new_by_task={t:128 for t in tasks},prompt='official input + answer_prefix',
        user_override='Start full rerun immediately before midnight shutdown, without waiting for pilot',
        preflight='Skip repeated preflight; prior Full/all parity passed; pilot completeness not yet confirmed',
        resumable='Atomic per-method partial files, flushed completed-question journal',
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
    dump(OUT/'manifest.json',manifest)
for entry in ['deps','RULER']:
    if not (OUT/entry).exists():(OUT/entry).symlink_to(SOURCE/entry,target_is_directory=True)
env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(PYTHONPATH=str(SOURCE/'deps'),TOKENIZERS_PARALLELISM='false',
           RULER_RUN_DIR=str(OUT),RULER_SHARDS='4',RULER_MAX_NEW='128',RULER_SKIP_PREFLIGHT='1')
workers=[]
try:
    for shard in range(4):
        with (OUT/f'evaluate_{shard}.log').open('a') as log:
            workers.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),'evaluate',str(shard)],
                cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT))
    dump(ROOT/'status.json',dict(stage='evaluating',pids=[p.pid for p in workers],questions=450,
        pilot_gate_bypassed_by_user=True,max_new=128,directory=str(OUT)))
    while any(p.poll() is None for p in workers):
        if any(p.poll() not in [None,0] for p in workers):raise RuntimeError('Evaluation worker failed')
        time.sleep(5)
    with (OUT/'summarize.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'evaluate.py'),'summarize'],cwd=ROOT,
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(ROOT/'status.json',dict(stage='complete',questions=450,max_new=128))
except Exception as exc:
    for p in workers:
        if p.poll() is None:p.terminate()
    dump(ROOT/'status.json',dict(stage='failed',error=str(exc)))
    raise
