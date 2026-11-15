import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

p=Path(__file__).resolve().parent
def status(stage,**kwargs):(p/'status.json').write_text(json.dumps(dict(stage=stage,**kwargs),indent=2))
status('collecting')
while True:
    logs=[p/f'collect_{i}.log' for i in range(4)]
    if any(f.exists() and 'Traceback' in f.read_text() for f in logs):
        status('failed',reason='collection traceback');raise RuntimeError('Collection failed')
    if all(f.exists() and f.read_text().rstrip().endswith('COMPLETE') for f in logs):break
    time.sleep(10)
assert len(list((p/'collected').glob('*.pt')))==72
for stage in ['merge','train','evaluate']:
    status(stage)
    with (p/f'{stage}.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(p/'engine.py'),stage],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
frozen={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'budget/best.pt',p/'budget/calibration.json',p/'protocol.json',p/'selected.json',p/'engine.py',p/'closed_loop.py']}
(p/'policy_frozen_before_hotpot.json').write_text(json.dumps(frozen,indent=2))
status('closed_loop_test')
def workers(stage):
    jobs=[];logs=[]
    for i in range(4):
        log=(p/f'{stage}_{i}.log').open('x');logs.append(log)
        args=[sys.executable,'-u',str(p/'closed_loop.py'),str(i)] if stage=='closed_loop' else [sys.executable,'-u',str(p/'engine.py'),'hotpot',str(i)]
        jobs.append(subprocess.Popen(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[job.wait() for job in jobs]
    for log in logs:log.close()
    if any(codes):status('failed',worker_stage=stage,codes=codes);raise RuntimeError((stage,codes))
workers('closed_loop')
status('hotpot_regression')
workers('hotpot')
status('evaluations_complete')
print('COMPLETE',flush=True)
