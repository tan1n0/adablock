"""Wait for collection, train matched branches, evaluate without changing old artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def status(stage):
    p=ROOT/'status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage)));tmp.replace(p)


def execute(script,args,gpu,logname):
    log=(ROOT/logname).open('x')
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false'),
                       stdout=log,stderr=subprocess.STDOUT)
    return p,log


def wait_all(processes):
    codes=[p.wait() for p,log in processes]
    for p,log in processes:log.close()
    if any(codes):raise RuntimeError(f'exit codes {codes}')


try:
    status('collecting')
    pids=json.loads((ROOT/'collection_pids.json').read_text())
    while True:
        logs=[(ROOT/f'collect_{i}.log').read_text() for i in range(4)]
        if any('Traceback (most recent call last)' in log for log in logs):
            raise RuntimeError('collection failed; inspect collect logs')
        if all(log.rstrip().endswith('COMPLETE') for log in logs):break
        for pid,log in zip(pids,logs):
            if log.rstrip().endswith('COMPLETE'):continue
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError(f'collection worker {pid} exited without COMPLETE')
        time.sleep(10)
    # Collection logs flush before process exit; wait until workers release their GPUs.
    time.sleep(5)
    status('preparing')
    wait_all([execute('prepare.py',[],'0','prepare.log')])
    status('training')
    wait_all([execute('train.py',[name],str(i),f'train_{name}.log') for i,name in enumerate(['S','A','B'])])
    status('evaluating')
    wait_all([execute('qa.py',['evaluate',str(i)],str(i),f'evaluate_{i}.log') for i in range(4)])
    wait_all([execute('qa.py',['summarize'],'0','summarize.log')])
except Exception:
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
