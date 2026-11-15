import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent

def execute(stage, gpu='0', shard=None):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false')
    args=[sys.executable,'-u',str(ROOT/'experiment.py'),stage]
    if shard is not None:args+=['--shard',str(shard)]
    log=(ROOT/f'{stage}{"_"+str(shard) if shard is not None else ""}.log').open('x')
    p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    return p,log

try:
    for stage in ['self_test','prepare','train']:
        p,log=execute(stage)
        code=p.wait();log.close()
        if code:raise RuntimeError(f'{stage} exited {code}')
    processes=[execute('evaluate',str(i),i) for i in range(4)]
    codes=[p.wait() for p,log in processes]
    for p,log in processes:log.close()
    if any(codes):raise RuntimeError(f'evaluate exit codes {codes}')
    p,log=execute('summarize');code=p.wait();log.close()
    if code:raise RuntimeError(f'summarize exited {code}')
except Exception:
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
