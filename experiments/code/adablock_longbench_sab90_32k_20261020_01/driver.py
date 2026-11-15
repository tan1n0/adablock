import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent

def status(stage, **extra):
    target=ROOT/'status.json'
    tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,**extra)))
    tmp.replace(target)

def launch(args, gpu, log):
    with (ROOT/log).open('x') as out:
        return subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),*args],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                TOKENIZERS_PARALLELISM='false'),stdout=out,stderr=subprocess.STDOUT)

try:
    status('preparing')
    assert launch(['prepare'],0,'prepare.log').wait()==0
    status('evaluating')
    workers=[launch(['evaluate',str(i)],i,f'evaluate_{i}.log') for i in range(4)]
    status('evaluating',worker_pids=[p.pid for p in workers])
    codes=[p.wait() for p in workers]
    assert codes==[0]*4, codes
    assert launch(['summarize'],0,'summarize.log').wait()==0
    status('complete',questions=300)
except Exception:
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
