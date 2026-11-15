import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(PYTHONPATH=str(ROOT/'deps'),NLTK_DATA=str(ROOT/'nltk_data'),TOKENIZERS_PARALLELISM='false')

def status(stage,**extra):
    tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,**extra)));tmp.replace(ROOT/'status.json')

def launch(script,args,gpu,log):
    with (ROOT/log).open('a') as out:
        return subprocess.Popen([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,
            env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=out,stderr=subprocess.STDOUT)

try:
    status('preparing')
    if not (ROOT/'manifest.json').exists():
        assert launch('prepare_data.py',[],0,'prepare.log').wait()==0
    assert launch('evaluate.py',['test'],0,'allocation_test.log').wait()==0
    # One worker validates longest input before admitting the other replicas.
    status('preflight')
    first=launch('evaluate.py',['evaluate','0'],0,'evaluate_0.log')
    while not (ROOT/'preflight_0.json').exists():
        if first.poll() is not None:raise RuntimeError('worker 0 preflight failed')
        time.sleep(5)
    workers=[first]+[launch('evaluate.py',['evaluate',str(i)],i,f'evaluate_{i}.log') for i in range(1,4)]
    status('evaluating',worker_pids=[p.pid for p in workers],questions=450)
    while any(p.poll() is None for p in workers):
        failed=[p.pid for p in workers if p.poll() not in [None,0]]
        if failed:
            for p in workers:
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'workers failed: {failed}')
        time.sleep(10)
    status('summarizing')
    assert launch('evaluate.py',['summarize'],0,'summarize.log').wait()==0
    status('complete',questions=450)
except Exception:
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
