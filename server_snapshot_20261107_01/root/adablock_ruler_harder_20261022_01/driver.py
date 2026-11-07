"""Audit and smoke test before running the harder-task subset. No gate bypass."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'protocol.json').read_text())
env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(PYTHONPATH=str(ROOT/'deps'),NLTK_DATA=str(ROOT/'nltk_data'),TOKENIZERS_PARALLELISM='false',
           RULER_SHARDS='4',RULER_MAX_NEW='128',RULER_SKIP_PREFLIGHT='0')
workers=[]

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def status(stage,**extra):dump(ROOT/'status.json',dict(stage=stage,**extra))

def stage_directory(name,rows):
    p=ROOT/name;p.mkdir(exist_ok=True)
    selected=json.dumps(rows)
    if (p/'selected.json').exists():assert (p/'selected.json').read_text()==selected
    else:(p/'selected.json').write_text(selected)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    manifest.update(n=len(rows),selected_sha256=hashlib.sha256(selected.encode()).hexdigest())
    dump(p/'manifest.json',manifest)
    for entry in ['deps','RULER']:
        if not (p/entry).exists():(p/entry).symlink_to(ROOT/entry,target_is_directory=True)
    return p

def launch(directory,mode,shard):
    with (directory/f'{mode}_{shard}.log').open('a') as out:
        return subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),mode,str(shard)],cwd=ROOT,
            env=dict(env,CUDA_VISIBLE_DEVICES=str(shard),RULER_RUN_DIR=str(directory)),stdout=out,stderr=subprocess.STDOUT)

def run_group(directory,stage):
    global workers
    workers=[launch(directory,'evaluate',i) for i in range(4)]
    status(stage,directory=str(directory),pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if any(p.poll() not in [None,0] for p in workers):raise RuntimeError(f'{stage} worker failed')
        time.sleep(10)

try:
    if not (ROOT/'manifest.json').exists():
        status('preparing_and_auditing')
        with (ROOT/'prepare.log').open('a') as out:
            subprocess.run([sys.executable,'-u',str(ROOT/'prepare.py')],cwd=ROOT,env=env,
                stdout=out,stderr=subprocess.STDOUT,check=True)
    rows=json.loads((ROOT/'selected.json').read_text())
    smoke=stage_directory('smoke',[r for r in rows if r['source_line']==0])
    assert len(json.loads((smoke/'selected.json').read_text()))==9
    if not (ROOT/'smoke_passed.json').exists():
        run_group(smoke,'alignment_and_smoke')
        answers=[json.loads(s) for f in smoke.glob('qa_[0-3].jsonl') for s in f.read_text().splitlines()]
        assert len(answers)==len({r['index'] for r in answers})==9
        problems=[];report=[]
        for r in answers:
            assert len(r['methods'])==13
            for name,m in r['methods'].items():
                record=dict(task=r['task'],length=r['length'],method=name,prediction=m['prediction'],
                    score=m['score'],tokens=len(m['ids']),eos=m['ended_with_eos'],truncated=m['hit_generation_limit'])
                report.append(record)
                if not m['prediction'].strip() or not m['ended_with_eos'] or m['hit_generation_limit']:
                    problems.append(record)
        full_scores={r['task']+'_'+str(r['length']):r['methods']['full']['score'] for r in answers}
        dump(ROOT/'smoke_report.json',dict(problems=problems,full_scores=full_scores,responses=report))
        if problems:raise RuntimeError('Official-cap smoke has empty/truncated outputs; inspect before changing protocol')
        # Do not select tasks or settings according to a favorable B-A difference.
        if any(v==0 for v in full_scores.values()):
            raise RuntimeError('Full has a zero-score smoke cell; inspect correctness/difficulty before expensive run')
        dump(ROOT/'smoke_passed.json',dict(questions=9,responses=117,full_scores=full_scores,
            note='Single sample per cell: does not establish difficulty or statistical performance'))
    final=stage_directory('evaluation',rows)
    run_group(final,'evaluating')
    status('summarizing')
    assert launch(final,'summarize',0).wait()==0
    status('complete',questions=len(rows))
except Exception:
    for p in workers:
        if p.poll() is None:p.terminate()
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
