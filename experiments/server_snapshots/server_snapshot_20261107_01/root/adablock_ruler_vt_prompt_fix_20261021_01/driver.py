"""Queue behind original workers; gate the full rerun on complete pilot outputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_budget_20261021_01')
ORIGINAL_PIDS=[4946,5277,5278,5279]
env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(PYTHONPATH=str(SOURCE/'deps'),TOKENIZERS_PARALLELISM='false')

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def status(stage,**extra):
    dump(ROOT/'status.json',dict(stage=stage,**extra))

def busy(gpu):
    path=Path(f'/proc/{ORIGINAL_PIDS[gpu]}/cmdline')
    return path.exists() and b'evaluate.py' in path.read_bytes()

def setup(name,rows,limit):
    directory=ROOT/name;directory.mkdir(exist_ok=False)
    for entry in ['deps','RULER']:(directory/entry).symlink_to(SOURCE/entry,target_is_directory=True)
    selected=[dict(r,max_new=limit) for r in rows]
    dump(directory/'selected.json',selected)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    manifest.update(selected_sha256=hashlib.sha256((directory/'selected.json').read_bytes()).hexdigest(),
                    source_directory=str(SOURCE),n=len(rows),max_new=limit,
                    source_indices=[r['original_index'] for r in rows])
    dump(directory/'manifest.json',manifest)
    return directory

def launch(directory,mode,shard,gpu,limit,shards=4):
    settings=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),RULER_RUN_DIR=str(directory),
                  RULER_SHARDS=str(shards),RULER_MAX_NEW=str(limit))
    with (directory/f'{mode}_{shard}.log').open('a') as out:
        return subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),mode,str(shard)],
                                cwd=ROOT,env=settings,stdout=out,stderr=subprocess.STDOUT)

def inspect_pilot(directory):
    rows=[json.loads(line) for line in (directory/'qa_0.jsonl').read_text().splitlines()]
    assert len(rows)==3 and {r['length'] for r in rows}=={8192,16384,32768}
    checks=[]
    for r in rows:
        assert len(r['methods'])==13
        for name,m in r['methods'].items():
            # Judge termination and whether an answer was reached, not correctness.
            variables=re.findall(r'\b[A-Z]{5}\b',m['prediction'])
            checks.append(dict(length=r['length'],method=name,tokens=len(m['ids']),
                eos=m['ended_with_eos'],truncated=m['hit_generation_limit'],
                has_variable=bool(variables),prediction=m['prediction'],answers=r['answers'],score=m['score']))
    passed=all(x['eos'] and not x['truncated'] and x['has_variable'] for x in checks)
    report=dict(passed=passed,checks=checks,max_generated_tokens=max(x['tokens'] for x in checks))
    dump(directory/'completion_check.json',report)
    return report

try:
    source_manifest=json.loads((SOURCE/'manifest.json').read_text())
    source_sha=hashlib.sha256((SOURCE/'selected.json').read_bytes()).hexdigest()
    assert source_sha==source_manifest['selected_sha256']
    source_rows=json.loads((SOURCE/'selected.json').read_text())
    rows=[dict(r,original_index=i) for i,r in enumerate(source_rows) if r['task']=='vt']
    assert len(rows)==150
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb',local_files_only=True)
    raw={length:{r['index']:r for r in [json.loads(line) for line in (SOURCE/f'data/{length}/vt/validation.jsonl').read_text().splitlines()]} for length in [8192,16384,32768]}
    for row in rows:
        original=raw[row['length']][row['sample_index']]
        assert tok.encode(original['input'],add_special_tokens=False)==row['input_ids']
        assert original['outputs']==row['answers'] and original['answer_prefix']
        row['original_input_sha256']=row['input_sha256']
        row['answer_prefix']=original['answer_prefix']
        row['input_ids']=tok.encode(original['input']+original['answer_prefix'],add_special_tokens=False)
        row['input_sha256']=hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()
        row['prompt_tokens']=len(row['input_ids'])
    dump(ROOT/'corrected_inputs.json',rows)
    # Deterministic, chosen without examining answers or scores.
    pilot=[next(r for r in rows if r['length']==length) for length in [8192,16384,32768]]
    dump(ROOT/'protocol.json',dict(source=str(SOURCE),source_selected_sha256=source_sha,
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
        questions=150,pilot_questions=3,
        pilot_selection='First original VT sample at each length; no score-based selection',
        candidate_limits=[30,128,512,1024],ratios=[.10,.15,.25,.40],lengths=[8192,16384,32768],
        prompt_fix='Use official input + answer_prefix; original context and questions unchanged',
        unchanged=['context','question','answers','checkpoints','budget rule','greedy decoding','scoring'],
        cache_reserve='max_new + 32, identical for all methods',
        input_length_note='Original context preserved; official answer prefix restored. Record actual prompt length; generation allowance may exceed original nominal total-length bin.',
        pilot_in_final=True,gate='All 39 responses reach EOS without truncation and contain a five-letter variable-like token; correctness is not a gate.'))
    status('waiting_for_gpu',original_pids=ORIGINAL_PIDS)
    while all(busy(gpu) for gpu in range(4)):time.sleep(15)
    gpu=next(gpu for gpu in range(4) if not busy(gpu))
    selected_limit=None
    for limit in [30,128,512,1024]:
        directory=setup(f'pilot_{limit}',pilot,limit)
        proc=launch(directory,'evaluate',0,gpu,limit,shards=1)
        status('pilot',gpu=gpu,pid=proc.pid,max_new=limit,directory=str(directory))
        if proc.wait()!=0:raise RuntimeError(f'Pilot failed; inspect {directory}')
        check=inspect_pilot(directory)
        if check['passed']:
            selected_limit=limit;break
        if not any(x['truncated'] for x in check['checks']):
            raise RuntimeError('Pilot ended but did not meet answer-completeness checks; inspect outputs before rerun')
    if selected_limit is None:raise RuntimeError('Pilot remains truncated at 1024 tokens; full rerun not started')
    dump(ROOT/'pilot_passed.json',dict(max_new=selected_limit,directory=str(directory),
        max_generated_tokens=check['max_generated_tokens'],responses=39))
    status('pilot_passed_waiting_for_gpus',max_new=selected_limit)
    while any(busy(gpu) for gpu in range(4)):time.sleep(15)
    final=setup('evaluation',rows,selected_limit)
    workers=[launch(final,'evaluate',i,i,selected_limit) for i in range(4)]
    status('evaluating',max_new=selected_limit,questions=150,pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if any(p.poll() not in [None,0] for p in workers):
            for p in workers:
                if p.poll() is None:p.terminate()
            raise RuntimeError('Evaluation worker failed')
        time.sleep(10)
    status('summarizing',max_new=selected_limit)
    if launch(final,'summarize',0,0,selected_limit).wait()!=0:raise RuntimeError('Summary failed')
    status('complete',max_new=selected_limit,questions=150)
except Exception:
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
