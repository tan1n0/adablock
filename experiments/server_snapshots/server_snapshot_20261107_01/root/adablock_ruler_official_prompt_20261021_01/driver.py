"""Restore official prompts; gate the 450-question rerun on a nine-question pilot."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_budget_20261021_01')
TASKS=['niah_single_1','niah_multivalue','vt']
LENGTHS=[8192,16384,32768]
env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(PYTHONPATH=str(SOURCE/'deps'),TOKENIZERS_PARALLELISM='false')
workers=[]

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def status(stage,**extra):
    dump(ROOT/'status.json',dict(stage=stage,**extra))

def setup(name,rows,limits):
    directory=ROOT/name;directory.mkdir(exist_ok=False)
    for entry in ['deps','RULER']:(directory/entry).symlink_to(SOURCE/entry,target_is_directory=True)
    selected=[dict(r,max_new=limits[r['task']]) for r in rows]
    dump(directory/'selected.json',selected)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    manifest.update(selected_sha256=hashlib.sha256((directory/'selected.json').read_bytes()).hexdigest(),
                    source_directory=str(SOURCE),n=len(rows),max_new_by_task=limits,
                    source_indices=[r['original_index'] for r in rows])
    dump(directory/'manifest.json',manifest)
    return directory

def launch(directory,mode,shard,limits):
    settings=dict(env,CUDA_VISIBLE_DEVICES=str(shard),RULER_RUN_DIR=str(directory),
                  RULER_SHARDS='4',RULER_MAX_NEW=str(max(limits.values())))
    with (directory/f'{mode}_{shard}.log').open('a') as out:
        return subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate.py'),mode,str(shard)],
                                cwd=ROOT,env=settings,stdout=out,stderr=subprocess.STDOUT)

def run_group(directory,stage,limits):
    global workers
    workers=[launch(directory,'evaluate',i,limits) for i in range(4)]
    status(stage,directory=str(directory),max_new_by_task=limits,pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if any(p.poll() not in [None,0] for p in workers):
            raise RuntimeError(f'Worker failed in {directory}')
        time.sleep(10)

def inspect_pilot(directory):
    rows=[json.loads(line) for f in directory.glob('qa_[0-3].jsonl') for line in f.read_text().splitlines()]
    assert len(rows)==len({r['index'] for r in rows})==9
    assert {(r['task'],r['length']) for r in rows}=={(t,l) for t in TASKS for l in LENGTHS}
    checks=[]
    for r in rows:
        assert len(r['methods'])==13
        for name,m in r['methods'].items():
            pattern=r'\b[A-Z]{5}\b' if r['task']=='vt' else r'\b\d+\b'
            checks.append(dict(task=r['task'],length=r['length'],method=name,tokens=len(m['ids']),
                eos=m['ended_with_eos'],truncated=m['hit_generation_limit'],
                expected_answer_format=bool(re.search(pattern,m['prediction'])),
                prediction=m['prediction'],answers=r['answers'],score=m['score']))
    # Wrong/refused sparse answers remain evaluation outcomes, not tuning targets.
    truncated_tasks=sorted({x['task'] for x in checks if x['truncated'] or not x['eos']})
    malformed_full=[x for x in checks if x['method']=='full' and not x['expected_answer_format']]
    empty=[x for x in checks if not x['prediction'].strip()]
    passed=not truncated_tasks and not malformed_full and not empty
    report=dict(passed=passed,truncated_tasks=truncated_tasks,malformed_full=malformed_full,
        empty_outputs=empty,checks=checks,max_generated_tokens_by_task={
            t:max(x['tokens'] for x in checks if x['task']==t) for t in TASKS})
    dump(directory/'completion_check.json',report)
    return report

try:
    status('preparing')
    source_manifest=json.loads((SOURCE/'manifest.json').read_text())
    source_sha=hashlib.sha256((SOURCE/'selected.json').read_bytes()).hexdigest()
    assert source_sha==source_manifest['selected_sha256']
    source_rows=json.loads((SOURCE/'selected.json').read_text())
    rows=[dict(r,original_index=i) for i,r in enumerate(source_rows)]
    assert len(rows)==450
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb',local_files_only=True)
    raw={(task,length):[json.loads(line) for line in
        (SOURCE/f'data/{length}/{task}/validation.jsonl').read_text().splitlines()]
        for task in TASKS for length in LENGTHS}
    offsets=defaultdict(int)
    for row in rows:
        # Upstream retrieval indices can repeat; preserve source line ordering.
        cell=(row['task'],row['length'])
        original=raw[cell][offsets[cell]]
        row['source_line']=offsets[cell]
        offsets[cell]+=1
        assert original['index']==row['sample_index']
        assert tok.encode(original['input'],add_special_tokens=False)==row['input_ids']
        assert original['outputs']==row['answers'] and original['answer_prefix']
        # Match upstream call_api.py, without wrapping in a second chat template.
        prompt=original['input']+original['answer_prefix']
        assert prompt.count('<|begin_of_text|>')==1
        assert prompt.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'+original['answer_prefix'])
        row['original_input_sha256']=row['input_sha256']
        row['answer_prefix']=original['answer_prefix']
        row['input_ids']=tok.encode(prompt,add_special_tokens=False)
        assert tok.decode(row['input_ids'],skip_special_tokens=False)==prompt
        row['input_sha256']=hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()
        row['prompt_tokens']=len(row['input_ids'])
    dump(ROOT/'corrected_inputs.json',rows)
    pilot=[next(r for r in rows if r['task']==task and r['length']==length)
           for length in LENGTHS for task in TASKS]
    limits={'niah_single_1':128,'niah_multivalue':128,'vt':30}
    dump(ROOT/'protocol.json',dict(source=str(SOURCE),source_selected_sha256=source_sha,
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
        questions=450,pilot_questions=9,pilot_responses=117,
        pilot_selection='First original sample in each task/length cell; no score-based selection',
        initial_max_new_by_task=limits,escalation=[30,128,512,1024],ratios=[.10,.15,.25,.40],
        lengths=LENGTHS,tasks=TASKS,
        prompt_fix='Exactly official input + answer_prefix; round-trip tokenization verified for all 450 inputs',
        unchanged=['context','question','answers','checkpoints','budget rule','greedy decoding','scoring'],
        cache_reserve='max(256, max generation allowance + 32), identical for all methods',
        input_length_stats={str(l):[min(r['prompt_tokens'] for r in rows if r['length']==l),
            max(r['prompt_tokens'] for r in rows if r['length']==l)] for l in LENGTHS},
        input_length_note='Original context preserved with official answer prefix restored. Actual input length recorded; increased generation allowance may exceed original nominal total-length bin.',
        pilot_in_final=True,
        gate='117 nonempty responses reach EOS without truncation; Full reaches expected answer format in every cell. No score threshold.'))
    for attempt in range(4):
        directory=setup(f'pilot_{attempt:02d}',pilot,limits)
        run_group(directory,'pilot',limits)
        check=inspect_pilot(directory)
        if check['passed']:break
        if not check['truncated_tasks']:raise RuntimeError('Non-truncation pilot failure; inspect completion_check.json')
        for task in check['truncated_tasks']:
            candidates=[v for v in [128,512,1024] if v>limits[task]]
            if not candidates:raise RuntimeError('Pilot remains truncated at 1024 tokens')
            limits[task]=min(candidates)
    else:raise RuntimeError('Pilot did not pass')
    dump(ROOT/'pilot_passed.json',dict(max_new_by_task=limits,directory=str(directory),
        max_generated_tokens_by_task=check['max_generated_tokens_by_task'],responses=117))
    final=setup('evaluation',rows,limits)
    run_group(final,'evaluating',limits)
    status('summarizing',max_new_by_task=limits)
    if launch(final,'summarize',0,limits).wait()!=0:raise RuntimeError('Summary failed')
    status('complete',max_new_by_task=limits,questions=450)
except Exception:
    for p in workers:
        if p.poll() is None:p.terminate()
    status('failed')
    (ROOT/'failure.txt').write_text(traceback.format_exc())
    raise
