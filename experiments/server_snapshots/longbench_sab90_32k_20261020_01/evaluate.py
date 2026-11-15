"""Frozen-checkpoint, paired LongBench evaluation; no training."""
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/root/adablock_sab90_20261019_01')
sys.path.insert(0, str(OLD))
import core as c
import evaluation_base as base
from utils.longbench_eval import format_longbench_prompt, score_prediction

TASKS = ['qasper', 'multifieldqa_en', 'musique']
LIMITS = dict(qasper=128, multifieldqa_en=64, musique=32)
NAMES = ['full', 'S', 'A', 'B']


def prepare():
    from transformers import AutoTokenizer
    sys.path.insert(0, '/root/adablock')
    from scripts.build_strict_training_corpus import sampled_ngram_hashes
    assert not (ROOT/'selected.json').exists()
    blocked = set()
    questions = set()
    for r in json.loads((c.QA/'selected.json').read_text()):
        blocked.update(sampled_ngram_hashes(r['context'], 13, 5))
        questions.add(r['question'].strip().lower())
    for split in ['rank_train', 'budget_train', 'validation', 'test']:
        for line in (c.PILOT/f'{split}.jsonl').read_text().splitlines():
            blocked.update(sampled_ngram_hashes(json.loads(line)['text'], 13, 5))
    tok = AutoTokenizer.from_pretrained(c.run.MODEL, local_files_only=True)
    rows = []
    audit = {}
    for task in TASKS:
        source = Path('/root/adablock/data/longbench')/f'{task}.jsonl'
        eligible = []
        excluded = []
        for idx, line in enumerate(source.read_text().splitlines()):
            r = json.loads(line)
            overlap = len(set(sampled_ngram_hashes(r['context'], 13, 5)) & blocked)
            if r['input'].strip().lower() in questions or overlap >= 3:
                excluded.append(dict(index=idx, overlap=overlap))
                continue
            eligible.append((idx, r))
        assert len(eligible) >= 100, (task, len(eligible))
        selected = random.Random(20261020+TASKS.index(task)).sample(eligible, 100)
        for idx, r in selected:
            prompt = format_longbench_prompt(task, r)
            ids = tok.apply_chat_template([dict(role='user', content=prompt)],
                                         tokenize=True, add_generation_prompt=True)
            original = len(ids)
            if original > 32768:
                ids = ids[:16384]+ids[-16384:]
            rows.append(dict(task=task, source_index=idx, source_id=r.get('_id'),
                             answers=r['answers'], question=r['input'], input_ids=ids,
                             prompt_tokens=len(ids), original_tokens=original,
                             truncated=original > 32768,
                             input_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        audit[task] = dict(source_sha256=c.sha(source), eligible=len(eligible), excluded=excluded)
    previous = json.loads(Path('/root/adablock_longbench_sab90_20261020_01/selected.json').read_text())
    assert [(r['task'],r['source_index'],r['source_id']) for r in rows] == [(r['task'],r['source_index'],r['source_id']) for r in previous]
    c.dump(ROOT/'selected.json', rows)
    model = Path(c.run.MODEL)
    c.dump(ROOT/'manifest.json', dict(
        selected_sha256=c.sha(ROOT/'selected.json'), sampling=audit,
        checkpoints={n:c.sha(OLD/'branches'/n/'best_pair.pt') for n in NAMES[1:]},
        model=str(model),
        model_config_sha256=c.sha(model/'config.json'),
        tokenizer_config_sha256=c.sha(model/'tokenizer_config.json'),
        protocol_sha256=c.sha(ROOT/'protocol.json')))
    print('PREPARED', len(rows), flush=True)


@torch.inference_mode()
def evaluate(shard):
    torch.set_num_threads(4)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert c.sha(ROOT/'selected.json') == manifest['selected_sha256']
    for n in NAMES[1:]:
        assert c.sha(OLD/'branches'/n/'best_pair.pt') == manifest['checkpoints'][n]
    c.run.AdaptiveLlama = base.CostController
    model,tok,oldrank,oldbudget,eos,_ = c.run.load()
    del oldrank,oldbudget
    models = {n:c.load_pair(OLD/'branches'/n/'best_pair.pt','cuda:0')[:2] for n in NAMES[1:]}
    rows = json.loads((ROOT/'selected.json').read_text())
    # Probe the maximum input length, not just a short prompt.
    probe = max(rows, key=lambda r:r['prompt_tokens'])
    pairs,logits,_ = c.run.prefill(model,probe['input_ids'])
    digests = c.rt.build_prefix_digests(pairs)
    ref = c.run.decode(model,pairs,logits,'full',None,None,None,eos,max_new=6,ignore_eos=True)
    all_read = c.run.decode(model,pairs,logits,'all',None,None,None,eos,max_new=6,ignore_eos=True)
    assert ref['ids'] == all_read['ids']
    for ranker,budget in models.values():
        c.run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,
                     quantile=c.Q,max_new=4,ignore_eos=True)
    c.dump(ROOT/f'preflight_{shard}.json',dict(full_all_parity=True,
           prompt_tokens=probe['prompt_tokens'],ragged_error=c.rt.ragged_test()))
    del pairs,logits,digests
    with (ROOT/f'qa_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i % 4 != shard:
                continue
            pairs,logits,prefill_seconds = c.run.prefill(model,row['input_ids'])
            c.run.sync()
            start = time.perf_counter()
            digests = c.rt.build_prefix_digests(pairs)
            c.run.sync()
            index_seconds = time.perf_counter()-start
            offset = (i+i//4) % 4
            order = NAMES[offset:]+NAMES[:offset]
            record = {k:v for k,v in row.items() if k != 'input_ids'}
            record.update(evaluation_index=i, execution_order=order, methods={})
            for name in order:
                ranker,budget = models[name] if name != 'full' else (None,None)
                result = c.run.decode(model,pairs,logits,'full' if name=='full' else 'adaptive',
                                      ranker,budget,None if name=='full' else digests,eos,
                                      quantile=c.Q,max_new=LIMITS[row['task']],capture=True)
                controller = result.pop('_controller')
                if controller is not None:
                    cost = controller.costs
                    read = sum(r['read'] for r in cost)
                    padded = sum(r['padded'] for r in cost)
                    denom = sum(r['full'] for r in cost)
                    assert all(n==result['decode_steps'] for n in result['audit']['ranker_calls'])
                    assert all(n==result['decode_steps'] for n in result['audit']['budget_calls'])
                else:
                    denom = sum((row['prompt_tokens']+s+1)*32*8 for s in range(result['decode_steps']))
                    read = padded = denom
                result.update(logical_slots=read,padded_slots=padded,full_slots=denom,
                              logical_fraction=read/denom if denom else None,
                              padded_fraction=padded/denom if denom else None,
                              prefill_seconds=prefill_seconds,
                              index_seconds=index_seconds if name!='full' else 0)
                result['prediction'] = tok.decode(result['ids'],skip_special_tokens=True).strip()
                result['f1'] = score_prediction(row['task'],result['prediction'],row['answers'])
                result['e2e_seconds'] = prefill_seconds+result['decode_seconds']+result['index_seconds']
                result.pop('steps')
                record['methods'][name] = result
                del controller
            journal.write(json.dumps(record,ensure_ascii=False)+'\n')
            journal.flush()
            print(json.dumps(dict(index=i,task=row['task'],
                  f1={n:r['f1'] for n,r in record['methods'].items()})),flush=True)
            del pairs,logits,digests,record
            gc.collect()
            torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


def summarize():
    rows = [json.loads(line) for p in sorted(ROOT.glob('qa_[0-3].jsonl'))
            for line in p.read_text().splitlines()]
    assert len(rows)==300 and len({r['evaluation_index'] for r in rows})==300
    metrics = ['f1','logical_fraction','padded_fraction','decode_seconds','e2e_seconds',
               'prefill_seconds','index_seconds','decode_steps','hit_generation_limit']
    def report(sub):
        means = {n:{k:float(np.mean([r['methods'][n][k] for r in sub
                                    if r['methods'][n][k] is not None]))
                    for k in metrics} for n in NAMES}
        paired = {}
        for left,right in [('B','A'),('S','full'),('A','full'),('B','full')]:
            delta = np.array([r['methods'][left]['f1']-r['methods'][right]['f1'] for r in sub])
            rng = np.random.default_rng(20261020)
            boot = delta[rng.integers(0,len(sub),size=(10000,len(sub)))].mean(1)
            paired[left+'_minus_'+right] = dict(delta=float(delta.mean()),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()))
        return dict(n=len(sub),means=means,paired=paired,
                    truncated=sum(r['truncated'] for r in sub),
                    input_length_quantiles=np.quantile([r['prompt_tokens'] for r in sub],[0,.25,.5,.75,1]).tolist())
    out = {}
    for task in TASKS:
        sub = [r for r in rows if r['task']==task]
        assert len(sub)==100
        out[task] = report(sub)
        out[task]['length_groups'] = {}
        for lo,hi in [(0,8192),(8192,16384),(16384,32769)]:
            group = [r for r in sub if lo<=r['prompt_tokens']<hi]
            if group:
                out[task]['length_groups'][f'{lo}-{hi}'] = report(group)
    c.dump(ROOT/'comparison.json',out)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='evaluate': evaluate(int(sys.argv[2]))
    elif sys.argv[1]=='summarize': summarize()
    else: raise ValueError(sys.argv[1])
