import os, sys, json, time, gc, hashlib, subprocess, statistics
from pathlib import Path
import experiment as e
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_official_prompt_20261021_01/evaluation')

def dump(name, value):
    path=ROOT/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(e.clean(value),indent=2))
    tmp.replace(path)

@torch.inference_mode()
def preflight():
    model,tok,models,eos=e.load()
    rows=json.loads((ROOT/'selected.json').read_text())
    p,l,_=e.prefill(model,rows[0]['input_ids'][:1024])
    d,_=e.build(p,'gpu_digest')
    report={}
    e.baseline.compact_digest=e.fast_digest.compact_digest
    for name,(rank,budget) in models.items():
        e.c.run.DynamicCache=e.BufferCache
        e.c.run.AdaptiveLlama=e.baseline.Controller
        a=e.c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=5,ignore_eos=True,capture=True,quantile=.95)
        e.c.run.AdaptiveLlama=e.gpu.Controller
        b=e.c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=5,ignore_eos=True,capture=True,quantile=.95)
        assert a['ids']==b['ids'],name
        assert abs(float(a['mean_read_fraction'])-float(b['mean_read_fraction']))<1e-6
        report[name]={'tokens_equal':True,'read_fraction':float(b['mean_read_fraction'])}
        del a,b
    del p,l,d
    gc.collect();torch.cuda.empty_cache()
    longest=max(rows,key=lambda r:len(r['input_ids']))
    p,l,_=e.prefill(model,longest['input_ids'])
    a=e.c.run.decode(model,p,l,'full',None,None,None,eos,max_new=3,ignore_eos=True)
    b=e.c.run.decode(model,p,l,'all',None,None,None,eos,max_new=3,ignore_eos=True)
    assert a['ids']==b['ids'],'Full/all mismatch'
    report['long_full_all_parity']=True
    report['long_prompt_tokens']=len(longest['input_ids'])
    dump('preflight.json',report)

@torch.inference_mode()
def worker(shard):
    model,tok,models,eos=e.load()
    e.c.run.DynamicCache=e.BufferCache
    e.c.run.AdaptiveLlama=e.gpu.Controller
    e.baseline.compact_digest=e.fast_digest.compact_digest
    rows=json.loads((ROOT/'selected.json').read_text())
    for i,row in enumerate(rows):
        if i%4!=shard:continue
        path=ROOT/f'partial_{i}.json'
        record=json.loads(path.read_text()) if path.exists() else dict(index=i,task=row['task'],length=row['length'],answers=row['answers'],methods={})
        if len(record['methods'])==4:continue
        p,l,prefill=e.prefill(model,row['input_ids'])
        d,index_time=e.build(p,'gpu_digest')
        order=['full','S','A','B'];order=order[i%4:]+order[:i%4]
        for name in order:
            if name in record['methods']:continue
            rank,budget=models[name] if name!='full' else (None,None)
            result=e.c.run.decode(model,p,l,'full' if name=='full' else 'adaptive',rank,budget,
                d if name!='full' else None,eos,max_new=row['max_new'],capture=True,quantile=.95)
            obj=result.pop('_controller')
            if obj is not None:
                denom=sum(v['full'] for v in obj.costs)
                logical=float(sum(v['read'] for v in obj.costs))/denom if denom else None
                padded=float(sum(v['padded'] for v in obj.costs))/denom if denom else None
                # Validate the last step's per-head counts against raw budget classes.
                for slot in obj.step_rows:
                    n=slot['candidate_blocks']
                    choices=torch.tensor([min(n,k) for k in e.baseline.BUDGETS[:-1]]+[n],device='cuda')
                    assert torch.equal(slot['middle_blocks'],choices[slot['classes']])
            else:logical=padded=1.
            prediction=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
            result.pop('steps')
            result.update(prediction=prediction,score=e.official.metric_for(row['task'])([prediction],[row['answers']]),
                logical_fraction=logical,padded_fraction=padded,prefill_seconds=prefill,index_seconds=index_time if name!='full' else 0,
                e2e_seconds=prefill+result['decode_seconds']+(index_time if name!='full' else 0),raw_budget_counts_verified=name!='full')
            record['methods'][name]=e.clean(result)
            dump(path.name,record)
            print(json.dumps(dict(index=i,task=row['task'],length=row['length'],method=name,score=result['score'],read=logical)),flush=True)
            del obj,result
        del p,l,d
        gc.collect();torch.cuda.empty_cache()

def summarize():
    records=[json.loads(p.read_text()) for p in ROOT.glob('partial_*.json')]
    report={}
    for task,length in sorted({(r['task'],r['length']) for r in records}):
        group=[r for r in records if (r['task'],r['length'])==(task,length) and len(r['methods'])==4]
        if not group:continue
        report[f'{task}_{length}']={'n':len(group),'methods':{n:{key:statistics.mean(r['methods'][n][key] for r in group if r['methods'][n][key] is not None)
            for key in ['score','logical_fraction','padded_fraction','mean_selected_tokens','decode_seconds','e2e_seconds','hit_generation_limit']} for n in ['full','S','A','B']}}
    dump('comparison.json',report)
    return len(records)==450 and all(len(r['methods'])==4 for r in records)

def launch():
    assert not (ROOT/'status.json').exists()
    (ROOT/'selected.json').write_bytes((SOURCE/'selected.json').read_bytes())
    protected=[SOURCE/'selected.json',SOURCE/'comparison.json']+[e.OLD/'branches'/n/'best_pair.pt' for n in ['S','A','B']]
    hashes={str(p):e.c.sha(p) for p in protected}
    dump('protocol.json',dict(n=450,tasks=['niah_single_1','niah_multivalue','vt'],lengths=[8192,16384,32768],
        coverage_target=.9,quantile=.95,policy='raw predicted per-head counts, no total quota redistribution',
        summary='FP32 Gram, no SVD guard',same_inputs_and_caps=True,protected_sha256=hashes,
        source_sha256={p.name:e.c.sha(p) for p in ROOT.glob('*.py')}))
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false')
    dump('status.json',dict(stage='preflight'))
    with (ROOT/'preflight.log').open('w') as log:
        subprocess.run([sys.executable,'-u',__file__,'preflight'],env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[]
    for shard in range(4):
        with (ROOT/f'worker_{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-u',__file__,'worker',str(shard)],env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT))
    dump('status.json',dict(stage='running',pids=[j.pid for j in jobs]))
    while any(j.poll() is None for j in jobs):
        if any(j.poll() not in [None,0] for j in jobs):
            for j in jobs:
                if j.poll() is None:j.terminate()
            raise RuntimeError('worker failed; inspect worker logs')
        summarize();time.sleep(30)
    assert summarize(),'incomplete results'
    assert all(e.c.sha(Path(p))==h for p,h in hashes.items())
    dump('status.json',dict(stage='complete',old_results_unchanged=True))

if __name__=='__main__':
    try:
        if sys.argv[1]=='worker':worker(int(sys.argv[2]))
        elif sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='summarize':summarize()
        else:launch()
    except Exception as exc:
        if sys.argv[1]=='launch':dump('status.json',dict(stage='failed',error=str(exc)))
        raise
