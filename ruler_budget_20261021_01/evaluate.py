import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
OLD=Path('/root/adablock_sab90_20261019_01')
sys.path.insert(0,str(OLD))
import core as c
import controller
from cache_buffer import BufferCache, reset_registry

RATIOS=[.10,.15,.25,.40]
VARIANTS=[(n,r) for r in RATIOS for n in ['S','A','B']]

@torch.inference_mode()
def prefill(model,ids):
    # Bounded workspace, full causal attention and the complete KV cache retained.
    reset_registry()
    cache=BufferCache(len(ids)+256)
    c.run.sync();start=time.perf_counter()
    for offset in range(0,len(ids),1024):
        output=model.model(input_ids=torch.tensor([ids[offset:offset+1024]],device='cuda:0'),
                           past_key_values=cache,use_cache=True)
        cache=output.past_key_values
    logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
    pairs=tuple(cache.to_legacy_cache())
    c.run.sync()
    return pairs,logits,time.perf_counter()-start

@torch.inference_mode()
def budget_tests():
    import random
    rng=random.Random(20261021)
    for _ in range(1000):
        cap=rng.randrange(1,2049);target=rng.randrange(8*cap+1)
        result=controller.allocate([rng.randrange(cap+1) for _ in range(8)],target,cap)
        assert sum(result)==target and all(0<=v<=cap for v in result)
    assert controller.allocate([0]*8,8,10)==[1]*8
    assert controller.allocate([1]*8,80,10)==[10]*8
    # Branches may replace their suffix but must never mutate the shared prompt.
    from cache_buffer import append_pair
    cache=BufferCache(12)
    k=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)
    v=k+100
    cache.update(k,v,0)
    prefix=tuple(cache.to_legacy_cache())
    for value in [7.,9.,11.]:
        branch=BufferCache.from_legacy_cache(prefix)
        new=torch.full((1,2,1,4),value)
        outk,outv=branch.update(new,new,0)
        assert torch.equal(outk[...,:3,:],k) and torch.equal(outv[...,:3,:],v)
        assert torch.equal(outk[...,3:,:],new)
        ak,av=append_pair(*prefix[0],new,new)
        assert torch.equal(ak,outk) and torch.equal(av,outv)
        assert torch.equal(prefix[0][0],k) and torch.equal(prefix[0][1],v)

@torch.inference_mode()
def evaluate(shard):
    budget_tests()
    torch.set_num_threads(4)
    rows=json.loads((ROOT/'selected.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert c.sha(ROOT/'selected.json')==manifest['selected_sha256']
    model,tok,oldrank,oldbudget,eos,_=c.run.load()
    del oldrank,oldbudget
    models={n:c.load_pair(OLD/'branches'/n/'best_pair.pt','cuda:0')[:2] for n in ['S','A','B']}
    for n in models:assert c.sha(OLD/'branches'/n/'best_pair.pt')==manifest['checkpoints'][n]
    assert c.state_digest(models['S'][0])==c.state_digest(models['A'][0])
    c.run.AdaptiveLlama=controller.Controller
    c.run.DynamicCache=BufferCache
    short=rows[0]['input_ids'][:2048]
    p,l,_=prefill(model,short);p2,l2,_=c.run.prefill(model,short)
    assert int(l.argmax())==int(l2.argmax())
    err=float((l-l2).abs().max())
    x=c.run.decode(model,p,l,'full',None,None,None,eos,max_new=8,ignore_eos=True)
    y=c.run.decode(model,p2,l2,'full',None,None,None,eos,max_new=8,ignore_eos=True)
    assert x['ids']==y['ids'], 'chunked prefill changed short probe generation'
    del p,l,p2,l2,x,y
    gc.collect();torch.cuda.empty_cache()
    probe=max(rows,key=lambda r:len(r['input_ids']))
    p,l,_=prefill(model,probe['input_ids'])
    print("LONG_PREFILL_COMPLETE",flush=True)
    assert isinstance(BufferCache.from_legacy_cache(p),BufferCache)
    x=c.run.decode(model,p,l,'full',None,None,None,eos,max_new=4,ignore_eos=True)
    y=c.run.decode(model,p,l,'all',None,None,None,eos,max_new=4,ignore_eos=True)
    assert x['ids']==y['ids']
    print('LONG_FULL_ALL_PARITY',flush=True)
    d=c.rt.build_prefix_digests(p)
    print("LONG_DIGEST_COMPLETE",flush=True)
    for name in ['S','A','B']:
        controller.Controller.ratio=.4
        rank,budget=models[name]
        result=c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=3,ignore_eos=True,quantile=.95,capture=True)
        assert result['decode_steps']==2
        del result
    c.dump(ROOT/f'preflight_{shard}.json',dict(short_chunk_logits_maxerr=err,short_generated_parity=True,
        longest_prompt_tokens=len(probe['input_ids']),all_read_parity=True,budget_allocation_tests=True,
        peak_memory_bytes=torch.cuda.max_memory_allocated()))
    del p,l,d,x,y;gc.collect();torch.cuda.empty_cache()
    journal=ROOT/f'qa_{shard}.jsonl'
    completed=set()
    if journal.exists():
        completed={json.loads(line)['index'] for line in journal.read_text().splitlines()}
    with journal.open('a') as out:
        for i,row in enumerate(rows):
            if i%4!=shard or i in completed:continue
            p,l,prefill_time=prefill(model,row['input_ids'])
            c.run.sync();start=time.perf_counter();d=c.rt.build_prefix_digests(p);c.run.sync()
            index_time=time.perf_counter()-start
            record={k:v for k,v in row.items() if k!='input_ids'}
            record.update(index=i,methods={})
            order=[('full',1.)]+VARIANTS
            shift=(i+i//4)%len(order);order=order[shift:]+order[:shift]
            for name,ratio in order:
                rank,budget=models[name] if name!='full' else (None,None)
                controller.Controller.ratio=ratio
                result=c.run.decode(model,p,l,'full' if name=='full' else 'adaptive',rank,budget,
                    None if name=='full' else d,eos,max_new=row['max_new'],capture=True,quantile=.95)
                obj=result.pop('_controller')
                if obj is not None:
                    costs=obj.costs
                    denom=sum(v['full'] for v in costs);read=sum(v['read'] for v in costs);padded=sum(v['padded'] for v in costs)
                    # All complete layer/step slots must obey the same quota to one block.
                    for v in costs:
                        assert v['read']<=ratio*v['full']+1e-6
                        assert ratio*v['full']-v['read']<16+1e-6
                else:
                    denom=sum((row['prompt_tokens']+step+1)*32*8 for step in range(result['decode_steps']))
                    read=padded=denom
                prediction=tok.decode(result['ids'],skip_special_tokens=True).strip()
                hit=[answer.lower() in prediction.lower() for answer in row['answers']]
                result.update(prediction=prediction,score=100*sum(hit)/len(hit),all_hit=100*all(hit),
                    ratio=ratio,logical_slots=read,padded_slots=padded,full_slots=denom,
                    logical_fraction=read/denom if denom else None,padded_fraction=padded/denom if denom else None,
                    prefill_seconds=prefill_time,index_seconds=index_time if name!='full' else 0,
                    e2e_seconds=prefill_time+result['decode_seconds']+(index_time if name!='full' else 0))
                result.pop('steps');record['methods'][name if name=='full' else f'{name}_{ratio:.2f}']=result
                del obj
            out.write(json.dumps(record)+'\n');out.flush()
            print(json.dumps(dict(index=i,task=row['task'],length=row['length'],full=record['methods']['full']['score'])),flush=True)
            del p,l,d,record;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)

def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import importlib.util
    spec=importlib.util.spec_from_file_location('ruler_metric',ROOT/'RULER/scripts/eval/synthetic/constants.py')
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    rows=[json.loads(line) for f in sorted(ROOT.glob('qa_[0-3].jsonl')) for line in f.read_text().splitlines()]
    assert len(rows)==len({r['index'] for r in rows})==450
    report={}
    tasks=['niah_single_1','niah_multivalue','vt'];lengths=[8192,16384,32768]
    for task in tasks:
        for length in lengths:
            sub=sorted([r for r in rows if r['task']==task and r['length']==length],key=lambda r:r['sample_index'])
            assert len(sub)==50
            cell={}
            for name in sub[0]['methods']:
                means={k:float(np.mean([r['methods'][name][k] for r in sub if r['methods'][name][k] is not None]))
                    for k in ['score','all_hit','logical_fraction','padded_fraction','decode_seconds','e2e_seconds','hit_generation_limit']}
                official=metric.string_match_all([r['methods'][name]['prediction'] for r in sub],[r['answers'] for r in sub])
                assert abs(official-means['score'])<.006
                means['official_score']=official;cell[name]=means
            paired={}
            for ratio in RATIOS:
                delta=np.array([r['methods'][f'B_{ratio:.2f}']['score']-r['methods'][f'A_{ratio:.2f}']['score'] for r in sub])
                rng=np.random.default_rng(20261021)
                boot=delta[rng.integers(0,50,size=(10000,50))].mean(1)
                paired[str(ratio)]=dict(B_minus_A=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
            report[f'{task}_{length}']=dict(n=50,means=cell,paired=paired)
    c.dump(ROOT/'comparison.json',report)
    for cost,quality in [(c,q) for c in ['logical_fraction','padded_fraction'] for q in ['score','all_hit']]:
        fig,axes=plt.subplots(3,3,figsize=(14,11),sharey=True)
        for ti,task in enumerate(tasks):
            for li,length in enumerate(lengths):
                ax=axes[ti,li];cell=report[f'{task}_{length}']['means']
                for n in ['S','A','B']:
                    ax.plot([100*cell[f'{n}_{r:.2f}'][cost] for r in RATIOS],
                            [cell[f'{n}_{r:.2f}'][quality] for r in RATIOS],marker='o',label=n)
                ax.axhline(cell['full'][quality],color='black',linestyle='--',label='Full')
                ax.set_title(f'{task} / {length//1024}K');ax.set_ylim(-2,102);ax.grid(alpha=.25)
                ax.set_xlabel(cost.replace('_',' ')+' (%)');ax.set_ylabel('RULER answer recall (%)' if quality=='score' else 'All-required-answers hit (%)')
        axes[0,0].legend();fig.suptitle('Fixed total useful-slot quota; 50 examples per task/length')
        fig.tight_layout();fig.savefig(ROOT/f'{quality}_vs_{cost}.png',dpi=180);fig.savefig(ROOT/f'{quality}_vs_{cost}.pdf');plt.close(fig)
    print('SUMMARIZED',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='evaluate':evaluate(int(sys.argv[2]))
    elif sys.argv[1]=='summarize':summarize()
    elif sys.argv[1]=='test':budget_tests();print('PASS')
