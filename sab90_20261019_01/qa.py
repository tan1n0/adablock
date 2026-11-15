"""Same frozen 32 QA, native full baseline and three independently trained pairs."""
from collections import defaultdict
import gc
import json
import sys
import time
import numpy as np
import torch
import core as c
import evaluation_base as base


@torch.inference_mode()
def evaluate(shard):
    torch.set_num_threads(4)
    c.run.AdaptiveLlama=base.CostController
    model,tok,oldrank,oldbudget,eos,_=c.run.load()
    del oldrank,oldbudget
    models={name:c.load_pair(c.ROOT/'branches'/name/'best_pair.pt','cuda:0')[:2] for name in ['S','A','B']}
    for ranker,budget in models.values():ranker.requires_grad_(False);budget.requires_grad_(False)
    rows=json.loads((c.ROOT/'qa_selected.json').read_text())
    previous={r['source_index']:r for f in c.PREVIOUS.glob('qa_*.jsonl') for line in f.read_text().splitlines() if (r:=json.loads(line))}
    c.run.prefill(model,rows[0]['input_ids'][:128])
    pairs,logits,_=c.run.prefill(model,rows[0]['input_ids'][:1024])
    digests=c.rt.build_prefix_digests(pairs)
    ref=c.run.decode(model,pairs,logits,'full',None,None,None,eos,max_new=6,ignore_eos=True)
    all_read=c.run.decode(model,pairs,logits,'all',None,None,None,eos,max_new=6,ignore_eos=True)
    assert ref['ids']==all_read['ids']
    for ranker,budget in models.values():
        c.run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=c.Q,max_new=4,ignore_eos=True)
    c.dump(c.ROOT/f'qa_preflight_{shard}.json',dict(full_all_parity=True,ragged_error=c.rt.ragged_test()))
    del pairs,logits,digests
    with (c.ROOT/f'qa_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            pairs,logits,prefill_seconds=c.run.prefill(model,row['input_ids'])
            c.run.sync();start=time.perf_counter();digests=c.rt.build_prefix_digests(pairs)
            c.run.sync();index_seconds=time.perf_counter()-start
            names=['full','S','A','B'];offset=(i+i//4)%4;names=names[offset:]+names[:offset]
            record=dict(source_index=i,source_id=row['source_id'],input_ids_sha256=row['input_ids_sha256'],
                        prompt_tokens=row['prompt_tokens'],prefill_seconds=prefill_seconds,index_seconds=index_seconds,
                        execution_order=names,methods={})
            for name in names:
                ranker,budget=models[name] if name!='full' else (None,None)
                result=c.run.decode(model,pairs,logits,'full' if name=='full' else 'adaptive',ranker,budget,
                                    None if name=='full' else digests,eos,quantile=c.Q,capture=True)
                controller=result.pop('_controller')
                if controller:
                    cost=controller.costs
                    read=sum(r['read'] for r in cost);padded=sum(r['padded'] for r in cost);denom=sum(r['full'] for r in cost)
                    assert all(n==result['decode_steps'] for n in result['audit']['ranker_calls'])
                    assert all(n==result['decode_steps'] for n in result['audit']['budget_calls'])
                else:
                    denom=sum((len(row['input_ids'])+s+1)*32*8 for s in range(result['decode_steps']))
                    read=padded=denom
                    assert result['ids']==previous[i]['methods']['full']['ids'],i
                result.update(logical_slots=read,padded_slots=padded,full_slots=denom,
                              logical_fraction=read/max(1,denom),padded_fraction=padded/max(1,denom))
                result['prediction']=tok.decode(result['ids'],skip_special_tokens=True).strip()
                result['f1']=100*max(c.run.token_f1(result['prediction'],answer) for answer in row['answers'])
                result['e2e_seconds']=prefill_seconds+result['decode_seconds']+(index_seconds if name!='full' else 0)
                result.pop('steps');record['methods'][name]=result
                del controller
            journal.write(json.dumps(record,ensure_ascii=False)+'\n');journal.flush()
            print(json.dumps(dict(source_index=i,f1={name:r['f1'] for name,r in record['methods'].items()})),flush=True)
            if i==shard:
                diagnostic={name:base.profile(model,pairs,logits,ranker,budget,digests,eos)
                            for name,(ranker,budget) in models.items()}
                c.dump(c.ROOT/f'profile_{shard}.json',diagnostic)
            del pairs,logits,digests,record;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


def summarize():
    rows=[json.loads(line) for path in sorted(c.ROOT.glob('qa_[0-3].jsonl')) for line in path.read_text().splitlines()]
    assert len(rows)==32 and len({r['source_index'] for r in rows})==32
    rows.sort(key=lambda r:r['source_index'])
    names=['full','S','A','B']
    metrics=['f1','logical_fraction','padded_fraction','logical_slots','padded_slots','decode_seconds',
             'decode_steps','seconds_per_decode_step','e2e_seconds','mean_middle_blocks']
    summary={name:{metric:float(np.mean([r['methods'][name][metric] for r in rows])) for metric in metrics} for name in names}
    paired={};rng=np.random.default_rng(c.SEED)
    for left,right in [('B','A'),('B','S'),('A','S'),('S','full'),('A','full'),('B','full')]:
        delta=np.array([r['methods'][left]['f1']-r['methods'][right]['f1'] for r in rows])
        bootstrap=delta[rng.integers(0,32,size=(10000,32))].mean(1)
        paired[left+'_minus_'+right]=dict(f1_delta=float(delta.mean()),bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                                        wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),
                                        relative_read_fraction_reduction=1-summary[left]['logical_fraction']/summary[right]['logical_fraction'],
                                        relative_padded_fraction_reduction=1-summary[left]['padded_fraction']/summary[right]['padded_fraction'])
    training={name:json.loads((c.ROOT/'branches'/name/'validation.json').read_text()) for name in ['S','A','B']}
    report=dict(n=32,summary=summary,paired=paired,training=training,
                protocol_sha256=c.sha(c.ROOT/'protocol.json'),qa_selection_sha256=c.sha(c.ROOT/'qa_selected.json'),
                limitations=['single seed and same previously inspected development QA, no final generalization claim',
                             'QA prompts 2.1K..3.1K; not a long-context benchmark',
                             'same update counts and capacity; actual pair counts and refresh costs are reported',
                             'slots are not memory traffic; independent generation may differ in length'])
    c.dump(c.ROOT/'comparison.json',report)
    c.dump(c.ROOT/'status.json',dict(stage='complete',questions=32))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='evaluate':evaluate(int(sys.argv[2]))
    else:summarize()
