"""Oracle budget granularity only; frozen ranker, original Hotpot50 protocol."""
import json,sys,gc
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
QA=Path('/root/adablock/results/llama_qa_budget_20261015_01')
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
sys.path.insert(0,str(QA));import engine as e
run,rt=e.run,e.rt
ORIGINAL=rt.selected_attention
RHO=.90

def budgets(curve):
    n=len(curve)-1
    exact=int(torch.nonzero(curve>=RHO-1e-7)[0])
    bucket=next(n if b=='ALL' else min(n,b) for b in rt.BUDGETS if (n if b=='ALL' else min(n,b))>=exact)
    assert exact<=bucket<=n
    assert float(curve[exact])>=RHO-1e-7
    if exact:assert float(curve[exact-1])<RHO-1e-7
    return exact,bucket

class Oracle(rt.AdaptiveLlama):
    replay_plan=None
    def choose(self,layer,query,scale):
        h=self.model.config.num_key_value_heads;group=self.model.config.num_attention_heads//h
        count=(self.position+15)//16
        sink,local,middle=rt.fixed_and_middle_blocks(count,2,32,count-1)
        ids=torch.tensor(sorted(middle),device=query.device,dtype=torch.long)
        digest=self.digests[layer];grouped=query[0,:,0].float().view(h,group,-1)*scale
        scenes=[dict(query=grouped[head],mean=digest['mean'][head,ids],basis=digest['basis'][head,ids],eigenvalues=digest['eigenvalues'][head,ids],counts=digest['counts'][ids],block_ids=ids,block_size=16,token_position=self.position,layer_id=layer,kv_head_id=head) for head in range(h)]
        scores=rt.forward_packed(self.ranker,rt.pack_scenes(scenes,query.device)).view(h,-1).float().cpu()
        self.audit['ranker_calls'][layer]+=1
        self.probe=(layer,ids.cpu(),scores.argsort(descending=True,stable=True),sink|local)
        # The hook receives the just-appended current K/V and computes the actual selection there.
        return torch.zeros((h,1),dtype=torch.long,device=query.device),torch.ones((h,1),dtype=torch.bool,device=query.device)

    def attention(self,query,keys,values,unused_indices,unused_valid,scale):
        layer,ids,orders,fixed_ids=self.probe;h=keys.shape[1];n=len(ids)
        if self.replay_plan is None:
            q=query[0,:,0].float().reshape(h,-1,query.shape[-1])
            mass=torch.einsum('hgd,htd->hgt',q,keys[0].float()).mul(scale).softmax(-1).mean(1)
            assert torch.allclose(mass.sum(-1),torch.ones(h,device=mass.device),atol=2e-6)
            blocks=torch.nn.functional.pad(mass,(0,(-mass.shape[-1])%16)).reshape(h,-1,16).sum(-1).cpu()
            candidate=blocks[:,ids];fixed=1-candidate.sum(-1)
            exact=[];bucket=[];coverage=[]
            for head in range(h):
                curve=torch.cat([fixed[head,None],fixed[head]+candidate[head,orders[head]].cumsum(0)]).clamp(max=1)
                a,b=budgets(curve);exact.append(a);bucket.append(b)
                coverage.append(float(curve[a if self.mode=='oracle_exact' else b]))
            takes=exact if self.mode=='oracle_exact' else bucket
        else:
            expected=self.replay_plan[len(self.rows)]
            assert expected['position']==self.position and expected['layer']==layer
            takes=expected['selected'];exact=expected['exact'];bucket=expected['bucket'];coverage=expected['coverage']
        selected=[];middle_ids=ids.tolist();order_ids=orders.tolist()
        for head in range(h):
            blocks=fixed_ids|{middle_ids[j] for j in order_ids[head][:takes[head]]}
            tokens=[t for b in sorted(blocks) for t in range(b*16,min((b+1)*16,self.position))]+[self.position]
            assert len(tokens)==len(set(tokens)) and max(tokens)==self.position
            selected.append(tokens)
        indices,valid=rt.pad_indices(selected,query.device)
        if not bool(valid.all()):self.audit['ragged_attention_calls']+=1
        lengths=[len(t) for t in selected];fixed_tokens=self.position+1-n*16
        assert lengths==[fixed_tokens+16*t for t in takes]
        self.step_rows.append(dict(layer=layer,selected_tokens=lengths,middle_blocks=takes,classes=[-1]*h))
        self.rows.append(dict(position=self.position,layer=layer,candidates=n,exact=exact,bucket=bucket,selected=takes,coverage=coverage,selected_tokens=lengths,padded_tokens=[max(lengths)]*h,fixed_tokens=fixed_tokens))
        return ORIGINAL(query,keys,values,indices,valid,scale)

    def install(self):
        self.rows=[];super().install();rt.selected_attention=self.attention
    def uninstall(self):
        rt.selected_attention=ORIGINAL;super().uninstall()

@torch.inference_mode()
def main(shard):
    # Boundary checks are meaningful here: zero, exact boundary, and ALL saturation.
    assert budgets(torch.tensor([.96,.98,1.]))==(0,0)
    assert budgets(torch.tensor([.1,.2,.3,.4,.90,.96,1.]))==(4,4)
    assert budgets(torch.tensor([.1,.2,.3,.4,.5,.90,1.]))==(5,6)
    model,tok,ranker,budget,eos,_=run.load()
    rows=sorted(json.loads((BASE/'selected.json').read_text()),key=lambda r:(-r['prompt_tokens'],r['source_index']))
    originals={x['source_index']:x for p in BASE.glob('worker_*.jsonl') for l in p.read_text().splitlines() if (x:=json.loads(l))}
    with (ROOT/f'worker_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            source=row['source_index'];assert row['input_ids_sha256']==originals[source]['input_ids_sha256']
            pairs,logits,_=run.prefill(model,row['input_ids']);digests=rt.build_prefix_digests(pairs)
            assert int(logits.argmax())==originals[source]['methods']['full']['ids'][0]
            record=dict(source_index=source,input_ids_sha256=row['input_ids_sha256'],answers=row['answers'],prompt_tokens=row['prompt_tokens'],truncated=row['truncated'],methods={})
            for mode in ['oracle_exact','oracle_bucket']:
                run.AdaptiveLlama=Oracle
                try:out=run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,capture=True)
                finally:run.AdaptiveLlama=rt.AdaptiveLlama
                controller=out.pop('_controller');out['diagnostics']=controller.rows
                assert len(controller.rows)==out['decode_steps']*32
                assert all(c==0 for c in controller.audit['budget_calls'])
                if i<4 or i>=len(rows)-4:
                    Oracle.replay_plan=controller.rows;run.AdaptiveLlama=Oracle
                    try:replay=run.decode(model,pairs,logits,mode,ranker,budget,digests,eos)
                    finally:Oracle.replay_plan=None;run.AdaptiveLlama=rt.AdaptiveLlama
                    assert out['ids']==replay['ids'],(source,mode)
                    out['teacher_free_replay_parity']=True
                out['prediction']=tok.decode(out['ids'],skip_special_tokens=True).strip()
                out['f1']=100*max(run.token_f1(out['prediction'],a) for a in row['answers'])
                record['methods'][mode]=out
                print(json.dumps(dict(source_index=source,mode=mode,f1=out['f1'],prediction=out['prediction'],blocks=out['mean_middle_blocks'])),flush=True)
                del controller
            journal.write(json.dumps(record)+'\n');journal.flush()
            del record,out,pairs,logits,digests;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)

if __name__=='__main__':main(int(sys.argv[1]))
