"""Frozen-model Hotpot50 diagnostics only. No training/calibration or new data."""
import json,sys,hashlib,gc
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
QA=Path('/root/adablock/results/llama_qa_budget_20261015_01')
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
OLD_DIAG=Path('/root/adablock/results/llama_hotpot50_fixed112_diagnostics_20261015_01')
sys.path.insert(0,str(QA));import engine as e
run,rt=e.run,e.rt
ORIGINAL=rt.selected_attention
CASES={165,181,121,16,160}
def read(root,pattern):return {x['source_index']:x for p in root.glob(pattern) for l in p.read_text().splitlines() if (x:=json.loads(l))}

class Diagnostic(rt.AdaptiveLlama):
    def choose(self,layer,query,scale):
        mode=self.mode
        if mode in ['repair95','floor112']:self.mode='adaptive'
        indices,valid=super().choose(layer,query,scale);self.mode=mode
        h=self.model.config.num_key_value_heads;group=self.model.config.num_attention_heads//h
        sink,local,middle=rt.fixed_and_middle_blocks((self.position+15)//16,2,32,(self.position+15)//16-1)
        ids=torch.tensor(sorted(middle),device=query.device,dtype=torch.long);digest=self.digests[layer]
        grouped=query[0,:,0].float().view(h,group,-1)*scale
        scenes=[dict(query=grouped[head],mean=digest['mean'][head,ids],basis=digest['basis'][head,ids],eigenvalues=digest['eigenvalues'][head,ids],counts=digest['counts'][ids],block_ids=ids,block_size=16,token_position=self.position,layer_id=layer,kv_head_id=head) for head in range(h)]
        scores=rt.forward_packed(self.ranker,rt.pack_scenes(scenes,query.device)).view(h,-1).float().cpu()
        features=torch.stack([rt.online_features(dict(s,query=s['query'].cpu()),scores[i]) for i,s in enumerate(scenes)])
        classes=rt.quantile_actions(self.budget(features.to(query.device)).softmax(-1).cpu(),self.quantile).tolist()
        if mode!='fixed112':assert classes==self.step_rows[-1]['classes']
        self.probe=(layer,ids,scores,classes,sink|local)
        return indices,valid

    def attention(self,query,keys,values,indices,valid,scale):
        layer,ids,scores,classes,fixed_ids=self.probe;h=keys.shape[1];n=len(ids)
        q=query[0,:,0].float().reshape(h,-1,query.shape[-1])
        mass=torch.einsum('hgd,htd->hgt',q,keys[0].float()).mul(scale).softmax(-1).mean(1)
        assert torch.allclose(mass.sum(-1),torch.ones(h,device=mass.device),atol=2e-6)
        blocks=torch.nn.functional.pad(mass,(0,(-mass.shape[-1])%16)).reshape(h,-1,16).sum(-1)
        candidate=blocks[:,ids].cpu();fixed=1-candidate.sum(-1);orders=scores.argsort(descending=True,stable=True)
        new_selection=[];takes=[]
        for head in range(h):
            curve=torch.cat([fixed[head,None],fixed[head]+candidate[head,orders[head]].cumsum(0)]).clamp(max=1)
            ideal=torch.cat([fixed[head,None],fixed[head]+candidate[head].sort(descending=True).values.cumsum(0)])
            required=int(torch.nonzero(curve>=.95-1e-7)[0]);oracle=int(torch.nonzero(ideal>=.95-1e-7)[0])
            bucket=next(n if b=='ALL' else min(n,b) for b in rt.BUDGETS if (n if b=='ALL' else min(n,b))>=required)
            dynamic=n if rt.BUDGETS[classes[head]]=='ALL' else min(n,rt.BUDGETS[classes[head]])
            take=self.step_rows[-1]['middle_blocks'][head]
            if self.mode=='repair95':take=max(take,bucket)
            if self.mode=='floor112':take=max(take,min(112,n))
            takes.append(take)
            chosen=fixed_ids|{int(ids[j]) for j in orders[head,:take]}
            tokens=[t for b in sorted(chosen) for t in range(b*16,min((b+1)*16,self.position))]+[self.position]
            new_selection.append(tokens)
            missed=orders[head,take:];top=missed[candidate[head,missed].argsort(descending=True)[:3]]
            self.rows.append(dict(position=self.position,layer=layer,head=head,candidates=n,required=required,oracle_required=oracle,minimum_bucket=bucket,dynamic=dynamic,selected=take,fixed_mass=float(fixed[head]),coverage=float(curve[take]),dynamic_coverage=float(curve[dynamic]),fixed112_coverage=float(curve[min(112,n)]),fixed32_coverage=float(curve[min(32,n)]),missed_top=[dict(block=int(ids[j]),mass=float(candidate[head,j])) for j in top]))
        if self.mode in ['repair95','floor112']:
            indices,valid=rt.pad_indices(new_selection,query.device)
            self.step_rows[-1]['middle_blocks']=takes
            self.step_rows[-1]['selected_tokens']=[len(t) for t in new_selection]
        return ORIGINAL(query,keys,values,indices,valid,scale)

    def install(self):
        self.rows=[];super().install();rt.selected_attention=self.attention
    def uninstall(self):
        rt.selected_attention=ORIGINAL;super().uninstall()

@torch.inference_mode()
def main(shard):
    model,tok,ranker,budget,eos,_=run.load()
    payload=torch.load(QA/'budget/best.pt',weights_only=True)
    budget=e.bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();budget.load_state_dict(payload['state_dict']);budget.requires_grad_(False)
    quantile=json.loads((QA/'budget/calibration.json').read_text())['selected_quantile'];assert quantile==.9
    rows=json.loads((BASE/'selected.json').read_text());old=read(OLD_DIAG,'worker_*.jsonl');new=read(QA,'hotpot_*.jsonl')
    # Each GPU starts with one changed-answer case for early validation, without changing the sample.
    rows=sorted(rows,key=lambda r:(r['source_index'] not in CASES,r['source_index']))
    with (ROOT/f'worker_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            source=row['source_index'];assert row['input_ids_sha256']==new[source]['input_ids_sha256']==old[source]['input_ids_sha256']
            pairs,logits,_=run.prefill(model,row['input_ids']);digests=rt.build_prefix_digests(pairs)
            record=dict(source_index=source,input_ids_sha256=row['input_ids_sha256'],answers=row['answers'],methods={})
            for mode in ['adaptive','fixed112']+(['repair95','floor112'] if source in CASES else []):
                run.AdaptiveLlama=Diagnostic
                try:out=run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,quantile=quantile,capture=True)
                finally:run.AdaptiveLlama=rt.AdaptiveLlama
                controller=out.pop('_controller');diag=controller.rows
                if mode in ['adaptive','fixed112']:
                    expected=new[source]['new_adaptive'] if mode=='adaptive' else old[source]['fixed112']
                    assert out['ids']==expected['ids'],(source,mode,out['ids'],expected['ids'])
                    out['original_generation_parity']=True
                out['prediction']=tok.decode(out['ids'],skip_special_tokens=True).strip()
                out['f1']=100*max(run.token_f1(out['prediction'],a) for a in row['answers'])
                out['diagnostics']=diag;record['methods'][mode]=out
                print(json.dumps(dict(source_index=source,mode=mode,f1=out['f1'],prediction=out['prediction'],scenes=len(diag))),flush=True)
                del controller
            journal.write(json.dumps(record)+'\n');journal.flush()
            del record,out,diag,pairs,logits,digests;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)

if __name__=='__main__':main(int(sys.argv[1]))
