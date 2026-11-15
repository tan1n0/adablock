import experiment as e
import stable_digest
import gc,json,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
PREVIOUS=Path('/root/adablock_runtime_optimization_20261022_01')

class Trace(e.gpu.Controller):
    def choose(self,layer,query,scale):
        original=e.gpu.base.forward_packed
        saved={}
        def forward(model,packed):
            score=original(model,packed)
            saved['scores']=score.detach().view(8,-1).cpu()
            saved['block_ids']=packed['block_ids'][:score.numel()//8].cpu()
            return score
        hook=self.budget.register_forward_hook(lambda module,args,out:saved.update(probability=out.detach().softmax(-1).cpu()))
        e.gpu.base.forward_packed=forward
        self.capture=False
        try:indices,valid=super().choose(layer,query,scale)
        finally:e.gpu.base.forward_packed=original;hook.remove();self.capture=True
        saved.update(position=self.position,layer=layer,query=query.cpu(),
            classes=self.step_rows[-1]['classes'].cpu(),takes=self.step_rows[-1]['middle_blocks'].cpu(),
            selected=[torch.unique(ix[mask]//16).cpu() for ix,mask in zip(indices,valid)])
        self.selected.append(saved)
        return indices,valid

def function(version):
    if version=='svd':return e.ORIGINAL_DIGEST
    if version=='gram':return e.fast_digest.compact_digest
    return lambda keys:stable_digest.compact_digest(keys,method=version)

@torch.inference_mode()
def build(p,version):
    fn=function(version);complete=p[0][0].shape[2]//16*16
    torch.cuda.synchronize();start=time.perf_counter()
    d=[fn(k[0,:,:complete]) for k,_ in p]
    torch.cuda.synchronize()
    return d,time.perf_counter()-start

@torch.inference_mode()
def generate(model,p,l,d,rank,budget,eos,version,cap,trace=False):
    cls=Trace if trace else e.gpu.Controller;cls.ratio=.25
    e.c.run.AdaptiveLlama=cls;e.c.run.DynamicCache=e.BufferCache
    e.baseline.compact_digest=function(version)
    try:return e.c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=cap,quantile=.95,capture=trace)
    finally:e.baseline.compact_digest=e.ORIGINAL_DIGEST

def compare(a,b,tok):
    report={'token_first_difference':next((i for i,(x,y) in enumerate(zip(a['ids'],b['ids'])) if x!=y),None),
            'selection_differing_layers':0,'budget_differing_layers':0}
    for ra,rb in zip(a['trace'],b['trace']):
        differs=not all(torch.equal(x,y) for x,y in zip(ra['selected'],rb['selected']))
        report['selection_differing_layers']+=differs
        report['budget_differing_layers']+=not torch.equal(ra['classes'],rb['classes'])
        if differs and 'first_selection_difference' not in report:
            heads=[]
            for h,(x,y) in enumerate(zip(ra['selected'],rb['selected'])):
                if torch.equal(x,y):continue
                scores=ra['scores'][h];order=scores.argsort(descending=True,stable=True);k=int(ra['takes'][h])
                heads.append(dict(head=h,old_take=k,new_take=int(rb['takes'][h]),old_class=int(ra['classes'][h]),new_class=int(rb['classes'][h]),
                    lost=sorted(set(x.tolist())-set(y.tolist())),gained=sorted(set(y.tolist())-set(x.tolist())),
                    cutoff_gap=float(scores[order[k-1]]-scores[order[k]]) if 0<k<len(scores) else None,
                    score_max_abs=float((scores-rb['scores'][h]).abs().max())))
            report['first_selection_difference']=dict(position=ra['position'],layer=ra['layer'],
                query_equal=torch.equal(ra['query'],rb['query']),heads=heads,
                old_probability=ra['probability'].tolist(),new_probability=rb['probability'].tolist())
    i=report['token_first_difference']
    if i is not None:report['first_tokens']=dict(before=tok.decode(a['ids'][max(0,i-5):i+5]),after=tok.decode(b['ids'][max(0,i-5):i+5]))
    return report

@torch.inference_mode()
def main():
    rows=json.loads((PREVIOUS/'selected.json').read_text())
    row=next(r for r in rows if r['task']=='cwe' and r['length']==8192)
    model,tok,models,eos=e.load();rank,budget=models['B']
    p,l,prefill=e.prefill(model,row['input_ids'])
    report=dict(task=row['task'],length=row['length'],prompt_tokens=len(row['input_ids']),answers=row['answers'],versions={})
    traces={}
    for version in ['svd','gram','double','qr']:
        d,seconds=build(p,version)
        result=generate(model,p,l,d,rank,budget,eos,version,row['max_new'],trace=version in ['svd','gram'])
        obj=result.pop('_controller',None)
        if obj is not None:
            traces[version]=dict(ids=result['ids'],trace=obj.selected)
            torch.save(traces[version],ROOT/f'trace_{version}.pt')
        prediction=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
        score=e.official.metric_for(row['task'])([prediction],[row['answers']])
        report['versions'][version]=dict(ids=result['ids'],prediction=prediction,score=score,index_seconds=seconds,
             decode_seconds=result['decode_seconds'],instrumented=version in ['svd','gram'])
        e.dump('diagnosis_partial.json',report)
        print(version,json.dumps(report['versions'][version]),flush=True)
        del obj,result,d;gc.collect();torch.cuda.empty_cache()
    report['gram_vs_svd']=compare(traces['svd'],traces['gram'],tok)
    e.dump('diagnosis.json',report)
    print(json.dumps(report['gram_vs_svd']),flush=True)

if __name__=='__main__':main()
