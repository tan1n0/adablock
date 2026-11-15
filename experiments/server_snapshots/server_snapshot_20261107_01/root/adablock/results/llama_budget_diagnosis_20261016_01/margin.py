import json,torch
import localize as l
d=l.d
@torch.inference_mode()
def main():
    model,tok,ranker,budget,eos,_=d.run.load()
    payload=torch.load(d.QA/'budget/best.pt',weights_only=True)
    budget=d.e.bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();budget.load_state_dict(payload['state_dict'])
    row=next(r for r in json.loads((d.BASE/'selected.json').read_text()) if r['source_index']==165)
    pairs,logits,_=d.run.prefill(model,row['input_ids']);digests=d.rt.build_prefix_digests(pairs)
    result={}
    for name,mode,allowed in [('adaptive','adaptive',[]),('repair_layer15','repair95',[15]),('full','full',[])]:
        l.LayerRepair.allowed=set(allowed)
        if allowed:d.run.AdaptiveLlama=l.LayerRepair
        try:out=d.run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,quantile=.9,return_logits=True)
        finally:d.run.AdaptiveLlama=d.rt.AdaptiveLlama
        v=out.pop('_logits')[1];values,ids=v.topk(5)
        result[name]=dict(prediction=tok.decode(out['ids'],skip_special_tokens=True),next_token_top5=[dict(id=int(i),token=tok.decode([int(i)]),logit=float(x)) for i,x in zip(ids,values)],end_turn_logit=float(v[128009]),ocean_logit=float(v[22302]),ocean_minus_end_turn=float(v[22302]-v[128009]))
    (d.ROOT/'token_margin.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
