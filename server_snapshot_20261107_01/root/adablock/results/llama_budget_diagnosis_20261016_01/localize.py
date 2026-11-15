"""Post-hoc one-layer budget repair on the Atlantic/Ocean example only."""
import json,sys
import torch
import diagnose as d

class LayerRepair(d.Diagnostic):
    allowed=set()
    def attention(self,*args):
        mode=self.mode
        if self.probe[0] not in self.allowed:self.mode='adaptive'
        try:return super().attention(*args)
        finally:self.mode=mode

@torch.inference_mode()
def main():
    model,tok,ranker,budget,eos,_=d.run.load()
    payload=torch.load(d.QA/'budget/best.pt',weights_only=True)
    budget=d.e.bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();budget.load_state_dict(payload['state_dict'])
    row=next(r for r in json.loads((d.BASE/'selected.json').read_text()) if r['source_index']==165)
    previous=d.read(d.ROOT,'worker_*.jsonl')[165]['methods']['adaptive']
    layers=sorted({r['layer'] for r in previous['diagnostics'] if r['selected']<r['required']})
    pairs,logits,_=d.run.prefill(model,row['input_ids']);digests=d.rt.build_prefix_digests(pairs)
    results=[]
    for allowed in [[],layers]+[[l] for l in layers]:
        LayerRepair.allowed=set(allowed);d.run.AdaptiveLlama=LayerRepair
        try:out=d.run.decode(model,pairs,logits,'repair95',ranker,budget,digests,eos,quantile=.9,capture=True)
        finally:d.run.AdaptiveLlama=d.rt.AdaptiveLlama
        controller=out.pop('_controller')
        if not allowed:assert out['ids']==previous['ids']
        pred=tok.decode(out['ids'],skip_special_tokens=True).strip()
        first=[r for r in controller.rows if r['position']==row['prompt_tokens']]
        rec=dict(repaired_layers=allowed,prediction=pred,ids=out['ids'],f1=100*max(d.run.token_f1(pred,a) for a in row['answers']),first_step_mean_blocks=sum(r['selected'] for r in first)/len(first),first_step_under=sum(r['selected']<r['required'] for r in first),first_step_rows=first)
        results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='first_step_rows'}),flush=True)
        del controller
    (d.ROOT/'localization.json').write_text(json.dumps(dict(source_index=165,layer_numbering='zero-based',scope='Post-hoc single-question causal interventions, no parameter fitting or protocol selection',results=results),indent=2))
if __name__=='__main__':main()
