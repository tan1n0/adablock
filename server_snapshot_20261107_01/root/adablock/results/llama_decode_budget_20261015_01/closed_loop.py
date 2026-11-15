"""Held-out generated states of the newly frozen budget policy, evaluated only."""
import json
import sys
import torch
import pipeline as p

@torch.inference_mode()
def main(shard):
    model,tok,ranker,oldbudget,eos,_=p.run.load()
    payload=torch.load(p.ROOT/'budget/best.pt',weights_only=True)
    budget=p.bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval()
    budget.load_state_dict(payload['state_dict'])
    quantile=json.loads((p.ROOT/'budget/calibration.json').read_text())['selected_quantile']
    rows=[r for r in json.loads((p.ROOT/'selected.json').read_text()) if r['split']=='test']
    (p.ROOT/'closed_loop').mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        if i%4!=shard:continue
        pairs,logits,_=p.run.prefill(model,row['input_ids'])
        digests=p.rt.build_prefix_digests(pairs)
        p.run.AdaptiveLlama=p.Collector
        try:result=p.run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=quantile,capture=True)
        finally:p.run.AdaptiveLlama=p.rt.AdaptiveLlama
        controller=result.pop('_controller');records=controller.records
        # Confirm that teacher collection cannot alter the deployed output.
        plain=p.run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=quantile)
        assert plain['ids']==result['ids']
        features=torch.stack([r['features'] for r in records])
        probabilities=budget(features.cuda()).softmax(-1).cpu()
        actions=p.bp.quantile_actions(probabilities,quantile)
        coverage=torch.stack([r['coverage'] for r in records]);actual=torch.stack([r['actual'] for r in records])
        n=len(records);indices=torch.arange(n)
        cov=coverage[indices,actions];cost=actual[indices,actions].float()
        summary=dict(document=row['document'],scenes=n,hit95=float((cov>=.95-1e-7).float().mean()),
            mean_coverage=float(cov.mean()),mean_blocks=float(cost.mean()),generated_ids=result['ids'],
            prediction=tok.decode(result['ids'],skip_special_tokens=True),parity=True)
        target=p.ROOT/'closed_loop'/f"test_{row['document']}.pt";assert not target.exists()
        torch.save(dict(records=records,document=row,summary=summary,actions=actions),target)
        p.dump(f"closed_loop_{row['document']}.json",summary)
        print(json.dumps(summary),flush=True)
        del controller,records,pairs,logits,digests,features,probabilities,result,plain
    print('COMPLETE',flush=True)

if __name__=='__main__':main(int(sys.argv[1]))
