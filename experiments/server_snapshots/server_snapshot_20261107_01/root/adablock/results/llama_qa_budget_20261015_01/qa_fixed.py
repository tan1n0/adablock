"""Fixed-budget and prior-summary-model controls on the frozen WikiHop test questions."""
import json
import sys
import torch
import engine as e

@torch.inference_mode()
def main(shard):
    model,tok,ranker,oldbudget,eos,_=e.run.load()
    previous=e.Path('/root/adablock/results/llama_decode_budget_20261015_01')
    payload=torch.load(previous/'budget/best.pt',weights_only=True)
    summary=e.bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();summary.load_state_dict(payload['state_dict'])
    quantile=json.loads((previous/'budget/calibration.json').read_text())['selected_quantile']
    rows=[r for r in json.loads((e.ROOT/'selected.json').read_text()) if r['split']=='test']
    with (e.ROOT/f'qa_fixed_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            pairs,logits,_=e.run.prefill(model,row['input_ids']);digests=e.rt.build_prefix_digests(pairs)
            result=dict(document=row['document'],doc_id=row['doc_id'],input_ids_sha256=row['input_ids_sha256'],methods={})
            for method,mode,budget,q in [('fixed32','fixed32',oldbudget,.95),('fixed112','fixed112',oldbudget,.95),('summary_budget','adaptive',summary,quantile)]:
                r=e.run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,quantile=q)
                r['prediction']=tok.decode(r['ids'],skip_special_tokens=True).strip()
                r['f1']=100*max(e.run.token_f1(r['prediction'],a) for a in row['answers'])
                result['methods'][method]=r
            journal.write(json.dumps(result)+'\n');journal.flush()
            print(json.dumps(dict(document=row['document'],scores={k:v['f1'] for k,v in result['methods'].items()})),flush=True)
            del pairs,logits,digests,result
    print('COMPLETE',flush=True)

if __name__=='__main__':main(int(sys.argv[1]))
