import json,hashlib
from pathlib import Path
import torch
r=Path(__file__).resolve().parent
out=[]
for i in [129,50]:
    base=torch.load(r/f'four_{i}_logits.pt',map_location='cpu',weights_only=True)
    a=json.loads((r/f'four_{i}.json').read_text())
    for mode in ['single','author']:
        b=json.loads((r/f'{mode}_{i}.json').read_text())
        v=torch.load(r/f'{mode}_{i}_logits.pt',map_location='cpu',weights_only=True)
        row=dict(index=i,comparison=f'four_vs_{mode}',same_tokens=a['token_ids']==b['token_ids'],four_F1=a['F1'],single_F1=b['F1'],same_prediction=a['prediction']==b['prediction'])
        for k in ['raw','processed']:
            row[k]=dict(exact_equal=torch.equal(base[k],v[k]),different_entries=int((base[k]!=v[k]).sum()),max_abs_diff=float((base[k]-v[k]).abs().max()))
        out.append(row)
before=json.loads((r/'four_manifest.json').read_text())['original_predictions_sha256']
original=Path('/root/adablock/results/fasa_full_reproduction_20261005_multigpu/run/predictions.jsonl')
assert hashlib.sha256(original.read_bytes()).hexdigest()==before
summary=dict(samples=2,indices_zero_based=[129,50],branches=['four','single','author'],comparisons=out,original_unchanged=True,scope='Short-input placement and loader diagnostic; not a 200-sample single-GPU rerun.')
with (r/'comparison.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
