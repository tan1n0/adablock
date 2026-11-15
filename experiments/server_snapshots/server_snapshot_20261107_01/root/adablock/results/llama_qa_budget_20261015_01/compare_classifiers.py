"""Read-only comparison of frozen classifiers on the same QA scenes."""
import json
import torch
import engine as e

torch.set_num_threads(2)
previous=e.Path('/root/adablock/results/llama_decode_budget_20261015_01')
paths=dict(original=e.PILOT/'budget/best.pt',summary=previous/'budget/best.pt',qa=e.ROOT/'budget/best.pt')
quantiles=dict(original=.95,summary=json.loads((previous/'budget/calibration.json').read_text())['selected_quantile'],
               qa=json.loads((e.ROOT/'budget/calibration.json').read_text())['selected_quantile'])
models={}
for name,path in paths.items():
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model=e.bp.BudgetClassifier(payload['config']['feature_size']).eval();model.load_state_dict(payload['state_dict']);models[name]=model
report={}
with torch.inference_mode():
    for split in ['validation','test']:
        data=torch.load(e.ROOT/'budget_data'/f'{split}.pt',weights_only=True)
        report[split]={}
        for name,model in models.items():
            logits=torch.cat([model(x) for x in data['features'].split(1024)])
            actions=e.bp.quantile_actions(logits.softmax(-1),quantiles[name])
            report[split][name]=dict(ce=float(torch.nn.functional.cross_entropy(logits,data['labels'])),
                accuracy=float((logits.argmax(-1)==data['labels']).float().mean()),quantile=quantiles[name],
                calibrated=e.bp.metrics(data,actions),argmax=e.bp.metrics(data,logits.argmax(-1)))
(e.ROOT/'classification_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
