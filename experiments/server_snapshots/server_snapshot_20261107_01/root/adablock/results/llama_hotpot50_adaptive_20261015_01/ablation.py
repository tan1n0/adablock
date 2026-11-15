"""One controlled offline ablation: ideal-order labels vs frozen-ranker labels."""
import json
import os
from pathlib import Path
import sys
import torch
from prepare import ROOT, PILOT
sys.path.insert(0,str(PILOT))
import budget_pilot as bp


def main():
    out=ROOT/'ablation_oracle'
    out.mkdir()
    (out/'ranker').mkdir(); (out/'budget_data').mkdir()
    os.link(PILOT/'ranker/best.pt',out/'ranker/best.pt')
    changed={}
    for split in ['budget_train','validation']:
        data=torch.load(PILOT/'budget_data'/f'{split}.pt',weights_only=True)
        old=data['labels'].clone()
        hit=data['actual_blocks'] >= data['oracle_sort_required_exact'][:,None]
        assert hit.any(-1).all()
        data['labels']=hit.long().argmax(-1)
        data['ranker_conditioned_labels']=old
        changed[split]={'changed_labels':int((old!=data['labels']).sum()),'scenes':len(old),
                        'larger_than_actual_labels':int((data['labels']>old).sum())}
        assert not (data['labels']>old).any()
        torch.save(data,out/'budget_data'/f'{split}.pt')
    (out/'label_changes.json').write_text(json.dumps(changed,indent=2))
    bp.ROOT=out
    bp.train()
    report={'scope':'offline independent-document evaluation; not additional HotpotQA generation arm',
            'controlled_change':'budget labels from ideal teacher ranking instead of frozen learned ranking',
            'held_fixed':'features, ranker, architecture, optimizer, seed, split, epochs, validation-selection and calibration rule',
            'changed_labels':changed,'results':{}}
    for name,folder in [('actual_ranking_labels',PILOT),('ideal_ranking_labels',out)]:
        payload=torch.load(folder/'budget/best.pt',map_location='cpu',weights_only=True)
        model=bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval()
        model.load_state_dict(payload['state_dict'])
        quantile=json.loads((folder/'budget/calibration.json').read_text())['selected_quantile']
        methods={}
        for split in ['validation','test']:
            data=torch.load(PILOT/'budget_data'/f'{split}.pt',weights_only=True)
            probability=bp.probabilities(model,data)
            methods[split]={'argmax':bp.metrics(data,probability.argmax(-1)),
                            'calibrated':bp.metrics(data,bp.quantile_actions(probability,quantile))}
        report['results'][name]={'epoch':payload['epoch'],'quantile':quantile,'splits':methods}
    (ROOT/'ablation_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
