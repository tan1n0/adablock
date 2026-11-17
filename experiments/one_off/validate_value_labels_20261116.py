"""Audit all real decode label shards, causal budgets and swap-set context."""
from __future__ import annotations
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/value_aware_selector_20261116/value_labels_full'
OUT=BASE/'validation_audit.json'

def run():
    report={'splits':{}};sets={}
    for split,expected in [('train',8),('validation',4)]:
        files=sorted((BASE/split).glob('doc_*.pt'))
        if len(files)!=expected: raise ValueError(f'{split}: expected {expected} shards, found {len(files)}')
        parts=set();records=0;swaps=0;error_mass=0.;error_value=0.;lengths=[]
        for path in files:
            item=torch.load(path,map_location='cpu',weights_only=True)
            lengths.append(item['input_length'])
            parts.update(c for p in item['source_parts'] for c in p['component_doc_ids'])
            for row in item['records']:
                records+=1
                m=row['teacher_mass_g'].float();fixed=row['fixed_mass_g'].float()
                if not torch.allclose(m.sum(-1)+fixed,torch.ones_like(fixed),atol=3e-3):
                    raise ValueError(f'{path}: probability mass does not sum to one')
                initial=row['mass_selected'].tolist();final=row['value_selected'].tolist()
                if len(initial)!=32 or len(set(initial))!=32 or min(initial)<0 or max(initial)>=m.shape[-1]:
                    raise ValueError(f'{path}: invalid mass set')
                chosen=set(initial)
                for outgoing,incoming,gain in row['swaps']:
                    if outgoing not in chosen or incoming in chosen or gain<=0:
                        raise ValueError(f'{path}: invalid context-dependent swap')
                    chosen.remove(outgoing);chosen.add(incoming);swaps+=1
                if chosen!=set(final):raise ValueError(f'{path}: final swap set mismatch')
                if row['value_mse']>row['mass_mse']+1e-6:
                    raise ValueError(f'{path}: local search worsened output')
                if split=='validation' and 'contribution_g' not in row:
                    raise ValueError(f'{path}: validation contributions missing')
                error_mass+=row['mass_mse'];error_value+=row['value_mse']
        sets[split]=parts
        report['splits'][split]={'documents':len(files),'records':records,'accepted_swaps':swaps,
            'context_lengths':lengths,'mean_mass_mse':error_mass/records,
            'mean_value_mse':error_value/records,
            'relative_mse_reduction':1-error_value/error_mass,'source_components':len(parts)}
    if sets['train']&sets['validation']:raise ValueError('Train/validation component overlap')
    report['train_validation_component_overlap']=0
    OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
