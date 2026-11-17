"""Reconcile the four completed 100-case baselines by input hash and task."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01'
B=ROOT/'results/ranker_sort_ablation_20261115'
S=ROOT/'results/s_no_fixed_20261116/ruler_eval'
ORACLE=ROOT/'results/oracle_block_granularity_20261115'
OUT=ROOT/'results/value_aware_selector_20261116/historical_audit.json'

def run():
    selected=json.loads((SOURCE/'selected.json').read_text())
    rows=[]
    for index,row in enumerate(selected):
        old=json.loads((SOURCE/f'quality_{index:03d}.json').read_text())
        b=json.loads((B/f'case_{index:03d}.json').read_text())
        s=json.loads((S/f'case_{index:03d}.json').read_text())
        oracle=json.loads((ORACLE/f'case_{index:03d}.json').read_text())
        if len({row['input_sha256'],old['input_sha256'],b['input_sha256'],s['input_sha256'],oracle['input_sha256']})!=1:
            raise ValueError(f'Input mismatch at case {index}')
        methods={'full':old['methods']['full'],'B_full':b['methods']['full_sort'],
                 'B_coarse256':b['methods']['coarse_256'],
                 'S':s['methods']['full_sort'],'mass_oracle':oracle['methods']['b16_k512']}
        rows.append({'index':index,'task':row['task'],'input_sha256':row['input_sha256'],
                     'methods':{name:{key:item[key] for key in ('ruler_score','all_hit','hits','decode_steps','cap_reached')}
                                for name,item in methods.items()}})
    names=list(rows[0]['methods'])
    groups={}
    for group in ('all','niah_multivalue','vt'):
        cases=[r for r in rows if group=='all' or r['task']==group]
        scores={name:{'score':mean(r['methods'][name]['ruler_score'] for r in cases),
                      'all_hit':sum(r['methods'][name]['all_hit'] for r in cases),
                      'mean_decode_steps':mean(r['methods'][name]['decode_steps'] for r in cases),
                      'cap_reached':sum(r['methods'][name]['cap_reached'] for r in cases)} for name in names}
        pairs={name:{'mean_score_delta_vs_full':mean(r['methods'][name]['ruler_score']-r['methods']['full']['ruler_score'] for r in cases),
                     'worse_score_cases':sum(r['methods'][name]['ruler_score']<r['methods']['full']['ruler_score'] for r in cases),
                     'lost_all_hit_cases':sum(r['methods']['full']['all_hit'] and not r['methods'][name]['all_hit'] for r in cases),
                     'gained_all_hit_cases':sum(not r['methods']['full']['all_hit'] and r['methods'][name]['all_hit'] for r in cases)} for name in names if name!='full'}
        groups[group]={'n':len(cases),'methods':scores,'paired_vs_full':pairs}
    report={'source':'same 100 input hashes; historical results use their documented backends and timing scopes',
            'groups':groups,'cases':rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    tmp=OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)); tmp.replace(OUT)
    print(json.dumps(groups,indent=2))

if __name__=='__main__':
    run()
