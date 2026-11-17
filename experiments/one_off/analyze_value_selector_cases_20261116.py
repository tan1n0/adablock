"""Paired RULER case audit with explicit multivalue misses and length shifts."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01'
SELECTOR=ROOT/'results/value_aware_selector_20261116/selector_full_100_gpu'
OUT=ROOT/'results/value_aware_selector_20261116/selector_full_100_gpu/paired_analysis.json'

def run():
    selected=json.loads((SOURCE/'selected.json').read_text())
    rows=[]
    for i,source in enumerate(selected):
        old=json.loads((SOURCE/f'quality_{i:03d}.json').read_text())
        new=json.loads((SELECTOR/f'case_{i:03d}.json').read_text())
        if len({source['input_sha256'],old['input_sha256'],new['input_sha256']})!=1:
            raise ValueError(f'Hash mismatch {i}')
        methods={'full':old['methods']['full'],'mlp':new['methods']['mlp'],'cobs':new['methods']['cobs']}
        rows.append({'index':i,'task':source['task'],'answers':source['answers'],
                     'methods':{name:{k:v[k] for k in ('ruler_score','all_hit','hits','decode_steps','cap_reached','prediction')}
                                for name,v in methods.items()}})
    groups={}
    for group in ('all','niah_multivalue','vt'):
        cases=[x for x in rows if group=='all' or x['task']==group]
        pairs={}
        for a,b in (('full','cobs'),('mlp','cobs')):
            pairs[f'{b}_vs_{a}']={'score_delta':mean(x['methods'][b]['ruler_score']-x['methods'][a]['ruler_score'] for x in cases),
                 'lost_all_hit':sum(x['methods'][a]['all_hit'] and not x['methods'][b]['all_hit'] for x in cases),
                 'gained_all_hit':sum(not x['methods'][a]['all_hit'] and x['methods'][b]['all_hit'] for x in cases),
                 'mean_decode_step_delta':mean(x['methods'][b]['decode_steps']-x['methods'][a]['decode_steps'] for x in cases)}
        groups[group]={'n':len(cases),'pairs':pairs}
    misses=[]
    for x in rows:
        if x['task']!='niah_multivalue' or x['methods']['cobs']['all_hit']:continue
        a=x['methods']['cobs']['prediction'].lower()
        misses.append({'index':x['index'],'full_score':x['methods']['full']['ruler_score'],
                       'mlp_score':x['methods']['mlp']['ruler_score'],
                       'cobs_score':x['methods']['cobs']['ruler_score'],
                       'full_hits':x['methods']['full']['hits'],
                       'mlp_hits':x['methods']['mlp']['hits'],
                       'cobs_hits':x['methods']['cobs']['hits'],
                       'cobs_decode_steps':x['methods']['cobs']['decode_steps'],
                       'answers_missing_literal':[v for v in x['answers'] if v.lower() not in a]})
    result={'groups':groups,'multivalue_cobs_non_all_hit':misses,
            'note':'literal missing list is diagnostic; official task metric and regex all-hit are authoritative'}
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(OUT)
    print(json.dumps({'groups':groups,'multivalue_cobs_non_all_hit':misses},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
