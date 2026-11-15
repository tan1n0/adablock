"""Frozen Hotpot50 three-threshold audit; no tuning or training."""
import json,csv,hashlib,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
RUNS={'85':Path('/root/adablock/results/llama_oracle85_20261016_01'),'90':ROOT,'95':Path('/root/adablock/results/llama_oracle_granularity_20261016_01')}
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
def load(r):return {x['source_index']:x for p in r.glob('worker_*.jsonl') for l in p.read_text().splitlines() if (x:=json.loads(l))}
def main():
    stats={k:json.loads((r/'results.json').read_text()) for k,r in RUNS.items()}
    rows={k:load(r) for k,r in RUNS.items()};full=load(BASE)
    selected={r['source_index']:r for r in json.loads((BASE/'selected.json').read_text())}
    for k in rows:
        assert set(rows[k])==set(selected) and len(rows[k])==50
        for i,r in rows[k].items():assert r['input_ids_sha256']==selected[i]['input_ids_sha256']
    out=dict(n=50,thresholds={k:v['arms'] for k,v in stats.items()},paired={},changed_vs95={},focus_cases={},full_reference_f1=stats['90']['full_reference_f1'],validation=stats['90']['validation'])
    table=[];details=[]
    for k,s in stats.items():
        for arm,a in s['arms'].items():table.append(dict(threshold=k,arm=arm,**a))
    for arm in ['oracle_exact','oracle_bucket']:
        for new,old in [('90','95'),('90','85'),('85','95')]:
            diff=np.array([rows[new][i]['methods'][arm]['f1']-rows[old][i]['methods'][arm]['f1'] for i in sorted(selected)])
            rng=np.random.default_rng(20261016);boot=rng.choice(diff,size=(20000,50),replace=True).mean(1)
            a=stats[old]['arms'][arm];b=stats[new]['arms'][arm]
            out['paired'][arm+'_'+new+'_vs_'+old]=dict(f1_delta=float(diff.mean()),wins=int((diff>1e-7).sum()),ties=int((abs(diff)<=1e-7).sum()),losses=int((diff<-1e-7).sum()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),relative_savings={key:1-b[key]/a[key] for key in ['blocks','logical_fraction','padded_fraction','total_logical_token_slots','total_padded_token_slots','padding_wasted_slots']})
        changed=[]
        for i in sorted(selected):
            v={k:rows[k][i]['methods'][arm] for k in rows}
            details.append(dict(source_index=i,arm=arm,answers=json.dumps(selected[i]['answers']),**{field+'_'+k:v[k][field] for k in v for field in ['f1','prediction','mean_middle_blocks']}))
            if v['90']['ids']!=v['95']['ids']:
                changed.append(dict(source_index=i,question=selected[i]['question'],answers=selected[i]['answers'],predictions={k:{field:v[k][field] for field in ['f1','prediction','ids']} for k in v}))
        out['changed_vs95'][arm]=changed
    for i in [165,181,127,160]:
        out['focus_cases'][str(i)]=dict(question=selected[i]['question'],answers=selected[i]['answers'],full={k:full[i]['methods']['full'][k] for k in ['prediction','f1']},thresholds={k:{arm:{field:rows[k][i]['methods'][arm][field] for field in ['prediction','f1','mean_middle_blocks']} for arm in ['oracle_exact','oracle_bucket']} for k in rows})
    out['limits']=['Full-attention oracle budgets, not predictions from a learned budget model.','Same reused Hotpot50; no final threshold selection or training was performed.','Full-attention prefill and shared first token, sparse generation only.','Padding-token slots are cost proxies, not measured inference speed or memory savings.','Independent thresholds have their own generated hidden states; per-question means and answer-total costs are both reported.']
    (ROOT/'three_threshold_comparison.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    for name,rr in [('three_threshold_summary.csv',table),('three_threshold_per_question.csv',details)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    with zipfile.ZipFile(ROOT/'reproduction_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in ['.py','.json','.jsonl','.csv','.log'] and p.name!='artifact_sha256.json':z.write(p,p.name)
    files=['results.json','protocol.json','per_question.csv','three_threshold_comparison.json','three_threshold_summary.csv','three_threshold_per_question.csv','reproduction_bundle.zip']
    (ROOT/'artifact_sha256.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2))
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
