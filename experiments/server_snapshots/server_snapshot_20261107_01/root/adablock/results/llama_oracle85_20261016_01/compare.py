"""Paired rho .85 versus frozen rho .95, no tuning."""
import json,hashlib,zipfile,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/adablock/results/llama_oracle_granularity_20261016_01')
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
def load(r):return {x['source_index']:x for f in r.glob('worker_*.jsonl') for l in f.read_text().splitlines() if (x:=json.loads(l))}
def main():
    old=json.loads((OLD/'results.json').read_text());new=json.loads((ROOT/'results.json').read_text())
    before=load(OLD);after=load(ROOT);full=load(BASE)
    selected={r['source_index']:r for r in json.loads((BASE/'selected.json').read_text())}
    assert set(before)==set(after)==set(selected) and len(after)==50
    result=dict(targets=[.95,.85],n=50,comparison={},changed_answers={},full_reference_f1=new['full_reference_f1'],validation=new['validation'])
    details=[]
    for arm in ['oracle_exact','oracle_bucket']:
        a=old['arms'][arm];b=new['arms'][arm];diff=[];changes=[];zero=[]
        for i,r in sorted(after.items()):
            assert r['input_ids_sha256']==before[i]['input_ids_sha256']
            prev=before[i]['methods'][arm];now=r['methods'][arm]
            delta=now['f1']-prev['f1'];diff.append(delta)
            ds=now['diagnostics'];zero.append(np.mean([k==0 for d in ds for k in d['selected']]))
            details.append(dict(source_index=i,arm=arm,f1_95=prev['f1'],f1_85=now['f1'],delta_f1=delta,blocks_95=prev['mean_middle_blocks'],blocks_85=now['mean_middle_blocks'],prediction_95=prev['prediction'],prediction_85=now['prediction'],answers=json.dumps(r['answers'])))
            if prev['ids']!=now['ids']:
                changes.append(dict(source_index=i,question=selected[i]['question'],answers=r['answers'],delta_f1=delta,before={k:prev[k] for k in ['prediction','f1','ids']},after={k:now[k] for k in ['prediction','f1','ids']},full={k:full[i]['methods']['full'][k] for k in ['prediction','f1']}))
        diff=np.array(diff);rng=np.random.default_rng(20261016);boot=rng.choice(diff,size=(20000,50),replace=True).mean(1)
        result['comparison'][arm]=dict(rho95=a,rho85=b,f1_delta=float(diff.mean()),wins=int((diff>1e-7).sum()),ties=int((abs(diff)<=1e-7).sum()),losses=int((diff<-1e-7).sum()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),identical_generated_tokens=50-len(changes),mean_zero_middle_budget_fraction=float(np.mean(zero)),relative_savings={k:1-b[k]/a[k] for k in ['blocks','logical_fraction','padded_fraction','padding_extra_fraction','total_logical_token_slots','total_padded_token_slots','padding_wasted_slots']})
        result['changed_answers'][arm]=changes
    result['limitations']=['Oracle teacher budgets, not learned budget predictions.','This reused Hotpot50 is diagnostic; do not select a final threshold on this set.','Coverage is mean across Q heads within each KV head group, not a per-Q-head guarantee.','Full attention prefill and shared first output token; result does not evaluate sparse prefill.','Padded slots are a cost proxy, not speed or GPU-memory savings.']
    (ROOT/'comparison_85_vs_95.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    with (ROOT/'comparison_per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    with zipfile.ZipFile(ROOT/'reproduction_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in ['.py','.json','.jsonl','.csv','.log'] and p.name!='artifact_sha256.json':z.write(p,p.name)
    files=['results.json','protocol.json','per_question.csv','comparison_85_vs_95.json','comparison_per_question.csv','reproduction_bundle.zip']
    (ROOT/'artifact_sha256.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2))
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
