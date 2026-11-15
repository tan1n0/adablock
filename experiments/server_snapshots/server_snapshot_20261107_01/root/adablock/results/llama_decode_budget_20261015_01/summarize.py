import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
b=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
f=Path('/root/adablock/results/llama_hotpot50_fixed112_diagnostics_20261015_01')
def read(pattern,root):return [json.loads(s) for file in root.glob(pattern) for s in file.read_text().splitlines()]
old={r['source_index']:r for r in read('worker_*.jsonl',b)}
fixed={r['source_index']:r for r in read('worker_*.jsonl',f)}
rows=sorted(read('hotpot_*.jsonl',p),key=lambda r:r['source_index'])
assert len(rows)==50 and len({r['source_index'] for r in rows})==50
for r in rows:assert r['input_ids_sha256']==old[r['source_index']]['input_ids_sha256']
methods={m:[old[r['source_index']]['methods'][m] for r in rows] for m in ['full','fixed32','fixed64','adaptive']}
methods['fixed112']=[fixed[r['source_index']]['fixed112'] for r in rows]
methods['new_adaptive']=[r['new_adaptive'] for r in rows]
def mean(x):return float(np.mean(x))
def padded(r):
    values=[]
    for step in r['steps']:
        pos=step['position'];n=max(0,(pos+15)//16-34);fixed_tokens=pos+1-n*16
        budgets=[0,4,8,16,32,64,128,256,n]
        classes=np.array(step['budget_classes']).reshape(32,8)
        sizes=np.array([min(n,x) for x in budgets])[classes]
        values.append(mean((fixed_tokens+sizes.max(1)*16)/(pos+1)))
    return mean(values)
summary={}
for method,rs in methods.items():
    summary[method]=dict(f1=mean([r['f1'] for r in rs]),mean_middle_blocks=mean([r['mean_middle_blocks'] for r in rs]),
        logical_read_fraction=mean([r['mean_read_fraction'] for r in rs]),
        padded_read_fraction=mean([padded(r) for r in rs]) if method in ['adaptive','new_adaptive'] else mean([r['mean_read_fraction'] for r in rs]),
        decode_ms_per_step_question_mean=1000*mean([r['seconds_per_decode_step'] for r in rs]),
        untruncated_f1=mean([r['f1'] for r,q in zip(rs,rows) if not q['truncated']]),
        truncated_f1=mean([r['f1'] for r,q in zip(rs,rows) if q['truncated']]),
        generation_limit_count=sum(r['hit_generation_limit'] for r in rs))
def paired(left,right):
    d=np.array([r['f1'] for r in left])-np.array([r['f1'] for r in right])
    rng=np.random.default_rng(20261015);boot=d[rng.integers(0,50,(10000,50))].mean(1)
    return dict(delta=mean(d),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((d>1e-8).sum()),ties=int((abs(d)<=1e-8).sum()),losses=int((d< -1e-8).sum()))
closed=[json.loads((p/f'closed_loop_{i}.json').read_text()) for i in range(6)]
closed_summary=dict(documents=6,scenes=sum(r['scenes'] for r in closed),
    document_mean={k:mean([r[k] for r in closed]) for k in ['hit95','mean_coverage','mean_blocks']},
    scene_weighted={k:sum(r[k]*r['scenes'] for r in closed)/sum(r['scenes'] for r in closed) for k in ['hit95','mean_coverage','mean_blocks']})
validation={}
for manifest in ['source_sha256_frozen.json','policy_frozen_before_hotpot.json']:
    for name,digest in json.loads((p/manifest).read_text()).items():
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
    validation[manifest]=True
for name in ['run.py','runtime.py','prepare.py','protocol.json']:assert (p/'source'/name).read_bytes()==(b/name).read_bytes()
results=dict(budget=json.loads((p/'budget_results.json').read_text()),hotpot=dict(samples=50,methods=summary,
    paired_new_minus={m:paired(methods['new_adaptive'],methods[m]) for m in methods if m!='new_adaptive'}),
    closed_loop=closed_summary,validation=validation,caveats=[
    'Only six training, six validation, and six test documents; generated scenes are highly correlated.',
    'Training material uses chat summarization; HotpotQA is short-answer QA. Transfer is not guaranteed.',
    'Hotpot50 was inspected in earlier rounds and is a regression set, not a fresh unbiased final benchmark.',
    'Original checkpoints and result files preserved. Quantiles selected using independent validation documents only.',
    'All KV retained. Unfused prototype timing, separate runs; logical reads do not equal kernel cost or speedup.'])
(p/'results.json').write_text(json.dumps(results,indent=2))
with (p/'hotpot_comparison.csv').open('w',newline='') as out:
    w=csv.DictWriter(out,fieldnames=['method']+list(summary['full']));w.writeheader();w.writerows(dict(method=k,**v) for k,v in summary.items())
with (p/'hotpot_per_question.csv').open('w',newline='') as out:
    fields=['source_index','truncated']+[m+'_f1' for m in methods]+[m+'_prediction' for m in methods]
    w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
    for i,r in enumerate(rows):w.writerow(dict(source_index=r['source_index'],truncated=r['truncated'],**{m+'_f1':rs[i]['f1'] for m,rs in methods.items()},**{m+'_prediction':rs[i]['prediction'] for m,rs in methods.items()}))
report=['Real-decoding budget training pilot',json.dumps(summary,indent=2),'Paired F1 differences:',json.dumps(results['hotpot']['paired_new_minus'],indent=2),
    'Held-out generated scenes:',json.dumps(results['budget']['splits']['test']['overall'],indent=2),'New policy closed-loop test:',json.dumps(closed_summary,indent=2)]+results['caveats']
(p/'report.txt').write_text('\n'.join(report))
print('\n'.join(report))
