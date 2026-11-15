import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
rows=sorted([json.loads(line) for f in ROOT.glob('worker_*.jsonl') for line in f.read_text().splitlines()],key=lambda r:r['source_index'])
old={r['source_index']:r for f in BASE.glob('worker_*.jsonl') for line in f.read_text().splitlines() if (r:=json.loads(line))}
assert len(rows)==50 and len({r['source_index'] for r in rows})==50
assert all(r['trajectory_ids_match'] and r['input_ids_sha256']==old[r['source_index']]['input_ids_sha256'] for r in rows)
def avg(xs): return float(np.mean(xs))
def paired(a,b):
    d=np.array(a)-np.array(b)
    rng=np.random.default_rng(20261015)
    boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    return dict(delta=avg(d),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((d>1e-8).sum()),ties=int((abs(d)<=1e-8).sum()),losses=int((d < -1e-8).sum()))
methods={m:[old[r['source_index']]['methods'][m] for r in rows] for m in ['full','fixed32','fixed64','adaptive']}
methods['fixed112']=[r['fixed112'] for r in rows]
summary={}
for m,rs in methods.items():
    summary[m]=dict(f1=avg([r['f1'] for r in rs]),mean_middle_blocks=avg([r['mean_middle_blocks'] for r in rs]),
        logical_read_fraction=avg([r['mean_read_fraction'] for r in rs]),
        decode_ms_per_step=1000*avg([r['seconds_per_decode_step'] for r in rs]),
        untruncated_f1=avg([r['f1'] for r,q in zip(rs,rows) if not q['truncated']]),
        truncated_f1=avg([r['f1'] for r,q in zip(rs,rows) if q['truncated']]))

def diagnose(ds):
    return dict(scenes=len(ds), **{key:avg([d[key] for d in ds]) for key in ['candidates','fixed_mass','required','oracle_required','minimum_bucket','predicted','argmax','mass_predicted','mass_argmax','mass32','mass64','mass112']},
        **{key+'_hit95':avg([d[key]>=.95 for d in ds]) for key in ['mass_predicted','mass_argmax','mass32','mass64','mass112']},
        fixed_alone_hit95=avg([d['fixed_mass']>=.95 for d in ds]),
        over_bucket=avg([d['predicted']>d['minimum_bucket'] for d in ds]),
        equal_bucket=avg([d['predicted']==d['minimum_bucket'] for d in ds]),
        under_bucket=avg([d['predicted']<d['minimum_bucket'] for d in ds]),
        excess_above_bucket=avg([max(0,d['predicted']-d['minimum_bucket']) for d in ds]))
diagnostics={}
for trajectory in ['full','adaptive']:
    ds=[d for r in rows for d in r['diagnostics'][trajectory]]
    byq=[diagnose(r['diagnostics'][trajectory]) for r in rows]
    diagnostics[trajectory]=dict(scene_weighted=diagnose(ds),question_weighted={key:avg([q[key] for q in byq]) for key in byq[0] if key!='scenes'},
        by_layer={str(layer):diagnose([d for d in ds if d['layer']==layer]) for layer in range(32)})
results=dict(samples=50,methods=summary,adaptive_minus_fixed112=paired([r['f1'] for r in methods['adaptive']],[r['f1'] for r in methods['fixed112']]),
             fixed112_minus_full=paired([r['f1'] for r in methods['fixed112']],[r['f1'] for r in methods['full']]),diagnostics=diagnostics,
             caveats=['Frozen 50-question pilot, not official 200-question benchmark.','Diagnostic teacher probabilities computed explicitly in FP32 on each trajectory; group mean over Q heads.',
             'Teacher diagnostic labels are TEST ONLY and must not be used for training or threshold selection.','Fixed112 timing is a separate run; unfused prototype with all KV retained.'])
(ROOT/'results.json').write_text(json.dumps(results,indent=2))
with (ROOT/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['method']+list(summary['full']));writer.writeheader()
    writer.writerows(dict(method=k,**v) for k,v in summary.items())
with (ROOT/'per_question.csv').open('w',newline='') as f:
    fields=['source_index','truncated']+[m+'_f1' for m in methods]+[m+'_prediction' for m in methods]
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for i,r in enumerate(rows):
        writer.writerow(dict(source_index=r['source_index'],truncated=r['truncated'],**{m+'_f1':rs[i]['f1'] for m,rs in methods.items()},**{m+'_prediction':rs[i]['prediction'] for m,rs in methods.items()}))
text=['Frozen Llama HotpotQA 50: fixed112 and real decoding teacher diagnostics','',json.dumps(summary,indent=2),
      'Adaptive minus fixed112: '+json.dumps(results['adaptive_minus_fixed112']),
      'Fixed112 minus full: '+json.dumps(results['fixed112_minus_full'])]
for t,d in diagnostics.items():text.extend([t+' question-weighted diagnostics:',json.dumps(d['question_weighted'],indent=2)])
text.extend(results['caveats'])
(ROOT/'report.txt').write_text('\n'.join(text))
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact_sha256.json']]
(ROOT/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('\n'.join(text))
