"""Additional descriptive diagnostics; no threshold fitting or training."""
import csv
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
rows=sorted([json.loads(s) for f in p.glob('worker_*.jsonl') for s in f.read_text().splitlines()],key=lambda r:r['source_index'])
assert len(rows)==50
out=[]
for r in rows:
    for trajectory,ds in r['diagnostics'].items():
        record=dict(source_index=r['source_index'],trajectory=trajectory,truncated=r['truncated'],scenes=len(ds))
        for k in ['predicted','argmax','required','minimum_bucket','oracle_required','mass_predicted','mass112']:
            record[k]=float(np.mean([d[k] for d in ds]))
        for k in ['mass_predicted','mass_argmax','mass32','mass64','mass112']:
            record[k+'_hit95']=float(np.mean([d[k]>=.95 for d in ds]))
        out.append(record)
with (p/'per_question_diagnostics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
results={}
for trajectory in ['full','adaptive']:
    ds=[d for r in rows for d in r['diagnostics'][trajectory]]
    results[trajectory]=dict(
        budget_quantiles={k:np.quantile([d[k] for d in ds],[0,.25,.5,.75,.9,.95,.99,1]).tolist() for k in ['required','minimum_bucket','predicted','argmax']},
        first_decode={k:float(np.mean([d[k] for r in rows for d in r['diagnostics'][trajectory] if d['position']==r['diagnostics'][trajectory][0]['position']])) for k in ['required','minimum_bucket','predicted','argmax','mass_predicted','mass112']})
(p/'diagnostic_details.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
