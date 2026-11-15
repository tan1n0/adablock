import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[json.loads(s) for p in root.glob('regression_*.jsonl') for s in p.read_text().splitlines()]
print('completed',len(rows),'of 54','tokens_equal',sum(r['tokens_equal'] for r in rows),
      'score_regressions',sum(r['guarded']['score']<r['reference']['score'] for r in rows))
for p in [root/'status.json',root/'fresh.jsonl']:
    if p.exists():
        if p.suffix=='.json':print(p.read_text())
        else:
            for s in p.read_text().splitlines():
                r=json.loads(s);print('fresh',r['length'],r['tokens_equal'],{k:v['score'] for k,v in r['methods'].items()})
