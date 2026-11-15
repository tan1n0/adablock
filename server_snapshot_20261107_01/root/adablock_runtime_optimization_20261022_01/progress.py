import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
print('status', (root/'status.json').read_text(), (root/'verification_status.json').read_text())
rows=[json.loads(p.read_text()) for p in root.glob('partial_*.json')]
for version in ['gpu','gpu_digest']:
    pairs=[(r['methods']['baseline/'+key.split('/',1)[1]], value) for r in rows for key,value in r['methods'].items()
           if key.startswith(version+'/') and 'baseline/'+key.split('/',1)[1] in r['methods']]
    if pairs:
        print(version,json.dumps(dict(n=len(pairs),same_tokens=sum(a['ids']==b['ids'] for a,b in pairs),
              score_changes=sum(a['score']!=b['score'] for a,b in pairs),
              decode_speedup=sum(a['decode_seconds'] for a,b in pairs)/sum(b['decode_seconds'] for a,b in pairs))))
print('completed_outputs',sum(len(r['methods']) for r in rows),'of',162)
for r in sorted(rows,key=lambda r:r['index']):print('prompt',r['index'],r['length'],len(r['methods']),r['index_seconds'])
for p in root.glob('error_*.json'): print(p.name,p.read_text()[-1500:])
