import json,hashlib,zipfile,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/adablock')
from scripts.build_strict_training_corpus import sampled_ngram_hashes
wiki=[json.loads(l) for p in sorted(ROOT.glob('worker_*.jsonl')) for l in p.read_text().splitlines()]
hotpot=[json.loads(l) for p in sorted(ROOT.glob('hotpot_check_*.jsonl')) for l in p.read_text().splitlines()]
assert len(wiki)==24 and len({r['raw_id'] for r in wiki})==24
assert len(hotpot)==20 and len({r['id'] for r in hotpot})==20
assert all(r['split']!='test' for r in wiki+hotpot)
selected=json.loads((ROOT/'hotpot_selected.json').read_text())
fps={s:set().union(*(set(sampled_ngram_hashes(r['context'],13,5)) for r in selected if r['split']==s)) for s in ['budget_train','validation','test']}
overlap={a+'__'+b:len(fps[a]&fps[b]) for a,b in [('budget_train','validation'),('budget_train','test'),('validation','test')]}
summary=json.loads((ROOT/'summary.json').read_text())
for s in summary:
    delta=[r['methods']['candidate_aware']['f1']-r['methods']['open']['f1'] for r in wiki if r['split']==s]
    summary[s]['paired']=dict(wins=sum(d>0 for d in delta),ties=sum(d==0 for d in delta),losses=sum(d<0 for d in delta))
hp={s:dict(n=len(rr),f1=sum(r['f1'] for r in rr)/len(rr),hit_generation_limit=sum(r['hit_generation_limit'] for r in rr)) for s in ['budget_train','validation'] if (rr:=[r for r in hotpot if r['split']==s])}
result=dict(wikihop=summary,hotpot=hp,hotpot_shared_fingerprints=overlap,hotpot_data=json.loads((ROOT/'hotpot_audit.json').read_text()),validation=json.loads((ROOT/'validation.json').read_text()),conclusion='Protocol audit only, no model training. Native Hotpot training cannot supply 4-8K contexts. Candidate-aware WikiHop helps but does not resolve low accuracy. Do not select data based on these output scores.')
(ROOT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
with zipfile.ZipFile(ROOT/'audit_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.iterdir()):
        if p.suffix in ['.py','.json','.jsonl','.log']:z.write(p,p.name)
files=['results.json','summary.json','hotpot_audit.json','validation.json','protocol.json','audit_bundle.zip']
(ROOT/'artifact_sha256.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2))
print(json.dumps(result,indent=2))
