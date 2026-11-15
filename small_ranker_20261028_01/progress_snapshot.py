"""Read-only progress and length-aware ETA; no GPU work."""
import json,time,statistics,collections,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
now=time.time();state=json.loads((ROOT/'status.json').read_text())
inputs=json.loads((ROOT/'selected.json').read_text())
fixed=set(json.loads((ROOT/'selection.json').read_text())['fixed'])
rows={}
for p in (ROOT/'evaluation').glob('partial_*.json'):
    x=json.loads(p.read_text());x['_mtime']=p.stat().st_mtime;rows[x['index']]=x
complete={i:x for i,x in rows.items() if len(x['methods'])==(19 if i in fixed else 7)}
smoke={}
for p in (ROOT/'smoke').glob('partial_*.json'):
    x=json.loads(p.read_text());v=x['methods']
    base=max(a['prefill_seconds']+a['index_seconds'] for a in v.values())
    auto=base+sum(a['decode_seconds'] for k,a in v.items() if '/' not in k)
    extra=sum(a['decode_seconds'] for k,a in v.items() if '/' in k)
    smoke[(x['task'],x['length'],False)]=auto
    smoke[(x['task'],x['length'],True)]=auto+extra
observed=collections.defaultdict(list)
for i,x in complete.items():
    v=x['methods'];cost=max(a['prefill_seconds']+a['index_seconds'] for a in v.values())+sum(a['decode_seconds'] for a in v.values())
    observed[(x['task'],x['length'],i in fixed)].append(cost)
start=min((x['_mtime'] for x in rows.values()),default=now)
elapsed=max(1,now-start)
measured=sum(sum(v) for v in observed.values())
overhead=max(1.,min(1.3,2*elapsed/max(1,measured)))
def estimate(i,x):
    key=(x['task'],x['length'],i in fixed)
    if len(observed[key])>=3:return statistics.mean(observed[key])
    # For unseen lengths, use smoke costs, corrected by observed task costs.
    ratios=[]
    for k,v in observed.items():
        if k[0]==key[0] and k[2]==key[2] and len(v)>=3:
            ratios.append(statistics.mean(v)/smoke[k])
    return smoke[key]*(statistics.mean(ratios) if ratios else 1.)
remaining=[]
for shard in range(2):
    remaining.append(overhead*sum(estimate(i,x) for i,x in enumerate(inputs) if i%2==shard and i not in complete))
eta=max(remaining) if len(complete)<450 else 0
names=['original_S','original_B','H64_S','H64_B','P32H64_S','P32H64_B']
stats={}
for name in names:
    v=[x['methods'][name] for x in complete.values()]
    if v:stats[name]=dict(n=len(v),score=statistics.mean(x['score'] for x in v),read_pct=100*statistics.mean(x['logical_fraction'] for x in v),
        ms_per_token=1000*sum(x['decode_seconds'] for x in v)/sum(x['decode_steps'] for x in v))
latest=max((x['_mtime'] for x in rows.values()),default=now)
out=dict(updated_unix=now,stage=state['stage'],completed=len(complete),total=450,started=len(rows),
    answers_done=sum(len(x['methods']) for x in rows.values()),answers_total=4230,
    by_length={str(L):sum(x['length']==L for x in complete.values()) for L in [8192,16384,32768]},
    eta_remaining_seconds=eta,eta_unix=now+eta,eta_low_unix=now+eta*.8,eta_high_unix=now+eta*1.25,
    eta_basis='Completed QA costs by task/length/fixed-budget group; unseen groups estimated from matched smoke cases; two-worker imbalance included.',
    last_result_age_seconds=now-latest,warning='结果已超过10分钟没有更新' if state['stage']=='evaluation' and now-latest>600 else None,
    cap_hits=sum(v['hit_generation_limit'] for x in rows.values() for v in x['methods'].values()),
    stats=stats,error=state.get('error'),old_files_unchanged=state.get('old_files_unchanged'))
print(json.dumps(out,ensure_ascii=False))
