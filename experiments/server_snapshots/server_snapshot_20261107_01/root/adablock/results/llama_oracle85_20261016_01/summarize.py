import json,hashlib,csv,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
def mean(x):return float(np.mean(x))
def load(root,pattern):return {r['source_index']:r for p in root.glob(pattern) for l in p.read_text().splitlines() if (r:=json.loads(l))}
def costs(ds,selection):
    blocks=[];logical=[];padded=[];total_logical=0;total_padded=0
    for r in ds:
        takes=r[selection];lengths=[r['fixed_tokens']+16*k for k in takes]
        blocks.extend(takes);logical.append(mean(lengths)/(r['position']+1));padded.append(max(lengths)/(r['position']+1))
        total_logical+=sum(lengths);total_padded+=len(lengths)*max(lengths)
    return dict(blocks=mean(blocks),logical_fraction=mean(logical),padded_fraction=mean(padded),padding_extra_fraction=mean(padded)-mean(logical),total_logical_token_slots=total_logical,total_padded_token_slots=total_padded,padding_wasted_slots=total_padded-total_logical)
def main():
    rows=load(ROOT,'worker_*.jsonl');assert len(rows)==50
    original=load(BASE,'worker_*.jsonl');selected={r['source_index']:r for r in json.loads((BASE/'selected.json').read_text())}
    summary=dict(n=50,arms={},same_state_costs={},paired={},changes=[],validation={})
    perquestion=[];methods={};replays=0;scenes=0
    for arm in ['oracle_exact','oracle_bucket']:
        entries=[];counters={'exact':[],'bucket':[]}
        for i,r in rows.items():
            assert r['input_ids_sha256']==selected[i]['input_ids_sha256']==original[i]['input_ids_sha256']
            o=r['methods'][arm];ds=o['diagnostics'];assert len(ds)==o['decode_steps']*32
            assert o['ids'][0]==original[i]['methods']['full']['ids'][0]
            assert all(v==0 for v in o['audit']['budget_calls'])
            cov=[v for d in ds for v in d['coverage']];assert min(cov)>=.85-1e-7
            assert all(a<=b for d in ds for a,b in zip(d['exact'],d['bucket']))
            assert all(d['selected']==d['exact' if arm=='oracle_exact' else 'bucket'] for d in ds)
            scenes+=len(cov);replays+=bool(o.get('teacher_free_replay_parity'))
            c=costs(ds,'selected')
            assert abs(c['blocks']-o['mean_middle_blocks'])<1e-6
            assert abs(c['logical_fraction']-o['mean_read_fraction'])<1e-6
            entry=dict(source_index=i,arm=arm,truncated=r['truncated'],prompt_tokens=r['prompt_tokens'],f1=o['f1'],prediction=o['prediction'],decode_steps=o['decode_steps'],hit_generation_limit=o['hit_generation_limit'],mean_coverage=mean(cov),min_coverage=min(cov),**c)
            entries.append(entry);perquestion.append(entry)
            for name in counters:counters[name].append(costs(ds,name))
        numeric=['f1','decode_steps','mean_coverage','blocks','logical_fraction','padded_fraction','padding_extra_fraction','total_logical_token_slots','total_padded_token_slots','padding_wasted_slots']
        summary['arms'][arm]={k:mean([r[k] for r in entries]) for k in numeric}
        summary['arms'][arm]['hit_generation_limit_count']=sum(r['hit_generation_limit'] for r in entries)
        summary['arms'][arm]['minimum_coverage']=min(r['min_coverage'] for r in entries)
        summary['same_state_costs'][arm]={name:{k:mean([r[k] for r in rr]) for k in rr[0]} for name,rr in counters.items()}
    exact=summary['arms']['oracle_exact'];bucket=summary['arms']['oracle_bucket']
    summary['savings_exact_vs_bucket']={k:dict(absolute_reduction=bucket[k]-exact[k],relative_reduction=1-exact[k]/bucket[k]) for k in ['blocks','logical_fraction','padded_fraction','padding_extra_fraction','total_logical_token_slots','total_padded_token_slots','padding_wasted_slots']}
    for trajectory,c in summary['same_state_costs'].items():
        c['relative_savings']={k:1-c['exact'][k]/c['bucket'][k] for k in c['exact'] if c['bucket'][k]}
    deltas=np.array([r['methods']['oracle_exact']['f1']-r['methods']['oracle_bucket']['f1'] for r in rows.values()])
    rng=np.random.default_rng(20261016);boot=rng.choice(deltas,size=(20000,len(deltas)),replace=True).mean(1)
    summary['paired']=dict(f1_delta_exact_minus_bucket=float(deltas.mean()),wins=int((deltas>1e-7).sum()),ties=int((abs(deltas)<=1e-7).sum()),losses=int((deltas<-1e-7).sum()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),identical_generated_tokens=sum(r['methods']['oracle_exact']['ids']==r['methods']['oracle_bucket']['ids'] for r in rows.values()))
    for i,r in sorted(rows.items()):
        a=r['methods']['oracle_exact'];b=r['methods']['oracle_bucket']
        if a['ids']!=b['ids']:
            summary['changes'].append(dict(source_index=i,question=selected[i]['question'],answers=r['answers'],exact={k:a[k] for k in ['prediction','f1','ids']},bucket={k:b[k] for k in ['prediction','f1','ids']},full={k:original[i]['methods']['full'][k] for k in ['prediction','f1']}))
    summary['full_reference_f1']=mean([r['methods']['full']['f1'] for r in original.values()])
    summary['subgroups']={}
    for truncated in [False,True]:
        summary['subgroups']['truncated' if truncated else 'untruncated']={}
        for arm in ['oracle_exact','oracle_bucket']:
            rr=[r for r in perquestion if r['truncated']==truncated and r['arm']==arm]
            summary['subgroups']['truncated' if truncated else 'untruncated'][arm]=dict(n=len(rr),**{k:mean([r[k] for r in rr]) for k in ['f1','blocks','logical_fraction','padded_fraction']})
    for path,digest in json.loads((ROOT/'protocol.json').read_text())['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert replays==16,replays
    summary['validation']=dict(frozen_hashes_unchanged=True,shared_first_tokens=100,teacher_free_replays=replays,coverage_scenes_checked=scenes,no_budget_model_calls=True,no_training=True)
    summary['limitations']=['Oracle budget uses full QK attention and is not deployable.','Per-question equal-weight means; independent trajectories can have different hidden states and output lengths.','Same-state counterfactual cost separates rounding overhead from trajectory changes.','Token-slot padding is a computation proxy, not measured memory traffic or speedup.','The reused Hotpot50 is a regression/diagnostic set, not a fresh final benchmark.']
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (ROOT/'per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(perquestion[0]));w.writeheader();w.writerows(perquestion)
    with zipfile.ZipFile(ROOT/'reproduction_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in ['.py','.json','.jsonl','.csv','.log']:z.write(p,p.name)
    files=['results.json','protocol.json','per_question.csv','reproduction_bundle.zip']
    (ROOT/'artifact_sha256.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
