import json,sys,hashlib,csv,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
OLD=Path('/root/adablock/results/llama_hotpot50_fixed112_diagnostics_20261015_01')
QA=Path('/root/adablock/results/llama_qa_budget_20261015_01')
def read(root,pat):
    return {x['source_index']:x for p in root.glob(pat) for l in p.read_text().splitlines() if (x:=json.loads(l))}
def mean(x):return float(np.mean(x))
def metrics(rows):
    return dict(scenes=len(rows),blocks=mean([r['selected'] for r in rows]),coverage=mean([r['coverage'] for r in rows]),hit95=mean([r['coverage']>=.95-1e-7 for r in rows]),under=mean([r['selected']<r['required'] for r in rows]),over_bucket=mean([r['selected']>r['minimum_bucket'] for r in rows]),minimum_bucket=mean([r['minimum_bucket'] for r in rows]),required=mean([r['required'] for r in rows]),ideal_required=mean([r['oracle_required'] for r in rows]),positive_excess=mean([max(0,r['selected']-r['required']) for r in rows]),positive_shortfall=mean([max(0,r['required']-r['selected']) for r in rows]),fixed112_hit_on_same_states=mean([r['fixed112_coverage']>=.95-1e-7 for r in rows]),dynamic_hit_on_same_states=mean([r['dynamic_coverage']>=.95-1e-7 for r in rows]))
def padded(out):
    if not out['steps']:return 0
    # 256 per-step selections = 32 layers x 8 heads; each layer pads to its largest head.
    diag=out['diagnostics'];groups=defaultdict(list)
    for r in diag:groups[(r['position'],r['layer'])].append(r)
    return mean([(max(r['selected'] for r in rr)*16+(pos+1)-rr[0]['candidates']*16)/(pos+1) for (pos,layer),rr in groups.items()])

def main():
    rows=read(ROOT,'worker_*.jsonl');assert len(rows)==50,len(rows)
    old=read(BASE,'worker_*.jsonl');selected={r['source_index']:r for r in json.loads((BASE/'selected.json').read_text())}
    report=dict(question_count=50,methods={},diagnostics={},cases={},hashes_verified=False)
    per_question=[];layer_rows=[]
    for mode in ['full','fixed32','fixed64','fixed112','old_adaptive','adaptive']:
        outputs=[r['methods'][mode] if mode in ['adaptive','fixed112'] else old[i]['methods']['adaptive' if mode=='old_adaptive' else mode] for i,r in rows.items()]
        report['methods'][mode]=dict(f1=mean([o['f1'] for o in outputs]),blocks=mean([o['mean_middle_blocks'] for o in outputs]),logical_read=mean([o['mean_read_fraction'] for o in outputs]))
        if mode in ['adaptive','fixed112']:report['methods'][mode]['padded_read']=mean([padded(o) for o in outputs])
    for mode in ['adaptive','fixed112']:
        allrows=[];perdoc=[];layers=defaultdict(list);steps=defaultdict(list)
        for i,r in rows.items():
            out=r['methods'][mode];assert out['original_generation_parity']
            ds=out['diagnostics'];allrows.extend(ds);m=metrics(ds);perdoc.append(m)
            per_question.append(dict(source_index=i,mode=mode,f1=out['f1'],prediction=out['prediction'],**m))
            for d in ds:
                layers[d['layer']].append(d)
                step=d['position']-selected[i]['prompt_tokens'];steps['first' if step==0 else 'later'].append(d)
        layer_metrics={str(k):metrics(v) for k,v in layers.items()}
        for layer,m in layer_metrics.items():layer_rows.append(dict(mode=mode,layer=int(layer),**m))
        report['diagnostics'][mode]=dict(scene_weighted=metrics(allrows),question_weighted={k:mean([d[k] for d in perdoc]) for k in perdoc[0] if k!='scenes'},layers=layer_metrics,decode_stage={k:metrics(v) for k,v in steps.items()})
    delta=np.array([r['methods']['adaptive']['f1']-r['methods']['fixed112']['f1'] for r in rows.values()])
    rng=np.random.default_rng(20261016);samples=rng.choice(delta,size=(20000,len(delta)),replace=True).mean(1)
    report['paired_adaptive_minus_fixed112']=dict(delta_f1=float(delta.mean()),wins=int((delta>1e-7).sum()),ties=int((abs(delta)<=1e-7).sum()),losses=int((delta<-1e-7).sum()),bootstrap95=np.quantile(samples,[.025,.975]).tolist())
    for i in [16,121,160,165,181]:
        r=rows[i];case=dict(question=selected[i].get('question'),answers=selected[i]['answers'],methods={})
        for mode,out in r['methods'].items():
            case['methods'][mode]=dict(prediction=out['prediction'],f1=out['f1'],blocks=out['mean_middle_blocks'],diagnostics=metrics(out['diagnostics']))
        case['methods']['full']={k:old[i]['methods']['full'][k] for k in ['prediction','f1']}
        a=r['methods']['adaptive']['ids'];b=r['methods']['fixed112']['ids']
        divergence=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        case['first_output_token_divergence_index']=divergence
        if divergence is not None and divergence>0:
            pos=selected[i]['prompt_tokens']+divergence-1
            dd=[d for d in r['methods']['adaptive']['diagnostics'] if d['position']==pos]
            case['at_first_divergence']=dict(position=pos,summary=metrics(dd),worst_heads=sorted(dd,key=lambda d:d['coverage'])[:5])
        report['cases'][str(i)]=case
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for path,digest in protocol['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    report['hashes_verified']=True
    report['generation_parity_checks']=100
    localization=json.loads((ROOT/'localization.json').read_text())
    report['localization']={k:v for k,v in localization.items() if k!='results'}
    report['localization']['results']=[{k:v for k,v in r.items() if k!='first_step_rows'} for r in localization['results']]
    report['token_margin']=json.loads((ROOT/'token_margin.json').read_text())
    report['limits']=['95% average-over-Q-head attention mass is a diagnostic proxy, not answer correctness.','Same-state counterfactual budget coverage does not replay later hidden states.','Oracle repair uses full QK information and is not a deployable method.','Intervention set selected by observed disagreements, not an unbiased test.','Instrumented timings are not performance benchmarks; all KV remains stored.']
    (ROOT/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    for name,rr in [('per_question.csv',per_question),('per_layer.csv',layer_rows)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    with zipfile.ZipFile(ROOT/'diagnostics_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in ['.py','.json','.jsonl','.csv','.log']:z.write(p,p.name)
    files=['results.json','protocol.json','per_question.csv','per_layer.csv','localization.json','token_margin.json','diagnostics_bundle.zip']
    (ROOT/'artifact_sha256.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2))
    print(json.dumps(dict(methods=report['methods'],diagnostics={k:v['question_weighted'] for k,v in report['diagnostics'].items()},cases={k:{m:(v['f1'],v['prediction']) for m,v in c['methods'].items()} for k,c in report['cases'].items()}),indent=2))
if __name__=='__main__':main()
