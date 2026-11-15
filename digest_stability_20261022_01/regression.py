import diagnose as d
import guarded_controller as guard
import gc,json,os,sys,time,hashlib,subprocess,statistics
import torch

@torch.inference_mode()
def run_one(model,p,l,rank,budget,eos,row,version):
    summaries,index_seconds=d.build(p,'svd' if version=='reference' else 'gram')
    cls=d.e.gpu.Controller if version=='reference' else guard.Controller
    cls.ratio=.25 if 'ratio' not in row else row['ratio']
    d.e.c.run.AdaptiveLlama=cls;d.e.c.run.DynamicCache=d.e.BufferCache
    d.e.baseline.compact_digest=d.function('svd' if version=='reference' else 'gram')
    result=d.e.c.run.decode(model,p,l,'adaptive',rank,budget,summaries,eos,max_new=row['max_new'],quantile=.95)
    result.pop('steps');result=d.e.clean(result);result['index_seconds']=index_seconds
    del summaries;gc.collect();torch.cuda.empty_cache()
    return result

def score(result,tok,row):
    prediction=d.e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
    result.update(prediction=prediction,score=d.e.official.metric_for(row['task'])([prediction],[row['answers']]))

@torch.inference_mode()
def worker(shard):
    model,tok,models,eos=d.e.load()
    rows=json.loads((d.PREVIOUS/'selected.json').read_text())
    previous={r['index']:r for f in d.PREVIOUS.glob('results_*.jsonl') for s in f.read_text().splitlines() for r in [json.loads(s)]}
    out=d.ROOT/f'regression_{shard}.jsonl'
    assert not out.exists()
    with out.open('w') as f:
        for i,row in enumerate(rows):
            if i%3!=shard:continue
            p,l,prefill=d.e.prefill(model,row['input_ids'])
            for ratio in [.1,.25]:
                for branch in ['S','A','B']:
                    rank,budget=models[branch]
                    result=run_one(model,p,l,rank,budget,eos,dict(row,ratio=ratio),'guard')
                    score(result,tok,row)
                    old=previous[i]['methods'][f'baseline/{branch}/{ratio:.2f}']
                    result['e2e_seconds']=prefill+result['index_seconds']+result['decode_seconds']
                    record=dict(index=i,task=row['task'],length=row['length'],branch=branch,ratio=ratio,
                        reference=old,guarded=result,tokens_equal=old['ids']==result['ids'])
                    f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                    print(i,branch,ratio,old['score'],result['score'],record['tokens_equal'],flush=True)
            del p,l;gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def fresh():
    rows=json.loads((d.e.SOURCE/'selected.json').read_text())
    old_ids={r['input_sha256'] for r in json.loads((d.PREVIOUS/'selected.json').read_text())}
    model,tok,models,eos=d.e.load();rank,budget=models['B']
    out=d.ROOT/'fresh.jsonl';assert not out.exists()
    with out.open('w') as f:
        for length in [8192,16384]:
            row=next(r for r in rows if r['task']=='cwe' and r['length']==length and r['input_sha256'] not in old_ids)
            p,l,prefill=d.e.prefill(model,row['input_ids'])
            record={k:v for k,v in row.items() if k!='input_ids'}
            record['methods']={}
            for version in ['reference','guarded']:
                result=run_one(model,p,l,rank,budget,eos,row,version)
                score(result,tok,row)
                result['e2e_seconds']=prefill+result['index_seconds']+result['decode_seconds']
                record['methods'][version]=result
            record['tokens_equal']=record['methods']['reference']['ids']==record['methods']['guarded']['ids']
            f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            print(length,{k:v['score'] for k,v in record['methods'].items()},record['tokens_equal'],flush=True)
            del p,l;gc.collect();torch.cuda.empty_cache()

def summarize():
    rows=[json.loads(s) for p in d.ROOT.glob('regression_*.jsonl') for s in p.read_text().splitlines()]
    assert len(rows)==54
    report=dict(n=len(rows),tokens_equal=sum(r['tokens_equal'] for r in rows),
        score_regressions=[{k:r[k] for k in ['index','task','length','branch','ratio']} for r in rows if r['guarded']['score']<r['reference']['score']],
        score_changes=sum(r['guarded']['score']!=r['reference']['score'] for r in rows),
        mean_scores={v:statistics.mean(r[v]['score'] for r in rows) for v in ['reference','guarded']},
        mean_decode_seconds={v:statistics.mean(r[v]['decode_seconds'] for r in rows) for v in ['reference','guarded']},
        mean_e2e_seconds={v:statistics.mean(r[v]['e2e_seconds'] for r in rows) for v in ['reference','guarded']},
        mean_refined_blocks=statistics.mean(r['guarded']['audit']['refined_blocks'] for r in rows))
    fresh=[json.loads(s) for s in (d.ROOT/'fresh.jsonl').read_text().splitlines()];assert len(fresh)==2
    report['fresh']=[dict(length=r['length'],tokens_equal=r['tokens_equal'],scores={k:v['score'] for k,v in r['methods'].items()}) for r in fresh]
    d.e.dump('regression_report.json',report)
    protected=json.loads((d.ROOT/'protocol.json').read_text())['protected_sha256']
    assert all(hashlib.sha256(open(p,'rb').read()).hexdigest()==h for p,h in protected.items())
    d.e.dump('status.json',dict(stage='complete',prior_results_unchanged=True))
    print(json.dumps(report),flush=True)

def launch():
    assert not (d.ROOT/'status.json').exists()
    protected=[d.PREVIOUS/'comparison.json',d.PREVIOUS/'fast_digest.py',d.PREVIOUS/'gpu_controller.py']+list(d.PREVIOUS.glob('results_*.jsonl'))
    d.e.dump('protocol.json',dict(score_guard=guard.Controller.score_guard,cdf_guard=guard.Controller.cdf_guard,
        evidence='All 54 existing frozen comparisons plus two new CWE prompts, B/25%, 8K and 16K. No prompt, answer or quota changes.',
        protected_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.ROOT.glob('*.py')}))
    d.e.dump('status.json',dict(stage='regression_running'));jobs=[]
    for gpu in range(4):
        args=['worker',str(gpu)] if gpu<3 else ['fresh']
        with (d.ROOT/f'regression_{gpu}.log').open('w') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u',__file__]+args,stdout=f,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))))
    codes=[p.wait() for p in jobs]
    if any(codes):
        d.e.dump('status.json',dict(stage='failed',codes=codes));raise RuntimeError(codes)
    summarize()

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
