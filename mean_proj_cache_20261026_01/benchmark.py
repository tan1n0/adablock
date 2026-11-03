"""Isolated mean_proj cache experiment; old modules and checkpoints remain read-only."""
import os,sys,json,time,gc,inspect,hashlib,subprocess,statistics,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_autonomous_20261022_01')
sys.path.insert(0,str(SOURCE))
import experiment as e
import torch
ORIGINAL=e.baseline.forward_packed
source=inspect.getsource(ORIGINAL)
assert source.count('mp = model.mean_proj(means)')==1
namespace=dict(ORIGINAL.__globals__)
exec(source.replace('mp = model.mean_proj(means)','mp = p["cached_mean_proj"]'),namespace)
CACHED=namespace['forward_packed']

def dump(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(e.clean(obj),indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean():gc.collect();torch.cuda.empty_cache()

class ProjectionCache:
    @torch.inference_mode()
    def __init__(self,ranker,digests):
        self.prefix=[d['mean'].shape[1] for d in digests]
        self.buffers=[]
        for d in digests:
            h,n,dim=d['mean'].shape
            out=ranker.mean_proj(d['mean'].reshape(-1,dim)).reshape(h,n,-1)
            buf=out.new_empty((h,n+9,out.shape[-1]))
            buf[:,:n].copy_(out);self.buffers.append(buf)
        self.bytes=sum(t.numel()*t.element_size() for t in self.buffers)
        self.prefix_bytes=sum(t.shape[0]*n*t.shape[2]*t.element_size() for t,n in zip(self.buffers,self.prefix))

class Controller(e.gpu.Controller):
    cache=None
    audit_mode=False
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.valid=list(self.cache.prefix) if self.cache else None
        self.trace=[];self.score_error=0.;self.projection_error=0.;self.incremental_blocks=0

    def choose(self,layer,query,scale):
        if self.cache:
            means=self.digests[layer]['mean'];n=means.shape[1];start=self.valid[layer]
            if n>start:
                assert n<=self.cache.buffers[layer].shape[1]
                h,_,dim=means.shape
                out=self.ranker.mean_proj(means[:,start:n].reshape(-1,dim)).reshape(h,n-start,-1)
                self.cache.buffers[layer][:,start:n].copy_(out)
                self.valid[layer]=n;self.incremental_blocks+=n-start
            def scorer(model,p):
                p['cached_mean_proj']=self.cache.buffers[layer][p['scene_ids'],p['block_ids']]
                score=CACHED(model,p)
                if self.audit_mode:
                    ref=ORIGINAL(model,p)
                    self.score_error=max(self.score_error,float((score-ref).abs().max()))
                    mp=model.mean_proj(p['mean'].float())
                    self.projection_error=max(self.projection_error,float((mp-p['cached_mean_proj']).abs().max()))
                return score
            e.baseline.forward_packed=scorer
        try:result=super().choose(layer,query,scale)
        finally:e.baseline.forward_packed=ORIGINAL
        if self.audit_mode:
            ix,valid=result
            digest=hashlib.sha256(ix.cpu().numpy().tobytes()+valid.cpu().numpy().tobytes()).hexdigest()
            self.trace.append((self.position,layer,digest,self.step_rows[-1]['classes'].cpu().tolist()))
        return result

@torch.inference_mode()
def worker(shard):
    torch.set_num_threads(4)
    model,tok,models,eos=e.load()
    e.c.run.DynamicCache=e.BufferCache;e.c.run.AdaptiveLlama=Controller
    e.baseline.compact_digest=e.fast_digest.compact_digest
    rows=json.loads((ROOT/'selected.json').read_text())
    for i,row in enumerate(rows):
        if i%4!=shard:continue
        p,l,prefill=e.prefill(model,row['input_ids']);d,summary_seconds=e.build(p,'gpu_digest')
        for name in ['S','B']:
            rank,budget=models[name]
            record=dict(index=i,task=row['task'],length=row['length'],prompt_tokens=len(row['input_ids']),branch=name,
                prefill_seconds=prefill,summary_seconds=summary_seconds,times={'baseline':[],'cached':[]},memory={'baseline':[],'cached':[]},builds=[])
            def run(cap,ignore=True,audit=False):
                Controller.audit_mode=audit
                return e.c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=cap,ignore_eos=ignore,capture=True,quantile=.95)
            def prepare(version):
                Controller.cache=None;clean()
                if version=='cached':
                    before=torch.cuda.memory_allocated();torch.cuda.synchronize();start=time.perf_counter()
                    Controller.cache=ProjectionCache(rank,d)
                    torch.cuda.synchronize()
                    record['builds'].append(dict(seconds=time.perf_counter()-start,tensor_bytes=Controller.cache.bytes,
                        prefix_bytes=Controller.cache.prefix_bytes,allocated_delta_bytes=torch.cuda.memory_allocated()-before))
            # Alternate order; actual decode timing excludes cache construction and validation.
            for repeat in range(3):
                for version in (['baseline','cached'] if repeat%2==0 else ['cached','baseline']):
                    prepare(version)
                    warm=run(3);del warm;gc.collect();torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
                    result=run(33)
                    assert result['decode_steps']==32
                    record['times'][version].append(result['decode_seconds']/32)
                    record['memory'][version].append(dict(resident_bytes=resident,peak_bytes=torch.cuda.max_memory_allocated()))
                    del result
            audits={};answers={}
            for version in ['baseline','cached']:
                prepare(version)
                result=run(33,audit=True);obj=result.pop('_controller')
                audits[version]=dict(ids=result['ids'],trace=obj.trace,score_error=obj.score_error,
                    projection_error=obj.projection_error,incremental_blocks=obj.incremental_blocks,digest_updates=obj.audit['digest_updates'])
                del result,obj
                result=run(row['max_new'],ignore=False);obj=result.pop('_controller');del obj
                prediction=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
                answers[version]=dict(ids=result['ids'],prediction=prediction,score=e.official.metric_for(row['task'])([prediction],[row['answers']]),
                    decode_steps=result['decode_steps'],hit_generation_limit=result['hit_generation_limit'])
                del result
            record['validation']=dict(fixed_tokens_equal=audits['baseline']['ids']==audits['cached']['ids'],
                selected_indices_and_classes_equal=audits['baseline']['trace']==audits['cached']['trace'],
                compared_layer_steps=len(audits['baseline']['trace']),score_max_abs_error=audits['cached']['score_error'],
                projection_max_abs_error=audits['cached']['projection_error'],incremental_layer_blocks=audits['cached']['incremental_blocks'],
                digest_updates=audits['cached']['digest_updates'],answer_tokens_equal=answers['baseline']['ids']==answers['cached']['ids'])
            record['answers']=answers
            record['median_ms']={v:1000*statistics.median(t) for v,t in record['times'].items()}
            record['speedup']=record['median_ms']['baseline']/record['median_ms']['cached']
            dump(f'case_{i}_{name}.json',record)
            assert all(record['validation'][k] for k in ['fixed_tokens_equal','selected_indices_and_classes_equal','answer_tokens_equal']),record['validation']
            assert record['validation']['incremental_layer_blocks']>=32,'Incremental update path not exercised'
            print(json.dumps(dict(case=i,branch=name,ms=record['median_ms'],speedup=record['speedup'],cache_MiB=record['builds'][0]['tensor_bytes']/2**20)),flush=True)
            Controller.cache=None;del record,audits,answers;clean()
        del p,l,d;e.reset_registry();clean()

def summarize():
    rows=[json.loads(p.read_text()) for p in ROOT.glob('case_*.json')]
    report={'completed':len(rows),'expected':18,'by_length':{}}
    for length in [8192,16384,32768]:
        rr=[r for r in rows if r['length']==length]
        if not rr:continue
        base=statistics.mean(r['median_ms']['baseline'] for r in rr);cached=statistics.mean(r['median_ms']['cached'] for r in rr)
        report['by_length'][length]=dict(n=len(rr),baseline_ms=base,cached_ms=cached,speedup=base/cached,reduction_pct=100*(1-cached/base),
            cache_MiB=statistics.mean(r['builds'][0]['tensor_bytes']/2**20 for r in rr),
            allocated_delta_MiB=statistics.mean(r['builds'][0]['allocated_delta_bytes']/2**20 for r in rr),
            peak_increase_MiB=statistics.mean((statistics.median(x['peak_bytes'] for x in r['memory']['cached'])-statistics.median(x['peak_bytes'] for x in r['memory']['baseline']))/2**20 for r in rr),
            cache_build_ms=statistics.mean(statistics.median(x['seconds'] for x in r['builds'])*1000 for r in rr))
    report['validation_passed']=all(all(r['validation'][k] for k in ['fixed_tokens_equal','selected_indices_and_classes_equal','answer_tokens_equal']) for r in rows)
    report['score_max_abs_error']=max((r['validation']['score_max_abs_error'] for r in rows),default=0)
    dump('summary.json',report)
    return len(rows)==18 and report['validation_passed']

def launch():
    assert not (ROOT/'status.json').exists()
    allrows=json.loads((SOURCE/'selected.json').read_text());seen=set();rows=[]
    for r in allrows:
        key=(r['task'],r['length'])
        if key not in seen:rows.append(r);seen.add(key)
    assert len(rows)==9;dump('selected.json',rows)
    paths=[SOURCE/'gpu_controller.py',SOURCE/'fast_digest.py',SOURCE/'comparison.json',SOURCE/'selected.json',Path(inspect.getfile(ORIGINAL))]
    paths += [e.OLD/'branches'/n/'best_pair.pt' for n in ['S','B']]
    hashes={str(p):sha(p) for p in paths}
    dump('protocol.json',dict(change='Cache only frozen mean_proj outputs; original features, sorting and budget policy unchanged',
        tasks=sorted({r['task'] for r in rows}),lengths=[8192,16384,32768],branches=['S','B'],fixed_decode_steps=32,repetitions=3,
        cache_dtype='float32',reserve_blocks=9,timing='alternating paired order, one worker per otherwise idle GPU; cache build reported separately',
        validation='32-step full selection/budget traces, incremental cache updates, natural full answers; no validation overhead in timed runs',
        protected_sha256=hashes,source_sha256=sha(__file__)))
    jobs=[]
    for shard in range(4):
        with (ROOT/f'worker_{shard}.log').open('w') as log:
            jobs.append(subprocess.Popen([sys.executable,'-u',__file__,'worker',str(shard)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT))
    dump('status.json',dict(stage='running',pids=[j.pid for j in jobs]))
    while any(j.poll() is None for j in jobs):
        if any(j.poll() not in [None,0] for j in jobs):
            for j in jobs:
                if j.poll() is None:j.terminate()
            raise RuntimeError('worker failed; remaining workers stopped')
        summarize();time.sleep(20)
    assert all(j.returncode==0 for j in jobs)
    assert summarize()
    assert all(sha(p)==h for p,h in hashes.items())
    dump('status.json',dict(stage='complete',old_files_unchanged=True))

if __name__=='__main__':
    try:
        if sys.argv[1]=='launch':launch()
        elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
        else:summarize()
    except Exception:
        dump('error_'+sys.argv[1]+'.json',dict(error=traceback.format_exc()))
        if sys.argv[1]=='launch':dump('status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
