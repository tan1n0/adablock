import gc, hashlib, importlib.util, json, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/adablock_sab90_20261019_01')
SOURCE=Path('/root/adablock_ruler_harder_20261022_01')
sys.path.insert(0,str(OLD))
import core as c
import torch
import baseline_controller as baseline
import gpu_controller as gpu
import fast_digest
from cache_buffer import BufferCache, reset_registry
spec=importlib.util.spec_from_file_location('official',SOURCE/'official.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
torch.set_num_threads(4)
ORIGINAL_DIGEST=baseline.compact_digest

def dump(name,value):c.dump(ROOT/name,value)
def clean(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

@torch.inference_mode()
def prefill(model,ids):
    reset_registry();cache=BufferCache(len(ids)+256)
    torch.cuda.synchronize();start=time.perf_counter()
    for offset in range(0,len(ids),1024):
        output=model.model(input_ids=torch.tensor([ids[offset:offset+1024]],device='cuda'),past_key_values=cache,use_cache=True)
    logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
    torch.cuda.synchronize()
    return tuple(cache.to_legacy_cache()),logits,time.perf_counter()-start

def build(pairs,version):
    torch.cuda.synchronize();start=time.perf_counter()
    d=c.rt.build_prefix_digests(pairs) if version!='gpu_digest' else fast_digest.build_prefix_digests(pairs)
    torch.cuda.synchronize()
    return d,time.perf_counter()-start

def decode(model,p,l,d,rank,budget,eos,version,ratio,cap,capture=False,ignore_eos=False):
    cls=baseline.Controller if version=='baseline' else gpu.Controller
    cls.ratio=ratio;c.run.AdaptiveLlama=cls;c.run.DynamicCache=BufferCache
    baseline.compact_digest=fast_digest.compact_digest if version=='gpu_digest' else ORIGINAL_DIGEST
    try:
        return c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=cap,quantile=.95,
                            capture=capture,ignore_eos=ignore_eos)
    finally:baseline.compact_digest=ORIGINAL_DIGEST

def load():
    model,tok,r,b,eos,_=c.run.load();del r,b
    models={n:c.load_pair(OLD/'branches'/n/'best_pair.pt','cuda')[:2] for n in ['S','A','B']}
    return model,tok,models,eos

def indices(obj):
    return [r['indices'] if isinstance(r['indices'],list) else
            [ix[valid].tolist() for ix,valid in zip(r['indices'],r['valid'])] for r in obj.selected]

@torch.inference_mode()
def preflight():
    model,tok,models,eos=load()
    row=json.loads((SOURCE/'smoke/selected.json').read_text())[0]
    p,l,_=prefill(model,row['input_ids'][:1024])
    d,_=build(p,'baseline');fast,fast_seconds=build(p,'gpu_digest')
    report={'branches':{},'digest_seconds':fast_seconds}
    for name,(rank,budget) in models.items():
        results={v:decode(model,p,l,fast if v=='gpu_digest' else d,rank,budget,eos,v,.25,18,True,True)
                 for v in ['baseline','gpu','gpu_digest']}
        original=results['baseline'];chosen=indices(original['_controller'])
        comparisons={}
        for version in ['gpu','gpu_digest']:
            result=results[version];selection=indices(result['_controller'])
            same=sum(a==b for a,b in zip(chosen,selection))
            comparisons[version]=dict(tokens_equal=original['ids']==result['ids'],
                selected_layers_equal=same,selected_layers_total=len(chosen),
                decode_seconds=result['decode_seconds'],digest_updates=result['audit']['digest_updates'])
            assert original['ids']==result['ids'],(name,version,'probe tokens changed')
            if version=='gpu':assert same==len(chosen),(name,'GPU decisions changed')
        report['branches'][name]=dict(baseline_seconds=original['decode_seconds'],comparisons=comparisons)
        del results,original,result
    dump('preflight.json',report);print(json.dumps(report),flush=True)

@torch.inference_mode()
def worker(shard):
    model,tok,models,eos=load()
    rows=json.loads((ROOT/'selected.json').read_text())
    # Warm model projections and SDPA before collecting any times.
    prefill(model,rows[0]['input_ids'][:128])
    path=ROOT/f'results_{shard}.jsonl'
    assert not path.exists()
    with path.open('w') as out:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            p,l,prefill_seconds=prefill(model,row['input_ids'])
            d,index_seconds=build(p,'baseline')
            record={k:v for k,v in row.items() if k!='input_ids'}
            record.update(index=i,prefill_seconds=prefill_seconds,index_seconds={'baseline':index_seconds,'gpu':index_seconds},methods={})
            full=c.run.decode(model,p,l,'full',None,None,None,eos,max_new=row['max_new'])
            prediction=official.normalize_prediction(tok.decode(full['ids'],skip_special_tokens=True),row['task'])
            record['full']=dict(ids=full['ids'],prediction=prediction,score=official.metric_for(row['task'])([prediction],[row['answers']]),decode_seconds=full['decode_seconds'])
            for version in ['baseline','gpu','gpu_digest']:
                if version=='gpu_digest':
                    del d;gc.collect();torch.cuda.empty_cache()
                    d,seconds=build(p,version);record['index_seconds'][version]=seconds
                combos=[(n,r) for r in [.10,.25] for n in ['S','A','B']]
                shift=i%len(combos);combos=combos[shift:]+combos[:shift]
                for name,ratio in combos:
                    rank,budget=models[name]
                    result=decode(model,p,l,d,rank,budget,eos,version,ratio,row['max_new'])
                    prediction=official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
                    result.pop('steps')
                    result.update(prediction=prediction,score=official.metric_for(row['task'])([prediction],[row['answers']]),
                                  e2e_seconds=prefill_seconds+record['index_seconds'][version]+result['decode_seconds'])
                    key=f'{version}/{name}/{ratio:.2f}';record['methods'][key]=clean(result)
                    dump(f'partial_{i}.json',record)
                    print(json.dumps(dict(index=i,version=version,branch=name,ratio=ratio,score=result['score'],seconds=result['decode_seconds'])),flush=True)
            out.write(json.dumps(record)+'\n');out.flush();os.fsync(out.fileno())
            del p,l,d,record,full;gc.collect();torch.cuda.empty_cache()

def summarize():
    rows=[json.loads(line) for f in ROOT.glob('results_*.jsonl') for line in f.read_text().splitlines()]
    assert len(rows)==9
    report={'n_prompts':len(rows),'n_outputs':sum(len(r['methods']) for r in rows),'comparisons':{}}
    for version in ['gpu','gpu_digest']:
        pairs=[(r['methods'][f'baseline/{name}/{ratio:.2f}'],r['methods'][f'{version}/{name}/{ratio:.2f}'])
               for r in rows for name in ['S','A','B'] for ratio in [.1,.25]]
        report['comparisons'][version]=dict(outputs_identical=sum(a['ids']==b['ids'] for a,b in pairs),n=len(pairs),
            score_before=statistics.mean(a['score'] for a,b in pairs),score_after=statistics.mean(b['score'] for a,b in pairs),
            score_changes=sum(a['score']!=b['score'] for a,b in pairs),
            decode_seconds_before=statistics.mean(a['decode_seconds'] for a,b in pairs),
            decode_seconds_after=statistics.mean(b['decode_seconds'] for a,b in pairs),
            weighted_ms_per_token_before=1000*sum(a['decode_seconds'] for a,b in pairs)/sum(a['decode_steps'] for a,b in pairs),
            weighted_ms_per_token_after=1000*sum(b['decode_seconds'] for a,b in pairs)/sum(b['decode_steps'] for a,b in pairs),
            e2e_seconds_before=statistics.mean(a['e2e_seconds'] for a,b in pairs),
            e2e_seconds_after=statistics.mean(b['e2e_seconds'] for a,b in pairs),
            truncated=sum(b['hit_generation_limit'] for a,b in pairs))
    report['by_length']={str(length):dict(index_seconds={v:statistics.mean(r['index_seconds'][v] for r in rows if r['length']==length)
                               for v in ['baseline','gpu','gpu_digest']}) for length in [8192,16384,32768]}
    dump('comparison.json',report)
    protected=json.loads((ROOT/'protocol.json').read_text())['protected_sha256']
    assert all(c.sha(Path(p))==value for p,value in protected.items()),'Protected source changed'
    dump('status.json',dict(stage='complete',old_sources_unchanged=True))
    print(json.dumps(report),flush=True)

def launch():
    assert not (ROOT/'status.json').exists()
    selected=SOURCE/'smoke/selected.json'
    (ROOT/'selected.json').write_bytes(selected.read_bytes())
    protected=[SOURCE/'status.json',SOURCE/'controller.py',SOURCE/'evaluate.py',SOURCE/'smoke_report.json',selected,
               OLD/'vendor/runtime.py',OLD/'vendor/run.py']+[OLD/'branches'/n/'best_pair.pt' for n in ['S','A','B']]
    dump('protocol.json',dict(versions=['baseline','gpu','gpu_digest'],ratios=[.1,.25],coverage_target=.9,budget_quantile=.95,
        frozen_checkpoints=True,official_prompts_and_caps_unchanged=True,n_prompts=9,
        note='Paired implementation regression on existing 9 official smoke prompts, not a statistical S/A/B superiority test. One host width scalar and existing SDPA mask branch remain. Capture disabled for all timed quality runs.',
        protected_sha256={str(p):c.sha(p) for p in protected},sources_sha256={p.name:c.sha(p) for p in ROOT.glob('*.py')}))
    dump('status.json',dict(stage='preflight'))
    with (ROOT/'preflight.log').open('w') as f:
        subprocess.run([sys.executable,'-u',__file__,'preflight'],stdout=f,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
    dump('status.json',dict(stage='paired_quality_running'))
    jobs=[]
    for shard in range(4):
        with (ROOT/f'worker_{shard}.log').open('w') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u',__file__,'worker',str(shard)],stdout=f,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard))))
    codes=[job.wait() for job in jobs]
    assert not any(codes),codes
    summarize()

if __name__=='__main__':
    try:
        if sys.argv[1]=='worker':worker(int(sys.argv[2]))
        else:globals()[sys.argv[1]]()
    except Exception:
        dump(f'error_{sys.argv[1]}.json',dict(traceback=traceback.format_exc()))
        if sys.argv[1]=='launch':dump('status.json',dict(stage='failed'))
        raise
