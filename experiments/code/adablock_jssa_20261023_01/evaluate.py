import os,sys,json,time,gc,hashlib,subprocess,statistics,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_autonomous_20261022_01')
sys.path.insert(0,str(SOURCE))
import experiment as e
import torch
from jssa_adapter import JSSA
BUDGETS=[512,1024,2048,4096]
METHODS=['full','S','B']+[f'JSSA_{b}' for b in BUDGETS]

def dump(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(e.clean(obj),indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup():
    e.c.run.DynamicCache=e.BufferCache;e.c.run.AdaptiveLlama=e.gpu.Controller
    e.baseline.compact_digest=e.fast_digest.compact_digest
    return e.load()
def clear():
    e.reset_registry();gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def preflight():
    model,tok,models,eos=setup();basis=torch.load(ROOT/'basis.pt',map_location='cpu',weights_only=False)['M']
    rows=json.loads((ROOT/'selected.json').read_text());row=max(rows,key=lambda x:len(x['input_ids']))
    p,l,_=e.prefill(model,row['input_ids'])
    ref=e.c.run.decode(model,p,l,'full',None,None,None,eos,max_new=3,ignore_eos=True,return_logits=True)
    cpu=[(k.cpu(),v.cpu()) for k,v in p];del p;clear()
    j=JSSA(model,cpu,basis,len(row['input_ids'])+128);del cpu
    out=j.decode(l,eos,len(row['input_ids'])+128,max_new=3,ignore_eos=True,return_logits=True)
    errors=[float((a-b).abs().max()) for a,b in zip(ref['_logits'],out['_logits'])]
    # BF16 fused attention reductions need not match SDPA bitwise.
    assert max(errors)<.5,errors
    assert ref['ids']==out['ids'],(ref['ids'],out['ids'])
    for budget in BUDGETS:
        probe=j.decode(l,eos,budget,max_new=3,ignore_eos=True)
        assert probe['mean_selected_tokens']==budget
        assert probe['logical_kv_token_reads']==budget*2*32*8
    dump('preflight.json',dict(passed=True,longest_tokens=len(row['input_ids']),all_read_ids_equal=True,
        all_read_max_logit_error=max(errors),basis_layers=len(basis),budgets=BUDGETS,
        execution='HF adapter, official BF16 projected cache and Triton kernels; not native vLLM'))

@torch.inference_mode()
def worker(shard,smoke=False):
    model,tok,models,eos=setup();basis=torch.load(ROOT/'basis.pt',map_location='cpu',weights_only=False)['M']
    rows=json.loads((ROOT/'selected.json').read_text());seen=set();indices=[]
    for i,r in enumerate(rows):
        key=(r['task'],r['length'])
        if not smoke or key not in seen:indices.append(i)
        seen.add(key)
    folder=ROOT/('smoke' if smoke else 'evaluation');folder.mkdir(exist_ok=True)
    warmed=set()
    for number,i in enumerate(indices):
        if number%4!=shard:continue
        row=rows[i];path=folder/f'partial_{i}.json'
        record=json.loads(path.read_text()) if path.exists() else dict(index=i,task=row['task'],length=row['length'],
            answers=row['answers'],prompt_tokens=len(row['input_ids']),input_ids_sha256=sha_ids(row['input_ids']),methods={})
        if all(n in record['methods'] for n in METHODS):continue
        p,l,prefill=e.prefill(model,row['input_ids']);d,index_seconds=e.build(p,'gpu_digest')
        warm=row['length'] not in warmed
        def save(name,result,build_seconds):
            result.pop('steps',None);result.pop('audit',None)
            prediction=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
            result.update(prediction=prediction,score=e.official.metric_for(row['task'])([prediction],[row['answers']]),
                prefill_seconds=prefill,index_seconds=build_seconds,e2e_seconds=prefill+build_seconds+result['decode_seconds'])
            record['methods'][name]=e.clean(result);dump(str(path.relative_to(ROOT)),record)
            print(json.dumps(e.clean(dict(index=i,method=name,score=result['score'],read=result['logical_fraction']))),flush=True)
        order=['full','S','B'];shift=i%3;order=order[shift:]+order[:shift]
        for name in order:
            rank,budget=models[name] if name!='full' else (None,None)
            def run(cap,ignore=False,capture=True):
                return e.c.run.decode(model,p,l,'full' if name=='full' else 'adaptive',rank,budget,d if name!='full' else None,
                    eos,max_new=cap,capture=capture,ignore_eos=ignore,quantile=.95)
            if warm:
                tmp=run(3,True);del tmp
            if name not in record['methods']:
                result=run(row['max_new']);obj=result.pop('_controller')
                if obj:
                    den=sum(x['full'] for x in obj.costs);read=sum(x['read'] for x in obj.costs);pad=sum(x['padded'] for x in obj.costs)
                    result.update(logical_fraction=read/den if den else None,padded_fraction=pad/den if den else None,
                        logical_kv_token_reads=read,full_kv_token_reads=den,padded_kv_token_reads=pad)
                else:result.update(logical_fraction=1.,padded_fraction=1.)
                del obj
                save(name,result,index_seconds if name!='full' else 0);del result
            if smoke:
                samples=[]
                for repeat in range(3):
                    t=run(17,True,False);samples.append(t['decode_seconds']/16);del t
                record.setdefault('fixed_16step_seconds',{})[name]=samples
        cpu=[(k.cpu(),v.cpu()) for k,v in p];del p,d;clear()
        torch.cuda.synchronize();start=time.perf_counter()
        j=JSSA(model,cpu,basis,len(row['input_ids'])+row['max_new']+32)
        torch.cuda.synchronize();build=time.perf_counter()-start;del cpu
        # CPU staging is an evaluation memory accommodation, not a serving cost.
        for budget in BUDGETS:
            name=f'JSSA_{budget}'
            if warm:j.decode(l,eos,budget,max_new=3,ignore_eos=True)
            if name not in record['methods']:save(name,j.decode(l,eos,budget,max_new=row['max_new']),build)
            if smoke:
                record.setdefault('fixed_16step_seconds',{})[name]=[j.decode(l,eos,budget,max_new=17,ignore_eos=True)['decode_seconds']/16 for _ in range(3)]
        dump(str(path.relative_to(ROOT)),record)
        warmed.add(row['length']);del j,l;clear()

def sha_ids(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def summarize(folder):
    rows=[json.loads(p.read_text()) for p in (ROOT/folder).glob('partial_*.json')]
    complete=[r for r in rows if all(n in r['methods'] for n in METHODS)]
    report={'completed':len(complete),'started':len(rows),'groups':{}}
    for task,length in sorted({(r['task'],r['length']) for r in complete}):
        group=[r for r in complete if (r['task'],r['length'])==(task,length)]
        report['groups'][f'{task}_{length}']={'n':len(group),'methods':{}}
        for name in METHODS:
            vals=[r['methods'][name] for r in group]
            stat={k:statistics.mean(v[k] for v in vals if v[k] is not None) for k in
                ['score','logical_fraction','padded_fraction','mean_selected_tokens','decode_seconds','hit_generation_limit']}
            stat['weighted_ms_per_decode_token']=1000*sum(v['decode_seconds'] for v in vals)/max(1,sum(v['decode_steps'] for v in vals))
            report['groups'][f'{task}_{length}']['methods'][name]=stat
    dump(f'{folder}/comparison.json',report);return len(complete)

def launch():
    assert not (ROOT/'status.json').exists()
    (ROOT/'selected.json').write_bytes((SOURCE/'selected.json').read_bytes())
    protected=[SOURCE/'selected.json',SOURCE/'comparison.json']+[e.OLD/'branches'/n/'best_pair.pt' for n in ['S','B']]
    hashes={str(p):sha(p) for p in protected}
    dump('protocol.json',dict(n=450,tasks=['niah_single_1','niah_multivalue','vt'],lengths=[8192,16384,32768],
        methods=METHODS,official_default_budget=2048,budget_unit='selected tokens per KV head, including all selected sink/local/current tokens',
        JSSA_rank=32,JSSA_projected_cache='BF16 official alternative, FP8 default unsupported by old Ampere stack',
        JSSA_topk='official torch.topk fallback; compiled vLLM radix extension unavailable',
        backend='Common HuggingFace BF16 eager model, official JSSA Triton kernels adapted imports only',
        native_vllm_reproduced=False,coverage_target=.9,budget_quantile=.95,SB_policy='autonomous unmodified; sink/local/padding included',
        prompts='exact previous selected.json input IDs; official RULER prompt+answer prefix and official scoring',
        generation='greedy, model EOS, existing shared 128-token cap; not a claim all official task caps are 128',
        prefill='common full SDPA chunk1024; excluded from decode timing',
        timing='one exclusive worker per GPU; warmup; quality decode timing plus 16 forced steps x3 on 9 smoke prompts',
        memory_note='JSSA scans all rank32 projected keys in addition to selected K/V; bytes recorded separately. CPU prefix staging excluded from decode; index time includes CPU->GPU adaptation and is not native end-to-end.',
        protected_sha256=hashes,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    dump('status.json',dict(stage='waiting_for_official_calibration'))
    while not (ROOT/'calibration_complete.json').exists():
        assert not (ROOT/'calibration_failed.json').exists(),'calibration failed'
        time.sleep(20)
    assert (ROOT/'kernel_check.json').exists(),'kernel check missing'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'deps'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    dump('status.json',dict(stage='preflight'))
    with (ROOT/'preflight.log').open('w') as log:
        subprocess.run([sys.executable,'-u',__file__,'preflight'],env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage in ['smoke','evaluation']:
        jobs=[]
        for shard in range(4):
            with (ROOT/f'{stage}_{shard}.log').open('w') as log:
                jobs.append(subprocess.Popen([sys.executable,'-u',__file__,stage,str(shard)],env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT))
        dump('status.json',dict(stage=stage,pids=[j.pid for j in jobs]))
        while any(j.poll() is None for j in jobs):
            if any(j.poll() not in [None,0] for j in jobs):
                for j in jobs:
                    if j.poll() is None:j.terminate()
                raise RuntimeError(f'{stage} worker failed; stopped remaining workers')
            time.sleep(20)
        assert all(j.returncode==0 for j in jobs)
        assert summarize(stage)==(9 if stage=='smoke' else 450)
    assert all(sha(p)==h for p,h in hashes.items())
    dump('status.json',dict(stage='complete',old_results_unchanged=True))

if __name__=='__main__':
    try:
        if sys.argv[1]=='launch':launch()
        elif sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='summarize':summarize(sys.argv[2])
        else:worker(int(sys.argv[2]),sys.argv[1]=='smoke')
    except Exception:
        dump('failure_'+sys.argv[1]+'.json',dict(traceback=traceback.format_exc()))
        if sys.argv[1]=='launch':dump('status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
