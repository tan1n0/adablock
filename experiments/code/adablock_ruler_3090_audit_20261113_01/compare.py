"""Equal-read, no-sink/no-window RULER comparison on one GPU per worker."""
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = Path('/root/adablock_jssa_20261023_01')
AUTO = Path('/root/adablock_ruler_autonomous_20261022_01')
BUDGETS = (64,128,256,512)
TASKS = ('niah_multivalue','vt')


def save(name, value):
    path = ROOT/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()


def prepare():
    source = AUTO/'selected.json'
    rows = json.loads(source.read_bytes())
    excluded = set(json.loads((ROOT/'excluded.json').read_text()))
    rng = random.Random(20261113)
    groups = []
    for task in TASKS:
        pool = [row for row in rows if row['length']==32768 and row['task']==task and row['input_sha256'] not in excluded]
        assert len(pool)>=10,(task,len(pool))
        groups.append(rng.sample(pool,min(40,len(pool))))
    selected = [row for pair in zip(*groups) for row in pair]
    for row in selected:
        assert len(row['input_ids'])==row['prompt_tokens']
        assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_sha256']
    target = ROOT/'selected.json'
    if target.exists():
        assert json.loads(target.read_text())==selected
    else:
        save('selected.json',selected)
    save('protocol.json',dict(seed=20261113,source_sha256=sha(source),selected_sha256=sha(target),
         tasks=TASKS,initial_cases=20,extension_cases=len(selected),history_budget=BUDGETS,coarse_candidates=256,
         block_size=16,fixed='visible current block only, no fixed sink/local, all decode layers sparse',
         cap=128,ranking='frozen B FP32, FP32 Gram; JSSA calibrated rank-32 BF16',
         attention='same official JSSA indexed kernel for both sparse branches; Full native SDPA',
         timing='exclusive GPU; 16 identical teacher-forced input steps, 3 repetitions; events separately',
         extension='extend if absolute all-hit difference is at most 2/20 at both 128 and 256'))


def worker(stage, shard=0, limit=20):
    sys.path.insert(0,str(AUTO))
    import experiment as e
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    sys.path.insert(0,str(OLD))
    from jssa_adapter import JSSA
    from triton_jssa_indexed_attend import jssa_indexed_attend
    from triton_jssa_score_topk import jssa_score
    from triton_jssa_store import jssa_project_and_store
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from utils.ranking_batch import forward_packed, pack_scenes
    from fast_digest import compact_digest

    class Meter:
        def __init__(self, enabled=False):
            self.enabled=enabled
            self.events=[]

        def call(self, name, function):
            if not self.enabled:
                return function()
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); out=function(); end.record()
            self.events.append((name,start,end))
            return out

        def totals(self, steps):
            out={}
            for name,start,end in self.events:
                out[name]=out.get(name,0)+start.elapsed_time(end)/steps
            return out

    class Sparse(JSSA):
        def __init__(self, model, prefix, basis, capacity):
            # Both methods share the same paged K/V. JSSA adds its projected
            # slice in the same slot; B uses its own small block summaries.
            super().__init__(model,prefix,basis,capacity)
            self.offsets=torch.arange(16,device='cuda')
            self.heads=torch.arange(8,device='cuda')
            self.sm_count=torch.cuda.get_device_properties(0).multi_processor_count

        def init_digest(self):
            n=self.prefix//16
            cap=self.nblocks
            self.digests=[]
            for cache in self.cache:
                keys=cache[:n,:,:,:128].permute(2,0,1,3).reshape(8,n*16,128)
                d=compact_digest(keys)
                buffers={}
                for name,x in d.items():
                    shape=list(x.shape); axis=0 if name=='counts' else 1
                    shape[axis]=cap
                    dst=torch.empty(shape,dtype=x.dtype,device=x.device)
                    dst.narrow(axis,0,n).copy_(x)
                    buffers[name]=dst
                self.digests.append(buffers)

        def update(self, i, key, value):
            cache=self.cache[i]
            if self.mode=='jssa':
                jssa_project_and_store(key,value,cache,self.slot,self.basis[i])
            else:
                block,offset=divmod(self.position,16)
                cache[block,offset,:,:128].copy_(key[0])
                cache[block,offset,:,128:256].copy_(value[0])

        def update_digest(self,i,block):
            d=compact_digest(self.cache[i][block,:,:,:128].transpose(0,1))
            for name,x in d.items():
                self.digests[i][name].narrow(0 if name=='counts' else 1,block,1).copy_(x)

        def coarse(self,i,q,complete,scale):
            grouped=q.view(8,4,128).float()*scale
            d=self.digests[i]
            score=(grouped.mean(1,keepdim=True) @ d['mean'][:,:complete].transpose(-1,-2)).squeeze(1)
            ids=score.topk(min(256,complete),sorted=False).indices
            return grouped,ids

        def fine(self,i,grouped,ids):
            d=self.digests[i]; h,n=ids.shape
            packed={name:d[name][self.heads[:,None],ids].flatten(0,1) for name in ('mean','basis','eigenvalues')}
            packed.update(query=grouped,counts=d['counts'][ids].reshape(-1),block_ids=ids.flatten(),
                scene_ids=self.heads.repeat_interleave(n),token_position=torch.full((h,),self.position,device='cuda'),
                layer_id=torch.full((h,),i,device='cuda'),kv_head_id=self.heads,block_size=torch.full((h,),16,device='cuda'))
            logits=forward_packed(self.ranker,packed).view(h,n)
            if self.audit:
                assert torch.isfinite(logits).all().item()
            if self.audit and i==0:
                scenes=[dict(query=grouped[head],mean=d['mean'][head,ids[head]],
                    basis=d['basis'][head,ids[head]],eigenvalues=d['eigenvalues'][head,ids[head]],
                    counts=d['counts'][ids[head]],block_ids=ids[head],block_size=16,
                    token_position=self.position,layer_id=i,kv_head_id=head) for head in range(h)]
                reference=forward_packed(self.ranker,pack_scenes(scenes,'cuda')).view(h,n)
                torch.testing.assert_close(logits,reference,rtol=0,atol=0)
            blocks=ids.gather(1,logits.topk(self.budget//16,sorted=False).indices)
            return (blocks[...,None]*16+self.offsets).view(1,h,-1)

        def jssa_pick(self,i,q,complete):
            projected=q.view(8,4,128).mean(1,keepdim=True) @ self.basis[i].transpose(1,2)
            scores=jssa_score(projected.transpose(0,1),self.cache[i],self.bt,self.seq,128,32,scores_out=self.scores)
            return scores[...,:complete*16].topk(self.budget).indices

        def forward(self,module,layer,hidden,embeddings):
            shape=(1,1,-1,128)
            q=module.q_proj(hidden).view(shape).transpose(1,2)
            k=module.k_proj(hidden).view(shape).transpose(1,2)
            v=module.v_proj(hidden).view(shape).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*embeddings)
            q=q[0,:,0].unsqueeze(0).contiguous()
            self.meter.call('cache_update',lambda:self.update(layer,k[:,:,0].contiguous(),v[:,:,0].contiguous()))
            complete,offset=divmod(self.position,16)
            if self.mode=='all':
                indices=torch.arange(self.position+1,device='cuda',dtype=torch.int32).expand(1,8,-1).contiguous()
            else:
                if self.mode=='jssa':
                    history=self.meter.call('projected_score_topk',lambda:self.jssa_pick(layer,q,complete))
                else:
                    grouped,ids=self.meter.call('coarse_score_topk',lambda:self.coarse(layer,q,complete,module.scaling))
                    history=self.meter.call('fine_score_topk',lambda:self.fine(layer,grouped,ids))
                tail=(complete*16+self.offsets[:offset+1]).view(1,1,-1).expand(1,8,-1)
                indices=torch.cat((history,tail),dim=-1).to(torch.int32).contiguous()
                assert indices.shape==(1,8,self.budget+offset+1)
                if self.audit:
                    assert history.min().item()>=0 and history.max().item()<complete*16
                    ordered=indices.sort(dim=-1).values
                    assert (ordered[...,1:]!=ordered[...,:-1]).all().item()
            y=self.meter.call('indexed_kv_and_attention',lambda:jssa_indexed_attend(
                q,self.cache[layer],self.bt,indices,128,float(module.scaling),self.output,self.scratch,self.sm_count))
            if self.mode=='B' and offset==15:
                self.meter.call('digest_update',lambda:self.update_digest(layer,complete))
            return module.o_proj(y.reshape(1,1,-1)),None

    def reset():
        e.reset_registry();gc.collect();torch.cuda.empty_cache()

    def decode(model,prefix,first,eos,mode,cap=128,budget=0,engine=None,forced=None,audit=False,profile=False):
        meter=Meter(profile)
        n=prefix[0][0].shape[2] if prefix else engine.prefix
        ids=[int(first.argmax())]
        tokens=torch.empty((1,1),device='cuda',dtype=torch.long)
        positions=torch.empty((1,),device='cuda',dtype=torch.long)
        cache=e.BufferCache.from_legacy_cache(prefix) if mode=='full' else None
        if cache and profile:
            original=cache.update
            cache.update=lambda *args,**kw:meter.call('cache_update',lambda:original(*args,**kw))
        if engine is not None:
            engine.mode,engine.budget,engine.ranker=mode,budget,ranker
            engine.audit,engine.meter=audit,meter
            engine.install()
        torch.cuda.synchronize(); start=time.perf_counter()
        reads=[]; logits_trace=[]
        try:
            for step in range(len(forced) if forced is not None else cap-1):
                if forced is None and ids[-1] in eos:
                    break
                p=n+step
                tokens.fill_(forced[step] if forced is not None else ids[-1]);positions.fill_(p)
                if engine is not None:
                    engine.position=p;engine.slot.fill_(p);engine.seq.fill_(p//16*16)
                out=model.model(input_ids=tokens,position_ids=positions[None],cache_position=positions,
                                past_key_values=cache,use_cache=mode=='full')
                logits=model.lm_head(out.last_hidden_state[:,-1:])[0,-1].float()
                ids.append(int(logits.argmax()))
                if audit and mode in ('full','all'):
                    logits_trace.append(logits.cpu())
                reads.append(p+1 if mode in ('full','all') else budget+p%16+1)
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
        finally:
            if engine is not None:
                engine.uninstall()
        return dict(ids=ids,decode_steps=len(reads),decode_seconds=seconds,reads_per_step=reads,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    stages_ms=meter.totals(max(1,len(reads))),_logits=logits_trace,
                    eos=ids[-1] in eos,cap_reached=forced is None and len(ids)==cap and ids[-1] not in eos)

    def variants(row, number):
        reset();torch.cuda.reset_peak_memory_stats()
        p,l,prefill_s=e.prefill(model,row['input_ids'])
        case=dict(task=row['task'],prompt_tokens=len(row['input_ids']),answers=row['answers'],
                  input_sha256=row['input_sha256'],prefill_seconds=prefill_s,methods={})
        if stage=='profile':
            natural=decode(model,p,l,eos,'full',cap=17,forced=None)
            forced=(natural['ids']+row['input_ids'][-16:])[:16]
        else:
            forced=None
        def run(mode,k=0,engine=None):
            torch.cuda.reset_peak_memory_stats()
            prefix=p if mode=='full' else None
            if stage=='profile':
                decode(model,prefix,l,eos,mode,budget=k,engine=engine,forced=forced[:3])
                samples=[decode(model,prefix,l,eos,mode,budget=k,engine=engine,forced=forced) for _ in range(3)]
                record=decode(model,prefix,l,eos,mode,budget=k,engine=engine,forced=forced,profile=True)
                record['samples_ms']=[1000*x['decode_seconds']/16 for x in samples]
                return record
            return decode(model,prefix,l,eos,mode,cap=3 if stage=='preflight' else 128,budget=k,engine=engine,
                          forced=[int(l.argmax())]*(18 if mode in ('B','jssa') else 2) if stage=='preflight' else None,audit=stage=='preflight')
        case['methods']['full']=run('full')
        torch.cuda.synchronize(); t=time.perf_counter()
        cpu=[(k.cpu(),v.cpu()) for k,v in p];p=None;reset()
        case['prefix_cpu_staging_seconds']=time.perf_counter()-t
        torch.cuda.synchronize(); t=time.perf_counter()
        engine=Sparse(model,cpu,basis,len(row['input_ids'])+128);del cpu
        torch.cuda.synchronize(); case['paged_projection_build_seconds']=time.perf_counter()-t
        if stage=='preflight':
            ref=case['methods']['full']; all_read=run('all',engine=engine)
            error=max(float((a-b).abs().max()) for a,b in zip(ref['_logits'],all_read['_logits']))
            assert ref['ids']==all_read['ids'] and error<.5,(ref['ids'],all_read['ids'],error)
            case['all_read_max_logit_error']=error
        order=list(BUDGETS)
        if stage=='quality':
            offset=number%len(order);order=order[offset:]+order[:offset]
        # JSSA runs first so B's digest allocation is outside its measurements.
        for k in order:
            case['methods'][f'jssa_{k}']=run('jssa',k,engine)
        torch.cuda.synchronize();t=time.perf_counter();engine.init_digest();torch.cuda.synchronize()
        case['B_digest_build_seconds']=time.perf_counter()-t
        for k in order:
            case['methods'][f'B_{k}']=run('B',k,engine)
        if stage=='preflight':
            for mode in ('jssa','B','B','jssa'):
                repeated=run(mode,64,engine)
                base=case['methods'][f'{mode}_64']
                assert repeated['ids']==base['ids'] and repeated['reads_per_step']==base['reads_per_step'], mode
            case['alternating_branch_replay_passed']=True
            case['ranker_pack_exact_match']=True
            case['sparse_checked_steps']=18
        for name,result in case['methods'].items():
            result.pop('_logits')
            if name!='full':
                k=int(name.split('_')[-1]);n=len(row['input_ids'])
                assert result['reads_per_step']==[k+(n+s)%16+1 for s in range(result['decode_steps'])]
            if stage=='quality':
                text=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
                result['prediction']=text
                result['ruler_score']=e.official.metric_for(row['task'])([text],[row['answers']])
                result['hits']=sum(bool(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',text,re.I)) for a in row['answers'])
                result['all_hit']=result['hits']==len(row['answers'])
        del engine
        reset()
        return case

    with torch.inference_mode():
        model,tok,models,eos=e.load()
        ranker=models['B'][0];del models
        assert sha(OLD/'basis.pt')=='ee0dcaaec31d8a3f92212d86e177768337ddf256685d43ba6f7ef81968fbe98c'
        basis=torch.load(OLD/'basis.pt',map_location='cpu',weights_only=True)['M']
        rows=json.loads((ROOT/'selected.json').read_text())
        if stage=='preflight':
            i=max(range(len(rows)),key=lambda i:len(rows[i]['input_ids']))
            save('preflight.json',variants(rows[i],i))
        elif stage=='profile':
            for i in (0,1):
                save(f'profile_{i}.json',variants(rows[i],i))
        else:
            for i in range(shard,limit,4):
                path=f'quality_{i:03d}.json'
                if not (ROOT/path).exists():
                    save(path,variants(rows[i],i))
                print('complete',i,rows[i]['task'],flush=True)


def summarize(limit):
    cases=[json.loads((ROOT/f'quality_{i:03d}.json').read_text()) for i in range(limit)]
    out={}
    for task in ('all',*TASKS):
        rows=[x for x in cases if task=='all' or x['task']==task]
        out[task]={}
        for name in rows[0]['methods']:
            m=[r['methods'][name] for r in rows]; steps=sum(x['decode_steps'] for x in m)
            full_ok=[r['methods'][name] for r in rows if r['methods']['full']['all_hit']]
            out[task][name]=dict(n=len(m),all_hit=sum(x['all_hit'] for x in m),hits=sum(x['hits'] for x in m),
                total_targets=sum(len(r['answers']) for r in rows),ruler_score=statistics.mean(x['ruler_score'] for x in m),
                full_correct_n=len(full_ok),all_hit_on_full_correct=sum(x['all_hit'] for x in full_ok),
                read_tokens_per_head_step=sum(sum(x['reads_per_step']) for x in m)/max(steps,1),
                decode_ms_per_step=1000*sum(x['decode_seconds'] for x in m)/max(steps,1),
                mean_decode_seconds=statistics.mean(x['decode_seconds'] for x in m),
                cap_reached=sum(x['cap_reached'] for x in m))
    save(f'summary_{limit}.json',out)
    return out['all']


def launch():
    prepare()
    env=dict(os.environ,PYTHONPATH=str(OLD/'deps'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    def group(stage,gpus,limit=20):
        jobs=[]
        try:
            for gpu in gpus:
                with (ROOT/f'{stage}_{limit}_{gpu}.log').open('a') as log:
                    jobs.append(subprocess.Popen([sys.executable,'-u',__file__,stage,str(gpu),str(limit)],
                        env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
            save('status.json',dict(stage=stage,limit=limit,pids=[x.pid for x in jobs]))
            while True:
                codes=[j.poll() for j in jobs]
                if any(c not in (None,0) for c in codes):
                    raise RuntimeError(f'{stage} failed: {codes}')
                if all(c==0 for c in codes):
                    return
                time.sleep(2)
        finally:
            for j in jobs:
                if j.poll() is None:
                    j.terminate()
            for j in jobs:
                j.wait()
    group('preflight',[0]);group('quality',range(4),20)
    pilot=summarize(20)
    group('profile',[0])
    limit=20
    if all(abs(pilot[f'B_{k}']['all_hit']-pilot[f'jssa_{k}']['all_hit'])<=2 for k in (128,256)):
        limit=len(json.loads((ROOT/'selected.json').read_text()))
        group('quality',range(4),limit);summarize(limit)
    save('status.json',dict(stage='complete',cases=limit))


if __name__=='__main__':
    if sys.argv[1]=='launch':
        try:
            launch()
        except Exception:
            save('status.json',dict(stage='failed',traceback=traceback.format_exc()))
            raise
    else:
        worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
