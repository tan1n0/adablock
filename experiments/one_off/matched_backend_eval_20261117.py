"""Matched Full/JSSA-512 RULER run on the frozen 100 selected 32K inputs.

COBS uses cobs_persistent_graph_eval_20261116.py under the same outer protocol.
Full uses HF SDPA. JSSA uses the pinned BF16 rank-32 projected-cache Triton
score and indexed-attention kernels, with 512 complete-history tokens plus the
visible current block. The JSSA index is built directly from GPU prefill K/V,
consuming and releasing one source layer at a time to fit a 24 GiB GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01'
DIGEST=ROOT/'experiments/server_snapshots/ruler_autonomous_20261022_01'
JSSA=ROOT/'experiments/runs/adablock_jssa_20261023_01'
RULER=ROOT/'experiments/runs/adablock_ruler_harder_20261022_01'
SELECTED_SHA='35eb364d354c7230c57282d3d5cbbe906941495c57946abd6597c9aac1c8f0ec'
BASIS_SHA='ee0dcaaec31d8a3f92212d86e177768337ddf256685d43ba6f7ef81968fbe98c'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def protocol(args):
    return {'method':args.method,'selected_sha256':sha(SOURCE/'selected.json'),
            'basis_sha256':sha(JSSA/'basis.pt') if args.method=='jssa_512' else None,
            'jssa_compat_kernel_sha256':sha(ROOT/'experiments/one_off/triton_jssa_indexed_attend_compat_20261117.py') if args.method=='jssa_512' else None,
            'model':str(args.model.resolve()),'gpu':'one exclusive physical RTX 3090, CUDA_VISIBLE_DEVICES=0',
            'model_dtype':'bfloat16','prefill':'HF native SDPA, 1024-token chunks',
            'generation':'greedy, shared Full prefill and first token, EOS, 128-token cap',
            'fixed':'no sink/local; 512 complete-history tokens plus visible current block for sparse',
            'jssa':'rank32 BF16 projected cache, torch topk, pinned Triton score and indexed attention with GQA-row padding for Triton 3.1; direct GPU per-layer cache conversion' if args.method=='jssa_512' else None,
            'full':'HF native SDPA' if args.method=='full' else None,
            'forced':'same historical Full generated IDs padded with prompt suffix to 16 tokens; three warmed timed repeats and separate event profile',
            'timing':'GPU-synchronized wall; natural decode, index build, 16 forced steps separately',
            'index_scope':'GPU-resident Full prefix to deployable sparse cache; source Full K/V released layer by layer',
            'scoring':'official RULER task metric; same 100 input hashes as selected source'}


def validate(args):
    if args.method not in ('full','jssa_512'):raise ValueError(args.method)
    if not (0<=args.start_index<args.limit<=100):raise ValueError('invalid case interval')
    if sha(SOURCE/'selected.json')!=SELECTED_SHA:raise ValueError('selected input hash mismatch')
    if args.method=='jssa_512' and sha(JSSA/'basis.pt')!=BASIS_SHA:raise ValueError('basis hash mismatch')
    if not (args.model/'config.json').exists():raise FileNotFoundError(args.model/'config.json')
    rows=json.loads((SOURCE/'selected.json').read_text())
    for i in range(args.start_index,args.limit):
        old=json.loads((SOURCE/f'quality_{i:03d}.json').read_text())
        if rows[i]['input_sha256']!=old['input_sha256']:raise ValueError(f'input hash mismatch {i}')
    return rows


def run(args):
    rows=validate(args);p=protocol(args);pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=p:raise ValueError('protocol mismatch, choose new output directory')
    save(pp,p)
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    sys.path.insert(0,str(DIGEST));from cache_buffer import BufferCache,reset_registry
    sys.path.insert(0,str(JSSA/'kernel_adapter'))
    if args.method=='jssa_512':
        from triton_jssa_store import jssa_project_and_store
        from triton_jssa_score_topk import jssa_score
        from triton_jssa_indexed_attend_compat_20261117 import jssa_indexed_attend
    spec=importlib.util.spec_from_file_location('ruler_official_matched',ROOT/'experiments/server_snapshots/ruler_harder_20261022_01/official.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official);official.PACKAGE=RULER
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.manual_seed(20261015)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    if config.rope_scaling and 'type' not in config.rope_scaling:config.rope_scaling['type']=config.rope_scaling['rope_type']
    model=AutoModelForCausalLM.from_pretrained(args.model,config=config,torch_dtype=torch.bfloat16,
                    device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    eos_value=model.generation_config.eos_token_id;eos=set(eos_value if isinstance(eos_value,list) else [eos_value])
    capacity=max(len(row['input_ids']) for row in rows)+128
    basis=None
    if args.method=='jssa_512':
        payload=torch.load(JSSA/'basis.pt',map_location='cpu',weights_only=True)['M']
        basis={i:payload[i].to(device='cuda',dtype=torch.bfloat16).contiguous() for i in range(32)}
        del payload

    @torch.inference_mode()
    def prefill(ids):
        reset_registry();cache=BufferCache(capacity)
        torch.cuda.synchronize();start=time.perf_counter()
        for off in range(0,len(ids),1024):
            output=model.model(input_ids=torch.tensor([ids[off:off+1024]],device='cuda'),
                               past_key_values=cache,use_cache=True)
        first=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        del output
        return cache,first,seconds

    class JSSAEngine:
        def __init__(self,pairs,prefix):
            self.model=model;self.prefix=prefix;self.h=8;self.g=4;self.d=128
            self.nblocks=(capacity+15)//16;self.bt=torch.arange(self.nblocks,device='cuda',dtype=torch.int32)[None]
            self.seq=torch.zeros(1,device='cuda',dtype=torch.int32)
            self.slot=torch.zeros(1,device='cuda',dtype=torch.int64)
            self.basis=basis;self.cache=[];self.original=[];self.scratch_buffers={}
            self.qkv=torch.empty((1,8,128),device='cuda',dtype=torch.bfloat16)
            self.projq=torch.empty((8,1,32),device='cuda',dtype=torch.bfloat16)
            self.scores=torch.empty((1,8,self.nblocks*16),device='cuda',dtype=torch.float32)
            self.output=torch.empty((1,32,128),device='cuda',dtype=torch.bfloat16)
            self.offsets=torch.arange(16,device='cuda',dtype=torch.int32)
            self.sm_count=torch.cuda.get_device_properties(0).multi_processor_count
            self.projected_index_bytes=0;self.total_cache_bytes=0
            slots=torch.arange(prefix,device='cuda',dtype=torch.int64)
            for layer in range(32):
                key,value=pairs[layer]
                dest=torch.empty((self.nblocks,16,8,288),device='cuda',dtype=torch.bfloat16)
                source_k=key[0].transpose(0,1).contiguous();source_v=value[0].transpose(0,1).contiguous()
                jssa_project_and_store(source_k,source_v,dest,slots,self.basis[layer])
                self.cache.append(dest)
                self.total_cache_bytes+=dest.numel()*dest.element_size()
                self.projected_index_bytes+=self.nblocks*16*8*32*2
                pairs[layer]=None
                del key,value,source_k,source_v,dest
            del slots

        def scratch(self,name,shape,dtype,device):
            key=(name,tuple(shape),dtype)
            if key not in self.scratch_buffers:self.scratch_buffers[key]=torch.empty(shape,dtype=dtype,device=device)
            return self.scratch_buffers[key]
        def install(self):
            for i,layer in enumerate(model.model.layers):
                module=layer.self_attn;self.original.append(module.forward)
                def forward(module,hidden_states,position_embeddings,_layer=i,**kwargs):
                    return self.forward(module,_layer,hidden_states,position_embeddings)
                module.forward=types.MethodType(forward,module)
        def uninstall(self):
            for layer,fn in zip(model.model.layers,self.original):layer.self_attn.forward=fn
            self.original=[]
        def timed(self,name,fn):
            if not self.profile:return fn()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();result=fn();end.record();self.events.append((name,start,end));return result
        def event_totals(self,steps):
            out={}
            for name,start,end in self.events:out[name]=out.get(name,0.)+start.elapsed_time(end)/steps
            return out
        def forward(self,module,layer,hidden,embeddings):
            shape=(1,1,-1,128)
            q=module.q_proj(hidden).view(shape).transpose(1,2)
            k=module.k_proj(hidden).view(shape).transpose(1,2)
            v=module.v_proj(hidden).view(shape).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*embeddings)
            q=q[0,:,0].unsqueeze(0).contiguous()
            self.timed('cache_project_store',lambda:jssa_project_and_store(
                k[:,:,0].contiguous(),v[:,:,0].contiguous(),self.cache[layer],self.slot,self.basis[layer]))
            def select():
                projected=q.view(8,4,128).mean(1,keepdim=True) @ self.basis[layer].transpose(1,2)
                scores=jssa_score(projected.transpose(0,1),self.cache[layer],self.bt,
                                  self.seq,128,32,scores_out=self.scores)
                return scores[...,:self.complete*16].topk(512).indices
            history=self.timed('projected_score_topk',select)
            tail=(self.complete*16+self.offsets[:self.offset+1]).view(1,1,-1).expand(1,8,-1)
            indices=torch.cat((history,tail),dim=-1).to(torch.int32).contiguous()
            assert indices.shape==(1,8,512+self.offset+1)
            def attend():
                return jssa_indexed_attend(q,self.cache[layer],self.bt,indices,128,
                                           float(module.scaling),self.output,self.scratch,self.sm_count)
            out=self.timed('indexed_attention',attend)
            return module.o_proj(out.reshape(1,1,-1)),None

    @torch.inference_mode()
    def full_decode(cache,first,prompt_len,forced=None,profile=False):
        for layer_index,layer in enumerate(cache.layers):
            key,value=cache.buffers[layer_index]
            layer.keys=key[...,:prompt_len,:];layer.values=value[...,:prompt_len,:]
        token=torch.empty((1,1),device='cuda',dtype=torch.long)
        position=torch.empty((1,),device='cuda',dtype=torch.long)
        ids=[int(first.argmax())]
        torch.cuda.synchronize();start=time.perf_counter()
        for step in range(len(forced) if forced is not None else 127):
            if forced is None and ids[-1] in eos:break
            p=prompt_len+step
            token.fill_(forced[step] if forced is not None else ids[-1]);position.fill_(p)
            output=model.model(input_ids=token,position_ids=position[None],cache_position=position,
                               past_key_values=cache,use_cache=True)
            logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
            ids.append(int(logits.argmax()))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        del token,position
        return {'ids':ids,'decode_steps':len(ids)-1,'decode_seconds':seconds,
                'cap_reached':forced is None and len(ids)==128 and ids[-1] not in eos,
                'stages_ms_per_step':{}}

    @torch.inference_mode()
    def jssa_decode(engine,first,prompt_len,forced=None,profile=False):
        ids=[int(first.argmax())]
        engine.profile=profile;engine.events=[];engine.install()
        token=torch.empty((1,1),device='cuda',dtype=torch.long)
        position=torch.empty((1,),device='cuda',dtype=torch.long)
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            for step in range(len(forced) if forced is not None else 127):
                if forced is None and ids[-1] in eos:break
                p=prompt_len+step
                token.fill_(forced[step] if forced is not None else ids[-1]);position.fill_(p)
                engine.position=p;engine.complete,engine.offset=divmod(p,16)
                engine.slot.fill_(p);engine.seq.fill_(engine.complete*16)
                output=model.model(input_ids=token,position_ids=position[None],
                                   cache_position=position,past_key_values=None,use_cache=False)
                logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
                ids.append(int(logits.argmax()))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
        finally:engine.uninstall()
        steps=len(ids)-1
        return {'ids':ids,'decode_steps':steps,'decode_seconds':seconds,
                'cap_reached':forced is None and len(ids)==128 and ids[-1] not in eos,
                'stages_ms_per_step':engine.event_totals(max(1,steps)) if profile else {}}

    with torch.inference_mode():
        for i in range(args.start_index,args.limit):
            target=args.output/f'case_{i:03d}.json'
            if target.exists():print('skip',i,flush=True);continue
            row=rows[i];old=json.loads((SOURCE/f'quality_{i:03d}.json').read_text())
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            cache,first,prefill_seconds=prefill(row['input_ids']);prompt_len=len(row['input_ids'])
            if int(first.argmax())!=old['methods']['full']['ids'][0]:raise ValueError(f'Full first token mismatch {i}')
            prefill_peak_alloc=torch.cuda.max_memory_allocated()/1048576
            prefill_peak_reserved=torch.cuda.max_memory_reserved()/1048576
            if args.method=='full':
                torch.cuda.synchronize();index_seconds=0.
                natural=full_decode(cache,first,prompt_len)
                engine=None;index_mib=0.;total_cache_mib=0.
            else:
                pairs=list(cache.to_legacy_cache())
                cache.buffers.clear();cache.layers.clear();del cache
                reset_registry();gc.collect()
                torch.cuda.synchronize();begin=time.perf_counter()
                engine=JSSAEngine(pairs,prompt_len)
                pairs.clear();del pairs
                torch.cuda.synchronize();index_seconds=time.perf_counter()-begin
                index_mib=engine.projected_index_bytes/1048576
                total_cache_mib=engine.total_cache_bytes/1048576
                natural=jssa_decode(engine,first,prompt_len)
            expected=old['methods'][args.method]['ids']
            matching_ids=natural['ids']==expected
            text=official.normalize_prediction(tokenizer.decode(natural['ids'],skip_special_tokens=True),row['task'])
            natural['prediction']=text
            natural['ruler_score']=official.metric_for(row['task'])([text],[row['answers']])
            natural['hits']=sum(bool(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',text,re.I)) for a in row['answers'])
            natural['all_hit']=natural['hits']==len(row['answers'])
            natural['matches_historical_ids']=matching_ids
            natural['reads_per_kv_head_step']=[(prompt_len+step+1 if args.method=='full' else 512+(prompt_len+step)%16+1) for step in range(natural['decode_steps'])]
            forced=(old['methods']['full']['ids'][:-1]+row['input_ids'][-16:])[:16]
            if len(forced)!=16:raise ValueError('Insufficient forced tokens')
            decode=(lambda seq,profile=False:full_decode(cache,first,prompt_len,forced=seq,profile=profile)) if args.method=='full' else (lambda seq,profile=False:jssa_decode(engine,first,prompt_len,forced=seq,profile=profile))
            decode(forced[:3])
            samples=[]
            for _ in range(3):samples.append(1000*decode(forced)['decode_seconds']/16)
            stages=decode(forced,profile=True)['stages_ms_per_step']
            forced_median=statistics.median(samples)
            peak_alloc=torch.cuda.max_memory_allocated()/1048576
            peak_reserved=torch.cuda.max_memory_reserved()/1048576
            case={'index':i,'task':row['task'],'input_sha256':row['input_sha256'],
                  'prompt_tokens':prompt_len,'prefill_seconds':prefill_seconds,
                  'prefill_peak_allocated_mib':prefill_peak_alloc,'prefill_peak_reserved_mib':prefill_peak_reserved,
                  'method':args.method,'natural':natural,'index_build_seconds':index_seconds,
                  'index_cache_mib':index_mib,'total_sparse_cache_mib':total_cache_mib,
                  'forced_input_ids':forced,'forced_samples_ms_per_step':samples,
                  'forced_median_ms_per_step':forced_median,'forced_stages_ms_per_step':stages,
                  'postprefill_forced16_seconds':index_seconds+16*forced_median/1000,
                  'peak_allocated_mib':peak_alloc,'peak_reserved_mib':peak_reserved,
                  'reference_historical_score':old['methods'][args.method]['ruler_score']}
            save(target,case)
            print('complete',i,args.method,'score',natural['ruler_score'],'ids_equal',matching_ids,
                  'index_s',round(index_seconds,3),'forced_ms',round(forced_median,2),flush=True)
            del natural,decode,engine,first
            if args.method=='full':del cache
            reset_registry();gc.collect();torch.cuda.empty_cache()


def summarize(args):
    rows=[json.loads((args.output/f'case_{i:03d}.json').read_text()) for i in range(args.limit)]
    values=[r['natural'] for r in rows]
    out={'method':args.method,'n':len(rows),
         'score':statistics.mean(x['ruler_score'] for x in values),
         'all_hit':sum(x['all_hit'] for x in values),'cap_reached':sum(x['cap_reached'] for x in values),
         'historical_ids_matching':sum(x['matches_historical_ids'] for x in values),
         'natural_decode_ms_per_step':1000*sum(x['decode_seconds'] for x in values)/sum(x['decode_steps'] for x in values),
         'forced_median_ms_per_step':statistics.median(r['forced_median_ms_per_step'] for r in rows),
         'mean_index_build_seconds':statistics.mean(r['index_build_seconds'] for r in rows),
         'mean_postprefill_forced16_seconds':statistics.mean(r['postprefill_forced16_seconds'] for r in rows),
         'median_peak_allocated_mib':statistics.median(r['peak_allocated_mib'] for r in rows),
         'median_peak_reserved_mib':statistics.median(r['peak_reserved_mib'] for r in rows)}
    save(args.output/f'summary_{args.limit}.json',out);print(json.dumps(out,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('check','run','summarize'))
    parser.add_argument('--method',choices=('full','jssa_512'),required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if args.command=='check':
        rows=validate(args);print(json.dumps({'ready':args.limit-args.start_index,'selected_sha256':sha(SOURCE/'selected.json'),'gpu':'exclusive RTX3090'}))
    elif args.command=='run':run(args)
    else:summarize(args)

if __name__=='__main__':main()
