"""Compare mass and value-aware block sets on identical Full decode Q/K/V.

Full Llama attention remains installed. A wrapper observes selected layer
queries after the original attention call and never changes its output. This
is a sampled local-error diagnostic, separate from free-generation scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.one_off.value_aware_20261116 import contributions,mass_oracle,reconstructed,swap_search

SOURCE=ROOT/'experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01'
BUFFER=ROOT/'experiments/server_snapshots/ruler_autonomous_20261022_01'
MODEL=Path('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb')
OUT=ROOT/'results/value_aware_selector_20261116/full_trajectory'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)

@torch.inference_mode()
def run(args):
    sys.path.insert(0,str(BUFFER))
    from cache_buffer import BufferCache,reset_registry
    selected=json.loads((SOURCE/'selected.json').read_text())
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if not set(args.layers).issubset(set(range(32))): raise ValueError('Invalid layers')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol={'selected_sha256':sha(SOURCE/'selected.json'),'model':str(args.model.resolve()),
              'layers':args.layers,'steps':args.steps,'limit':args.limit,'budget_blocks':32,
              'block_size':16,'candidate_pool':args.candidate_pool,'max_swaps':args.max_swaps,
              'trajectory':'Full attention model on forced Full reference token IDs',
              'reference':'same post-RoPE Full Q/K/V for both block sets',
              'metric':'mean over GQA heads of sum of squared output-coordinate error',
              'search':'steepest improving one-block swap; bounded local solution'}
    pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol: raise ValueError('Output protocol mismatch')
    save(pp,protocol)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    if config.rope_scaling and 'type' not in config.rope_scaling:
        config.rope_scaling['type']=config.rope_scaling['rope_type']
    model=AutoModelForCausalLM.from_pretrained(args.model,config=config,torch_dtype=torch.bfloat16,
                                               device_map={'':'cuda:0'},attn_implementation='sdpa',
                                               local_files_only=True).eval()
    h=config.num_key_value_heads; g=config.num_attention_heads//h; d=config.hidden_size//config.num_attention_heads
    for case_index in range(args.limit):
        target=args.output/f'case_{case_index:03d}.json'
        if target.exists():
            print('skip',case_index,flush=True);continue
        row=selected[case_index]
        old=json.loads((SOURCE/f'quality_{case_index:03d}.json').read_text())
        if row['input_sha256']!=old['input_sha256']: raise ValueError('Input mismatch')
        ids=row['input_ids']
        reset_registry()
        cache=BufferCache(len(ids)+args.steps+32)
        for start in range(0,len(ids),1024):
            output=model.model(input_ids=torch.tensor([ids[start:start+1024]],device='cuda'),
                               past_key_values=cache,use_cache=True)
        first=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
        reference=old['methods']['full']['ids']
        if int(first.argmax())!=reference[0]: raise ValueError('Full first token mismatch')
        trace=[]; original=[]; current=[len(ids)]
        for layer_index in args.layers:
            module=model.model.layers[layer_index].self_attn
            original.append((module,module.forward))
            def wrapper(module,hidden_states,position_embeddings,attention_mask=None,
                        past_key_value=None,cache_position=None,_layer=layer_index,**kwargs):
                result=next(f for m,f in original if m is module)(hidden_states,position_embeddings,
                          attention_mask,past_key_value=past_key_value,cache_position=cache_position,**kwargs)
                q=module.q_proj(hidden_states).view(1,1,-1,d).transpose(1,2)
                k=module.k_proj(hidden_states).view(1,1,-1,d).transpose(1,2)
                q,_=apply_rotary_pos_emb(q,k,*position_embeddings)
                keys,values=cache.buffers[_layer]
                pos=current[0]
                mass,weighted,full=contributions(q[0,:,0].reshape(h,g,d),
                       keys[0,:,:pos+1],values[0,:,:pos+1],float(module.scaling))
                initial=mass_oracle(mass,32)
                for head in range(h):
                    chosen,swaps=swap_search(mass[head],weighted[head],initial[head],
                         max_swaps=args.max_swaps,candidate_pool=args.candidate_pool)
                    before=reconstructed(mass[head],weighted[head],initial[head])
                    after=reconstructed(mass[head],weighted[head],chosen)
                    mass_before=float(mass[head].mean(0)[initial[head]].sum()+mass[head,:,-1].mean())
                    mass_after=float(mass[head].mean(0)[chosen].sum()+mass[head,:,-1].mean())
                    trace.append({'step':pos-len(ids),'layer':_layer,'kv_head':head,
                                  'mass_mse':float((before-full[head]).square().sum(-1).mean()),
                                  'value_mse':float((after-full[head]).square().sum(-1).mean()),
                                  'mass_coverage':mass_before,'value_coverage':mass_after,
                                  'swaps':len(swaps),'overlap':len(set(chosen.tolist())&set(initial[head].tolist()))/32})
                return result
            module.forward=types.MethodType(wrapper,module)
        try:
            token=torch.empty((1,1),dtype=torch.long,device='cuda')
            position=torch.empty((1,),dtype=torch.long,device='cuda')
            start=time.perf_counter()
            steps=min(args.steps,len(reference))
            for step in range(steps):
                current[0]=len(ids)+step
                token.fill_(reference[step]);position.fill_(current[0])
                output=model.model(input_ids=token,position_ids=position[None],cache_position=position,
                                   past_key_values=cache,use_cache=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
        finally:
            for module,fn in original: module.forward=fn
        result={'index':case_index,'task':row['task'],'input_sha256':row['input_sha256'],
                'steps':steps,'layers':args.layers,'seconds_oracles_included':elapsed,'records':trace,
                'summary':{'mass_mse':sum(r['mass_mse'] for r in trace)/len(trace),
                           'value_mse':sum(r['value_mse'] for r in trace)/len(trace),
                           'mass_coverage':sum(r['mass_coverage'] for r in trace)/len(trace),
                           'value_coverage':sum(r['value_coverage'] for r in trace)/len(trace),
                           'mean_swaps':sum(r['swaps'] for r in trace)/len(trace),
                           'mean_overlap':sum(r['overlap'] for r in trace)/len(trace)}}
        save(target,result)
        print('case',case_index,result['summary'],flush=True)
        del cache
        reset_registry()
        torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--limit',type=int,default=1)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--layers',type=int,nargs='+',default=[0,8,16,24,31])
    parser.add_argument('--candidate-pool',type=int,default=256)
    parser.add_argument('--max-swaps',type=int,default=4)
    run(parser.parse_args())
