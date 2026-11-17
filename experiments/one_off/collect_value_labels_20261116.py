"""Collect real Full-decoding queries, block mass/contribution and swap labels.

Uses non-RULER 16K/32K Llama-tokenized sources prepared for this round.
One document is an atomic, resumable shard. The base model is frozen.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
from transformers import AutoConfig,AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.one_off.value_aware_20261116 import contributions,mass_oracle,reconstructed,swap_search
from models.cacheable_selectors import exact_block_summary

SOURCES=ROOT/'results/value_aware_selector_20261116/long_sources'
BUFFER=ROOT/'experiments/server_snapshots/ruler_autonomous_20261022_01'
MODEL=Path('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb')
OUTPUT=ROOT/'results/value_aware_selector_20261116/value_labels'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)

@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    sys.path.insert(0,str(BUFFER))
    from cache_buffer import BufferCache,reset_registry
    source=SOURCES/f'{args.split}.jsonl'
    contexts=[json.loads(line) for line in source.read_text().splitlines()]
    args.output.mkdir(parents=True,exist_ok=True)
    protocol={'split':args.split,'source_sha256':sha(source),'model':str(args.model.resolve()),
              'selected_layers':args.layers,'decode_steps':args.steps,'max_documents':args.max_documents,
              'candidate_pool':args.candidate_pool,'max_swaps':args.max_swaps,
              'budget_blocks':32,'block_size':16,'fixed':'current visible block',
              'queries':'real greedy Full-decoding queries after a Full prefill',
              'teacher':'per-query-head full attention block mass; current block mass separate',
              'summary':'exact population K mean/diagonal variance and V mean/diagonal variance of full blocks',
              'swap_labels':'initial mass top-32 context plus accepted outgoing/incoming and MSE gain',
              'contributions_saved':args.save_contributions,'base_model':'frozen'}
    pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol: raise ValueError('Protocol mismatch')
    save_json(pp,protocol)
    cfg=AutoConfig.from_pretrained(args.model,local_files_only=True)
    if cfg.rope_scaling and 'type' not in cfg.rope_scaling:
        cfg.rope_scaling['type']=cfg.rope_scaling['rope_type']
    model=AutoModelForCausalLM.from_pretrained(args.model,config=cfg,torch_dtype=torch.bfloat16,
             device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    h=cfg.num_key_value_heads;g=cfg.num_attention_heads//h;d=cfg.hidden_size//cfg.num_attention_heads
    for doc_index,context in enumerate(contexts[:args.max_documents]):
        target=args.output/f'doc_{doc_index:03d}.pt'
        if target.exists():
            print('skip',args.split,doc_index,flush=True);continue
        ids=context['input_ids']
        reset_registry();cache=BufferCache(len(ids)+args.steps+32)
        for start in range(0,len(ids),1024):
            result=model.model(input_ids=torch.tensor([ids[start:start+1024]],device='cuda'),
                               past_key_values=cache,use_cache=True)
        first=int(model.lm_head(result.last_hidden_state[:,-1:])[0,-1].float().argmax())
        current=[len(ids)]; records=[]; summaries={}; original=[]
        for layer_index in args.layers:
            module=model.model.layers[layer_index].self_attn
            original.append((module,module.forward))
            def wrapper(module,hidden_states,position_embeddings,attention_mask=None,
                        past_key_value=None,cache_position=None,_layer=layer_index,**kwargs):
                old=next(fn for m,fn in original if m is module)
                result=old(hidden_states,position_embeddings,attention_mask,
                           past_key_value=past_key_value,cache_position=cache_position,**kwargs)
                pos=current[0]
                q=module.q_proj(hidden_states).view(1,1,-1,d).transpose(1,2)
                k=module.k_proj(hidden_states).view(1,1,-1,d).transpose(1,2)
                q,_=apply_rotary_pos_emb(q,k,*position_embeddings)
                q=q[0,:,0].reshape(h,g,d)
                keys,values=cache.buffers[_layer]
                keys=keys[0,:,:pos+1];values=values[0,:,:pos+1]
                complete=pos//16
                if _layer not in summaries:
                    blocks_k=keys[:,:complete*16].reshape(h,complete,16,d)
                    blocks_v=values[:,:complete*16].reshape(h,complete,16,d)
                    summaries[_layer]=exact_block_summary(blocks_k,blocks_v).half().cpu()
                mass,weighted,full=contributions(q,keys,values,float(module.scaling))
                selected=mass_oracle(mass,32)
                for head in range(h):
                    chosen,swaps=swap_search(mass[head],weighted[head],selected[head],
                          max_swaps=args.max_swaps,candidate_pool=args.candidate_pool)
                    before=reconstructed(mass[head],weighted[head],selected[head])
                    after=reconstructed(mass[head],weighted[head],chosen)
                    item={'layer':_layer,'kv_head':head,'decode_step':pos-len(ids),
                          'token_position':pos,'query_scaled':(q[head].float()*float(module.scaling)).half().cpu(),
                          'teacher_mass_g':mass[head,:,:complete].half().cpu(),
                          'fixed_mass_g':mass[head,:,-1].half().cpu(),
                          'mass_selected':selected[head].short().cpu(),
                          'value_selected':chosen.short().cpu(),
                          'swaps':swaps,
                          'mass_mse':float((before-full[head]).square().sum(-1).mean()),
                          'value_mse':float((after-full[head]).square().sum(-1).mean())}
                    if args.save_contributions:
                        item['contribution_g']=weighted[head].half().cpu()
                    records.append(item)
                return result
            module.forward=types.MethodType(wrapper,module)
        try:
            token=torch.empty((1,1),dtype=torch.long,device='cuda')
            position=torch.empty((1,),dtype=torch.long,device='cuda')
            generated=[first]
            for step in range(args.steps):
                current[0]=len(ids)+step
                token.fill_(generated[-1]);position.fill_(current[0])
                output=model.model(input_ids=token,position_ids=position[None],cache_position=position,
                                   past_key_values=cache,use_cache=True)
                generated.append(int(model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float().argmax()))
        finally:
            for module,fn in original: module.forward=fn
        payload={'context_sha256':context['input_sha256'],'source_parts':context['source_parts'],
                 'split':args.split,'input_length':len(ids),'generated_ids':generated,
                 'summaries':summaries,'records':records,'protocol':protocol}
        tmp=target.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(target)
        print('complete',args.split,doc_index,'length',len(ids),'records',len(records),
              'swaps',sum(len(r['swaps']) for r in records),flush=True)
        # The last forward result and wrapper closure both hold the cache.
        # Release them before allocating the next 32K document on a 3090.
        del payload, summaries, records, result, output, wrapper, original
        del token, position, cache
        reset_registry()
        gc.collect()
        torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('train','validation'),required=True)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--max-documents',type=int,default=8)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--layers',type=int,nargs='+',default=[0,10,20,31])
    parser.add_argument('--candidate-pool',type=int,default=256)
    parser.add_argument('--max-swaps',type=int,default=4)
    parser.add_argument('--save-contributions',action='store_true')
    run(parser.parse_args())
