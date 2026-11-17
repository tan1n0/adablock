"""Measure Full Llama decoding on the forced sequence used by selector profiles.

Same model, PyTorch SDPA, 32K selected RULER inputs, and 16 forced tokens.
Each repeat starts from a fresh Full prefill; prefill is outside decode timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from transformers import AutoConfig,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01'
BUFFER=ROOT/'experiments/server_snapshots/ruler_autonomous_20261022_01'
MODEL=Path('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb')
OUTPUT=ROOT/'results/value_aware_selector_20261116/full_forced_gpu'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    sys.path.insert(0,str(BUFFER))
    from cache_buffer import BufferCache,reset_registry
    selected=json.loads((SOURCE/'selected.json').read_text())
    protocol={'source_sha256':sha(SOURCE/'selected.json'),'model':str(args.model.resolve()),
              'attention':'Full PyTorch SDPA','forced_steps':16,'repeats':args.repeats,
              'case_limit':args.limit,'decode_timing':'GPU synchronized wall time; one Full prefill, prompt prefix reused and generated suffix reset between repeats',
              'forced_tokens':'same Full ids then prompt suffix as selector profiler',
              **({'include_lm_head_argmax':True} if args.include_lm_head else {})}
    pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol mismatch')
    save(pp,protocol)
    cfg=AutoConfig.from_pretrained(args.model,local_files_only=True)
    if cfg.rope_scaling and 'type' not in cfg.rope_scaling:
        cfg.rope_scaling['type']=cfg.rope_scaling['rope_type']
    model=AutoModelForCausalLM.from_pretrained(args.model,config=cfg,torch_dtype=torch.bfloat16,
              device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    for index in range(args.limit):
        target=args.output/f'case_{index:03d}.json'
        if target.exists(): print('skip',index,flush=True);continue
        row=selected[index]
        old=json.loads((SOURCE/f'quality_{index:03d}.json').read_text())
        if row['input_sha256']!=old['input_sha256']:raise ValueError('Input hash mismatch')
        full_ids=old['methods']['full']['ids']
        forced=(full_ids[:-1]+row['input_ids'][-16:])[:16]
        if len(forced)!=16:raise ValueError('Need 16 forced tokens')
        reset_registry();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        cache=BufferCache(len(row['input_ids'])+32)
        torch.cuda.synchronize();begin=time.perf_counter()
        for start in range(0,len(row['input_ids']),1024):
            output=model.model(input_ids=torch.tensor([row['input_ids'][start:start+1024]],device='cuda'),
                               past_key_values=cache,use_cache=True)
        first=int(model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float().argmax())
        if first!=full_ids[0]:raise ValueError(f'Full first token mismatch: {index}')
        torch.cuda.synchronize();prefill=time.perf_counter()-begin
        del output
        token=torch.empty((1,1),dtype=torch.long,device='cuda')
        position=torch.empty((1,),dtype=torch.long,device='cuda')
        samples=[];peaks=[];reserved=[]
        for rep in range(args.repeats+1):
            # Prompt prefix is immutable; only the generated suffix is reset.
            for layer_index,layer in enumerate(cache.layers):
                k,v=cache.buffers[layer_index]
                layer.keys=k[...,:len(row['input_ids']),:]
                layer.values=v[...,:len(row['input_ids']),:]
            torch.cuda.synchronize();begin=time.perf_counter()
            for step,forced_id in enumerate(forced):
                token.fill_(forced_id);position.fill_(len(row['input_ids'])+step)
                output=model.model(input_ids=token,position_ids=position[None],cache_position=position,
                                   past_key_values=cache,use_cache=True)
                if args.include_lm_head:
                    logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
                    _=int(logits.argmax())
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            if rep:
                samples.append(1000*seconds/16)
                peaks.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            del output
        del cache,token,position
        reset_registry();torch.cuda.empty_cache()
        case={'index':index,'task':row['task'],'input_sha256':row['input_sha256'],
              'forced_ms_per_step':samples,'median_forced_ms_per_step':statistics.median(samples),
              'peak_allocated_mib':peaks,'peak_reserved_mib':reserved,
              'prefill_seconds':prefill}
        save(target,case)
        print('case',index,'full_forced_ms',case['median_forced_ms_per_step'],flush=True)
    rows=[json.loads((args.output/f'case_{i:03d}.json').read_text()) for i in range(args.limit)]
    report={'n':len(rows),'mean_case_median_forced_ms_per_step':statistics.mean(x['median_forced_ms_per_step'] for x in rows),
            'median_peak_allocated_mib':statistics.median(statistics.median(x['peak_allocated_mib']) for x in rows),
            'median_peak_reserved_mib':statistics.median(statistics.median(x['peak_reserved_mib']) for x in rows)}
    save(args.output/f'summary_{args.limit}.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--include-lm-head',action='store_true')
    run(parser.parse_args())
