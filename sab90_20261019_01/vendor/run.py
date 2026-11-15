import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from prepare import ROOT, PILOT, MODEL, MAX_NEW, sha
from runtime import AdaptiveLlama, BudgetClassifier, build_prefix_digests, ragged_test
from models.block_ranker import load_ranker
from utils.longbench_eval import token_f1


def dump(name, value):
    target = ROOT / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(target)


def sync():
    torch.cuda.synchronize()


def load():
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    config = json.loads((ROOT/'protocol.json').read_text())
    assert sha(PILOT/'ranker/best.pt') == config['ranker_sha256']
    assert sha(PILOT/'budget/best.pt') == config['budget_sha256']
    ranker, rp = load_ranker(PILOT/'ranker/best.pt','cuda:0')
    bp = torch.load(PILOT/'budget/best.pt',map_location='cpu',weights_only=True)
    budget = BudgetClassifier(bp['config']['feature_size']).cuda().eval()
    budget.load_state_dict(bp['state_dict'])
    ranker.requires_grad_(False); budget.requires_grad_(False)
    model = AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},
                                                attn_implementation='sdpa',local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    assert 128009 in eos
    return model,tokenizer,ranker,budget,eos,config


@torch.inference_mode()
def prefill(model, ids):
    sync(); start=time.perf_counter()
    output=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=True)
    logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
    pairs=tuple(output.past_key_values.to_legacy_cache())
    sync()
    return pairs,logits,time.perf_counter()-start


@torch.inference_mode()
def decode(model,pairs,first_logits,mode,ranker,budget,digests,eos,quantile=.95,max_new=MAX_NEW,
           ignore_eos=False,capture=False,return_logits=False):
    generated=[int(first_logits.argmax())]
    logits_trace=[first_logits.cpu()] if return_logits else None
    controller=None
    if mode=='full':
        cache=DynamicCache.from_legacy_cache(pairs)
    else:
        controller=AdaptiveLlama(model,pairs,digests,ranker,budget,mode,quantile,capture)
        controller.install()
    sync(); start=time.perf_counter()
    try:
        for step in range(1,max_new):
            if generated[-1] in eos and not ignore_eos:
                break
            position=pairs[0][0].shape[2]+step-1
            kwargs=dict(input_ids=torch.tensor([[generated[-1]]],device='cuda:0'),
                        position_ids=torch.tensor([[position]],device='cuda:0'),
                        cache_position=torch.tensor([position],device='cuda:0'))
            if mode=='full':
                output=model.model(**kwargs,past_key_values=cache,use_cache=True)
            else:
                controller.begin_step()
                output=model.model(**kwargs,use_cache=False)
                controller.finish_step()
            logits=model.lm_head(output.last_hidden_state[:,-1:])[0,-1].float()
            generated.append(int(logits.argmax()))
            if return_logits:
                logits_trace.append(logits.cpu())
        sync(); elapsed=time.perf_counter()-start
    finally:
        if controller is not None:
            controller.uninstall()
    steps=controller.steps if controller else [{'position':pairs[0][0].shape[2]+i,
               'read_fraction':1.,'mean_selected_tokens':pairs[0][0].shape[2]+i+1,
               'mean_middle_blocks':max(0,(pairs[0][0].shape[2]+i+15)//16-34)} for i in range(len(generated)-1)]
    result=dict(ids=generated,decode_seconds=elapsed,decode_steps=len(generated)-1,
                seconds_per_decode_step=elapsed/max(1,len(generated)-1),
                mean_read_fraction=sum(r['read_fraction'] for r in steps)/max(1,len(steps)),
                mean_selected_tokens=sum(r['mean_selected_tokens'] for r in steps)/max(1,len(steps)),
                mean_middle_blocks=sum(r['mean_middle_blocks'] for r in steps)/max(1,len(steps)),
                ended_with_eos=generated[-1] in eos,hit_generation_limit=len(generated)==max_new and generated[-1] not in eos,
                steps=steps,audit=controller.audit if controller else None)
    if return_logits:
        result['_logits']=logits_trace
    if capture:
        result['_controller']=controller
    return result


def preflight():
    report={'ragged_mask_max_abs_error':ragged_test(),'full_parity':[]}
    model,tok,ranker,budget,eos,config=load()
    rows=json.loads((ROOT/'selected.json').read_text())
    for i in [0,len(rows)-1]:
        ids=rows[i]['input_ids']
        ids=ids[:512]+ids[-512:]
        pairs,logits,_=prefill(model,ids)
        reference=decode(model,pairs,logits,'full',ranker,budget,None,eos,max_new=6,ignore_eos=True,return_logits=True)
        all_read=decode(model,pairs,logits,'all',ranker,budget,None,eos,max_new=6,ignore_eos=True,return_logits=True)
        assert reference['ids']==all_read['ids'],(reference['ids'],all_read['ids'])
        errors=[float((a-b).abs().max()) for a,b in zip(reference.pop('_logits'),all_read.pop('_logits'))]
        report['full_parity'].append(dict(source_index=rows[i]['source_index'],tokens=reference['ids'],max_logit_error=max(errors)))
        dump('preflight_partial.json',report)
        assert max(errors) <= .5, errors
    digests=build_prefix_digests(pairs)
    dynamic=decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=config['budget_quantile'],
                   max_new=18,ignore_eos=True,capture=True)
    controller=dynamic.pop('_controller')
    assert all(n==17 for n in controller.audit['ranker_calls'])
    assert all(n==17 for n in controller.audit['budget_calls'])
    assert controller.audit['ragged_attention_calls']>0
    assert controller.audit['digest_updates']==32
    scene,score,features,cls=controller.feature_probe
    with torch.inference_mode():
        independent=ranker(scene).cpu()
    assert torch.allclose(independent,score,atol=2e-4,rtol=2e-4)
    report['active_adaptive_path']=dynamic['audit']
    report['ranker_batch_max_error']=float((independent-score).abs().max())
    # Every appended summary must match a fresh direct computation.
    from runtime import compact_digest
    for layer in [0,15,31]:
        block=compact_digest(controller.pairs[layer][0][0,:,1024:1040])
        digest=controller.digests[layer]
        assert torch.equal(digest['mean'][:,64:65],block['mean'])
        q=torch.randn(8,4,128,device='cuda:0')
        def quadratic(basis,eigen):
            return ((torch.einsum('hgd,hbrd->hgbr',q,basis)**2)*eigen[:,None]).sum(-1)
        assert torch.allclose(quadratic(digest['basis'][:,64:65],digest['eigenvalues'][:,64:65]),
                              quadratic(block['basis'],block['eigenvalues']),atol=1e-4,rtol=1e-4)
    report['completed_block_update']='passed'
    # Exercise exact production length and one adaptive decode step for memory.
    del controller,dynamic,digests,pairs,reference,all_read
    gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    longest=max(rows,key=lambda r:r['prompt_tokens'])
    pairs,logits,_=prefill(model,longest['input_ids'])
    digests=build_prefix_digests(pairs)
    probe=decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=config['budget_quantile'],max_new=3,ignore_eos=True)
    report['production_length_probe']={'tokens':len(longest['input_ids']),
             'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'mean_read_fraction':probe['mean_read_fraction']}
    report['passed']=True
    dump('preflight.json',report)
    print(json.dumps(report),flush=True)


def worker(shard):
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    model,tok,ranker,budget,eos,config=load()
    rows=json.loads((ROOT/'selected.json').read_text())
    journal=ROOT/f'worker_{shard}.jsonl'
    if journal.exists():
        raise FileExistsError(journal)
    # Warm kernels before measured examples.
    prefill(model,rows[0]['input_ids'][:128])
    for number,row in enumerate(rows):
        if number%4!=shard:
            continue
        torch.cuda.reset_peak_memory_stats()
        pairs,logits,prefill_seconds=prefill(model,row['input_ids'])
        sync(); start=time.perf_counter()
        digests=build_prefix_digests(pairs)
        sync(); index_seconds=time.perf_counter()-start
        # Rotate order deterministically to avoid always measuring one method first.
        methods=config['methods']
        offset=number%len(methods)
        methods=methods[offset:]+methods[:offset]
        result=dict(source_index=row['source_index'],source_id=row['source_id'],answers=row['answers'],
                    input_ids_sha256=row['input_ids_sha256'],prompt_tokens=row['prompt_tokens'],
                    original_prompt_tokens=row['original_prompt_tokens'],truncated=row['truncated'],
                    prefill_seconds=prefill_seconds,index_seconds=index_seconds,shard=shard,methods={})
        for method in methods:
            record=decode(model,pairs,logits,method,ranker,budget,digests,eos,quantile=config['budget_quantile'])
            record['prediction']=tok.decode(record['ids'],skip_special_tokens=True).strip()
            record['f1']=100*max(token_f1(record['prediction'],answer) for answer in row['answers'])
            record['total_seconds']=prefill_seconds+(0 if method=='full' else index_seconds)+record['decode_seconds']
            if method!='full' and record['decode_steps']:
                assert all(n==record['decode_steps'] for n in record['audit']['ranker_calls'])
                if method=='adaptive':
                    assert all(n==record['decode_steps'] for n in record['audit']['budget_calls'])
                else:
                    assert not any(record['audit']['budget_calls'])
            result['methods'][method]=record
        result['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30
        with journal.open('a') as handle:
            handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
        dump(f'worker_{shard}_status.json',dict(stage='running',completed=number//4+1,source_index=row['source_index']))
        print(json.dumps({'sample':number,'source_index':row['source_index'],'prefill_seconds':prefill_seconds,
                          'index_seconds':index_seconds,'scores':{k:r['f1'] for k,r in result['methods'].items()}}),flush=True)
        del pairs,logits,digests,result
        gc.collect();torch.cuda.empty_cache()
    dump(f'worker_{shard}_status.json',{'stage':'complete'})


def launch():
    if (ROOT/'status.json').exists():
        raise FileExistsError('run already started')
    dump('source_sha256.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
    dump('status.json',{'stage':'preflight'})
    with (ROOT/'preflight.log').open('w') as handle:
        subprocess.run([sys.executable,'-u',__file__,'preflight'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                       stdout=handle,stderr=subprocess.STDOUT,check=True)
    start_workers()


def start_workers():
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert (ROOT/'ablation_results.json').exists(), 'Complete the separate ablation before timed workers'
    assert not list(ROOT.glob('worker_*.jsonl')), 'Preserve existing generation results'
    dump('source_sha256_frozen.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
    dump('status.json',{'stage':'hotpot50_running'})
    workers=[];handles=[]
    for shard in range(4):
        handle=(ROOT/f'worker_{shard}.log').open('w');handles.append(handle)
        workers.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(shard)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
            stdout=handle,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for h in handles:h.close()
    if any(codes):
        raise RuntimeError(f'Worker exits: {codes}')
    dump('status.json',{'stage':'generation_complete'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['launch','preflight','worker','start_workers']);p.add_argument('--shard',type=int)
    args=p.parse_args()
    try:
        worker(args.shard) if args.stage=='worker' else globals()[args.stage]()
    except Exception:
        dump(f'error_{args.stage}_{args.shard}.json',{'error':traceback.format_exc()})
        raise
