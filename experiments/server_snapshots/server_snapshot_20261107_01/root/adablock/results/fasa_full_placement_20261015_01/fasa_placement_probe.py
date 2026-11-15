import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch, transformers, flash_attn
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE=Path('/root/adablock/results/fasa_full_reproduction_20261005_multigpu')
MODEL='/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
sys.path.insert(0,str(BASE/'patched_source'))
import longbench.longbench_pred as official
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['four','single','author']);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/(a.mode+'.json')).exists()
    assert transformers.__version__=='4.43.0' and flash_attn.__version__=='2.7.4.post1'
    assert torch.cuda.device_count()==4
    assert all(torch.cuda.mem_get_info(i)[0]>23*1024**3 for i in range(4))
    official.seed_everything(42);torch.set_num_threads(4)
    refs=[json.loads(x) for x in (BASE/'run/predictions.jsonl').read_text().splitlines()]
    data=official.read_jsonl('/root/adablock/data/longbench/hotpotqa.jsonl')
    before=sha(BASE/'run/predictions.jsonl')
    kw=dict(torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',trust_remote_code=False,local_files_only=True)
    if a.mode!='author':
        kw.update(low_cpu_mem_usage=True,device_map=json.loads((BASE/'placement_plan.json').read_text())['device_map'] if a.mode=='four' else {'':0})
    model=AutoModelForCausalLM.from_pretrained(MODEL,**kw)
    if a.mode=='author': model=model.to('cuda')
    model.eval()
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False,local_files_only=True);tok.pad_token_id=tok.eos_token_id
    assert all(type(layer.self_attn).__name__=='Qwen2FlashAttention2' for layer in model.model.layers)
    assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
    meta=dict(mode=a.mode,torch=torch.__version__,transformers=transformers.__version__,flash_attn=flash_attn.__version__,device_map=getattr(model,'hf_device_map',{'':'cuda:0'}),samples=[129,50],source_sha256=sha(official.__file__),original_predictions_sha256=before,script_sha256=sha(__file__))
    save(a.out/(a.mode+'_manifest.json'),meta)
    capture={}; current={}
    def head_hook(module,inputs,output):
        if 'raw' not in capture: capture['raw']=output[0,-1,:].detach().float().cpu().clone()
    hook=model.lm_head.register_forward_hook(head_hook)
    old_processors=model._get_logits_processor
    def processors(*args,**kwargs):
        ps=old_processors(*args,**kwargs)
        capture['processors']=[type(p).__name__ for p in ps]
        assert capture['processors']==['RepetitionPenaltyLogitsProcessor','MinLengthLogitsProcessor']
        assert ps[0].penalty==1.05
        class Record:
            def __call__(self,input_ids,scores):
                if 'processed' not in capture: capture['processed']=scores[0].detach().float().cpu().clone()
                return scores
        ps.append(Record());return ps
    model._get_logits_processor=processors
    old_generate=model.generate
    def generate(*args,**kwargs):
        ids=kwargs['input_ids']; n=ids.shape[-1]
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()==current['ref']['input_ids_sha256']
        assert kwargs['max_new_tokens']==32 and kwargs['do_sample']==False and kwargs['num_beams']==1 and kwargs['min_length']==n+1
        output=old_generate(*args,**kwargs)
        capture['token_ids']=output[0,n:].cpu().tolist()
        return output
    model.generate=generate
    records=[]
    for i in [129,50]:
        capture.clear();current['ref']=refs[i]
        for d in range(4): torch.cuda.reset_peak_memory_stats(d)
        started=time.monotonic()
        result=official.get_pred_single_gpu([data[i]],127500,32,official.dataset2prompt['hotpotqa'],'hotpotqa','Qwen2.5-7B-Instruct',str(a.out/(a.mode+'_official.jsonl')),tok,model)[0]
        vectors={k:capture[k] for k in ['raw','processed']}
        torch.save(vectors,a.out/f'{a.mode}_{i}_logits.pt')
        tops={}
        for name,v in vectors.items():
            values,ids=torch.topk(v,5)
            tops[name]=dict(ids=ids.tolist(),scores=values.tolist(),top2_margin=(values[0]-values[1]).item())
        row=dict(index=i,id=data[i]['_id'],prediction=result['pred'],token_ids=capture['token_ids'],matches_retained_tokens=capture['token_ids']==refs[i]['token_ids'],retained_tokens=refs[i]['token_ids'],F1=official.scorer('hotpotqa',[result['pred']],[data[i]['answers']],data[i]['all_classes']),retained_F1=refs[i]['F1'],first_logits=tops,processors=capture['processors'],seconds=time.monotonic()-started,peak_reserved_mib=[torch.cuda.max_memory_reserved(d)/2**20 for d in range(4)])
        save(a.out/f'{a.mode}_{i}.json',row);records.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if a.mode=='four': assert row['matches_retained_tokens'],'Baseline did not reproduce'
        torch.cuda.empty_cache()
    hook.remove()
    assert sha(BASE/'run/predictions.jsonl')==before
    save(a.out/(a.mode+'.json'),dict(records=records,original_unchanged=True))
if __name__=='__main__': main()
