"""Official JSSA data/tokenization/moment estimator; memory-distributed HF prefill."""
import gc,sys,json,time,hashlib,importlib.util
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import transformers.models.llama.modeling_llama as llama
ROOT=Path(__file__).resolve().parent
MODEL='/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb'
spec=importlib.util.spec_from_file_location('jssa_calibrate',ROOT/'vendor_official/calibration/jssa_calibrate.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def dump(name,data):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.manual_seed(20261023)
 devices={'model.embed_tokens':0,'model.norm':3,'model.rotary_emb':0,'lm_head':3}
 devices.update({f'model.layers.{i}':i//8 for i in range(32)})
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map=devices,attn_implementation='sdpa',local_files_only=True).eval()
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 acc=official.GramAccumulator();state={'layer':None,'record':False}
 original=llama.apply_rotary_pos_emb
 def rope(*args,**kwargs):
  q,k=original(*args,**kwargs)
  if state['record']:acc(state['layer'],q,k,4)
  return q,k
 def before(layer):
  def hook(*args):state['layer']=layer
  return hook
 handles=[l.self_attn.register_forward_pre_hook(before(i)) for i,l in enumerate(model.model.layers)]
 probe=torch.tensor([[128000]+[100]*63],device='cuda:0')
 a=model.model(input_ids=probe,use_cache=False).last_hidden_state
 llama.apply_rotary_pos_emb=rope
 state['record']=True
 b=model.model(input_ids=probe,use_cache=False).last_hidden_state
 assert torch.equal(a,b),'Moment hook changed model outputs'
 assert len(acc.G_K)==32
 del a,b,probe
 acc=official.GramAccumulator()
 manifest=dict(model=MODEL,rank=32,num_sequences=8,target_tokens=131072,calibration_data_sha256=hashlib.sha256((ROOT/'longalign_calib.jsonl').read_bytes()).hexdigest(),
   method='Official tokenization, GramAccumulator and compute_basis; unchanged HF forward with post-RoPE hook',
   execution_difference='HF 4.55 SDPA; model layers distributed across four RTX3090; no KV cache, full-sequence prefill',hook_output_parity=True,samples=[])
 for idx in range(8):
  text=official.load_longalign_packed(str(ROOT/'longalign_calib.jsonl'),idx,int(131072*1.15))
  ids=official.tokenize_truncate(tok,text,131072).to('cuda:0')
  start=time.time();print('CALIBRATION_START',idx,ids.shape[1],flush=True)
  out=model.model(input_ids=ids,use_cache=False)
  del out
  for dev in range(4):torch.cuda.synchronize(dev)
  manifest['samples'].append(dict(index=idx,tokens=ids.shape[1],input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),seconds=time.time()-start))
  dump('calibration_progress.json',manifest)
  torch.save({'G_Qm':acc.G_Qm,'G_K':acc.G_K},ROOT/'grams_partial.pt')
  del ids;gc.collect();torch.cuda.empty_cache()
  print('CALIBRATION_DONE',idx,flush=True)
 state['record']=False;llama.apply_rotary_pos_emb=original
 for h in handles:h.remove()
 M=official.compute_basis(acc.G_Qm,acc.G_K,32)
 assert len(M)==32 and all(v.shape==(8,32,128) and torch.isfinite(v).all() for v in M.values())
 torch.save({'M':M,'rank':32,'model':MODEL,'objective':'TopEig(G_Qm/tr + G_K/tr)'},ROOT/'basis.pt')
 manifest['basis_sha256']=hashlib.sha256((ROOT/'basis.pt').read_bytes()).hexdigest()
 dump('calibration_complete.json',manifest)
 print('CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  dump('calibration_failed.json',dict(error=str(exc)));raise
