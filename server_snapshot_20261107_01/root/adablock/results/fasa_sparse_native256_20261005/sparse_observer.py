"""Observe the official selector and attention calls; return every original result unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import monkey_patch.frequency.dynamic_frequency as df
import monkey_patch.frequency.qwen_df as qf
from monkey_patch.frequency.utils import constructe_adaptive_selection
W=Path(__file__).parent
stats={};context={};sample_index=None;input_length=None;digest=None;examples={}
def begin(index,n):
 global stats,context,sample_index,input_length,digest,examples
 sample_index=index;input_length=n;context={};digest=hashlib.sha256();examples={}
 stats={i:dict(prefill_calls=0,decode_calls=0,selector_calls=0,fa2_calls=0,kv_union_min=[10**9]*4,kv_union_max=[0]*4,kv_union_sum=[0]*4,all_head_union_min=10**9,all_head_union_max=0,all_head_union_sum=0,self_selected_heads=0,token0_selected_heads=0,head_token_count_min=10**9,head_token_count_max=0) for i in range(28)}
def finish(tokens):
 for i,s in stats.items():
  assert s["prefill_calls"]==1 and s["decode_calls"]==tokens-1,(i,s,tokens)
  assert s["fa2_calls"]==tokens
  assert s["selector_calls"]==(tokens-1 if i>0 else 0)
  if s["selector_calls"]:
   assert s["head_token_count_min"]==s["head_token_count_max"]==256
  else:
   s["kv_union_min"]=[None]*4;s["all_head_union_min"]=None;s["head_token_count_min"]=None
 if sample_index==0 and examples:
  (W/"run/selected_indices_sample0_first_decode.json").write_text(json.dumps(examples))
 return dict(layers=stats,selected_index_sha256=digest.hexdigest(),prefill_full=True,layer0_dense=True,selection_per_query_head=256,query_heads=28,kv_heads=4,fixed_regions=False)
def prepare():
 records=constructe_adaptive_selection("Qwen2.5-7B-Instruct",256,0.8,16,16,dataset_name="qasper")
 evidence={}
 for layer,r in records.items():
  x=r["head_selection"].cpu();m=r["selection_masks"].cpu()
  assert tuple(x.shape)==(28,32) and m.all()
  assert x.min()>=0 and x.max()<128
  assert all(len(torch.unique(row//2))==16 and len(torch.unique(row))==32 for row in x)
  evidence[layer]=dict(indices=x.tolist(),masks_all_true=True)
 (W/"frequency_selection.json").write_text(json.dumps(evidence))
 old_core=qf.core_module_with_padding_old
 def core(query,key,value,layer,budget,records):
  assert budget==256 and query.shape[1:3]==(28,1) and key.shape[1]==28
  assert layer==context["layer"] and layer>0
  grabbed={};original_topk=torch.topk
  def capture_topk(t,*a,**kw):
   out=original_topk(t,*a,**kw)
   assert len(grabbed)==0,"Unexpected extra topk in selector"
   grabbed["indices"]=out.indices.detach().cpu().numpy()
   return out
  torch.topk=capture_topk
  try:out=old_core(query,key,value,layer,budget,records)
  finally:torch.topk=original_topk
  assert len(grabbed)==1
  idx=grabbed["indices"].reshape(28,256)
  assert idx.min()>=0 and idx.max()<key.shape[2]
  assert np.all(np.diff(np.sort(idx,axis=1),axis=1)>0)
  assert tuple(out[0].shape)==tuple(out[1].shape)==(1,28,256,128)
  s=stats[layer];s["selector_calls"]+=1
  s["head_token_count_min"]=min(s["head_token_count_min"],idx.shape[1]);s["head_token_count_max"]=max(s["head_token_count_max"],idx.shape[1])
  unions=[len(np.unique(idx[h*7:(h+1)*7])) for h in range(4)]
  for h,u in enumerate(unions):
   s["kv_union_min"][h]=min(s["kv_union_min"][h],u);s["kv_union_max"][h]=max(s["kv_union_max"][h],u);s["kv_union_sum"][h]+=u
  u=len(np.unique(idx));s["all_head_union_min"]=min(s["all_head_union_min"],u);s["all_head_union_max"]=max(s["all_head_union_max"],u);s["all_head_union_sum"]+=u
  s["self_selected_heads"]+=int((idx==key.shape[2]-1).any(axis=1).sum());s["token0_selected_heads"]+=int((idx==0).any(axis=1).sum())
  digest.update(np.asarray([sample_index,layer,key.shape[2]],dtype=np.int64).tobytes());digest.update(idx.tobytes())
  if sample_index==0 and s["selector_calls"]==1:examples[layer]=dict(seq_len=key.shape[2],indices=idx.tolist(),kv_unions=unions)
  return out
 qf.core_module_with_padding_old=core
 old_fa=qf._flash_attention_forward
 def flash(q,k,v,mask,qlen,*a,**kw):
  layer=context["layer"];s=stats[layer]
  assert q.shape[2]==k.shape[2]==28 and k.shape==v.shape
  expected=input_length+s["decode_calls"] if layer==0 or qlen>1 else 256
  assert k.shape[1]==expected,(layer,qlen,k.shape[1],expected)
  assert mask is None and kw["dropout"]==0 and kw["sliding_window"] is None
  s["fa2_calls"]+=1
  return old_fa(q,k,v,mask,qlen,*a,**kw)
 qf._flash_attention_forward=flash
 old_forward=df.Qwen2FlashAttention2_forward_dynamic_frequency
 def forward(self,hidden_states,*a,**kw):
  layer=self.layer_idx;qlen=hidden_states.shape[1];s=stats[layer]
  context["layer"]=layer
  if qlen>1:
   assert qlen==input_length
   s["prefill_calls"]+=1
  else:s["decode_calls"]+=1
  return old_forward(self,hidden_states,*a,**kw)
 df.Qwen2FlashAttention2_forward_dynamic_frequency=forward
 df.replace_qwen_df(256,records)
 return records
