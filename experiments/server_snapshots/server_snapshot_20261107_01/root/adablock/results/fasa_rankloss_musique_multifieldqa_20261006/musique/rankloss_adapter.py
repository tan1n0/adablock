"""HF4.43 adapter: native projection/RoPE/full cache; frozen learned selector; FA2."""
from pathlib import Path
import sys,inspect,hashlib,torch
W=Path(__file__).parent
sys.path.insert(0,str(W/"snapshot"))
from models.block_ranker import load_ranker
import models.layer_head_sparse_qwen as legacy
from utils.qk_block_oracle import build_qk_block_digests,QKBlockDigests
from utils.ranking_data import check_metadata,model_signature
from utils.block_oracle import make_block_ranges,fixed_and_middle_blocks
import transformers.models.qwen2.modeling_qwen2 as hf
CKPT=Path("/root/adablock/checkpoints/ranker_all28_curve_20261005/best.pt")
CKPT_SHA="d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae"
def forbidden(*args,**kwargs): raise RuntimeError("COBS baseline fallback forbidden")
legacy.score_qk_block_digests=forbidden

def state(layers,ranker):
 c=object.__new__(legacy.LayerHeadQKSparseController)
 c.layers=list(layers);c.block_size=16;c.sink_blocks=2;c.local_window_blocks=32
 c.profile=False;c.timings={};c.ranker=ranker;c.ranker_layers=set(range(28));c._rankers_by_device={}
 c.qk_score_selector="cobs";c.qk_layer_indices=set(range(28));c.cobs_rank=4;c.qk_digests=[None]*28
 c._max_blocks=66;c._requested_middle_k=32;c._fill_to_budget=False;c._reuse_previous=False
 c._previous_selected_by_head={};c._step_reused_heads=0
 return c

def clear_stats(c):
 c._step_block_counts=[];c._step_token_counts=[];c._step_union_blocks=set();c._step_union_tokens=set()

def update_digest(c,layer,compact_key,n):
 old=c.qk_digests[layer];bid=n//16
 new=build_qk_block_digests(compact_key[0,:,bid*16:,:],16,cobs_rank=4)
 fields={}
 for f in old.__dataclass_fields__:
  a=getattr(old,f);b=getattr(new,f);dim=0 if f=="counts" else 1
  if bid>=old.mean.shape[1]: fields[f]=torch.cat([a,b],dim=dim)
  else:
   v=a.clone()
   if f=="counts":v[bid]=b[0]
   else:v[:,bid]=b[:,0]
   fields[f]=v
 c.qk_digests[layer]=QKBlockDigests(**fields)

def choose(c,layer,q,compact_key):
 n=compact_key.shape[2]-1;c._history_len=n;clear_stats(c)
 if c.qk_digests[layer] is None:
  c.qk_digests[layer]=build_qk_block_digests(compact_key[0,:,:n,:],16,cobs_rank=4)
 assert int(c.qk_digests[layer].counts.sum().item())==n
 assert c.ranker is not None and layer in c.ranker_layers
 ids=c._select_token_indices(layer,q.transpose(1,2))
 assert ids.shape[0]==4 and ids.dtype==torch.long
 return ids

def gather(compact,ids):
 ix=ids.to(compact.device)[None,:,:,None].expand(1,4,ids.shape[1],128)
 chosen=torch.gather(compact,2,ix)
 return hf.repeat_kv(torch.cat([chosen,compact[:,:,-1:,:]],dim=2),7).transpose(1,2).contiguous()

def budget_check(ids,n):
 ranges=make_block_ranges(n,16)
 sink,local,middle=fixed_and_middle_blocks(len(ranges),2,32,len(ranges)-1)
 fixed=sink|local;take=min(32,max(0,min(66,len(ranges))-len(fixed)),len(middle))
 sets=[]
 for row in ids.tolist():
  assert row==sorted(set(row)) and row[0]>=0 and row[-1]<n
  blocks={t//16 for t in row}
  assert fixed<=blocks and len(blocks-fixed)==take
  assert all((b+1)*16<=n for b in blocks-fixed)
  assert row==[t for b in sorted(blocks) for t in range(*ranges[b])]
  sets.append(set(row))
 return dict(history=n,historical_tokens=ids.shape[1],total_with_self=ids.shape[1]+1,
             fixed_blocks=len(fixed),middle_blocks=take,selected_blocks=len(fixed)+take,
             kv_union=len(set.union(*sets)),self_count=1)

class Observer:
 def __init__(self,variant):
  assert variant in ("layer0_full","all_layers");self.variant=variant
  assert hashlib.sha256(CKPT.read_bytes()).hexdigest()==CKPT_SHA
  self.ranker,self.payload=load_ranker(CKPT);assert self.payload["training"]["step"]==50000
  self.learned=set(range(1 if variant=="layer0_full" else 0,28))
  self.native=hf.Qwen2FlashAttention2.forward;self.fa=hf._flash_attention_forward
  self.sig=inspect.signature(self.native);self.active_layer=None;self.c=None;self.model=None
 def prepare(self):
  observer=self
  def forward(module,*args,**kwargs):
   bound=observer.sig.bind_partial(module,*args,**kwargs).arguments
   hs=bound["hidden_states"];ql=hs.shape[1];layer=module.layer_idx
   assert hs.shape[0]==1 and hs.dtype==torch.bfloat16
   cache=bound.get("past_key_value");assert cache is not None
   n=cache.get_seq_length(layer)
   pos=bound["position_ids"];assert pos.shape[-1]==ql
   assert torch.equal(pos[0],torch.arange(n,n+ql,device=pos.device))
   cp=bound.get("cache_position")
   if cp is not None: assert torch.equal(cp,torch.arange(n,n+ql,device=cp.device))
   assert observer.active_layer is None
   observer.active_layer=layer;observer.context=(n,ql)
   result=observer.native(module,*args,**kwargs)
   assert cache.get_seq_length(layer)==n+ql
   observer.active_layer=None
   return result
  forward.__name__="rankloss_native_forward"
  hf.Qwen2FlashAttention2.forward=forward
  def flash(q,k,v,attention_mask,query_length,*args,**kwargs):
   return observer.flash(q,k,v,attention_mask,query_length,*args,**kwargs)
  hf._flash_attention_forward=flash
 def attach(self,model):
  self.model=model
  meta=self.payload["data_metadata"]
  observed=dict(meta,model_signature=model_signature(model.config),block_size=16,
   sink_blocks=2,local_window_blocks=32,cobs_rank=4,query_scaling="attention_scale_applied_once",
   rope="post_rope",gqa="mean_of_per_query_softmax_block_mass")
  check_metadata(meta,observed)
  assert set(meta["layers"])==set(range(28))
  for x in model.model.layers:
   x.self_attn.scaling=x.self_attn.head_dim**-0.5
  self.c=state(model.model.layers,self.ranker)
  orig=legacy.LayerHeadQKSparseController._learned_block_scores
  def counted(layer,query,digests,ids):
   assert layer in self.learned and query.shape==(4,7,128)
   result=orig(self.c,layer,query,digests,ids)
   assert torch.isfinite(result[:,ids]).all().item()
   self.scoring[layer]+=1;self.ranker_calls[layer]+=4
   return result
  self.c._learned_block_scores=counted
 def begin(self,index,input_length):
  self.index=index;self.input_length=input_length;self.c.qk_digests=[None]*28
  self.c._previous_selected_by_head={}
  self.prefill=[0]*28;self.decode=[0]*28;self.selected=[0]*28
  self.scoring=[0]*28;self.ranker_calls=[0]*28;self.budgets=[];self.first_examples={}
 def flash(self,q,k,v,mask,ql,*args,**kwargs):
  layer=self.active_layer;assert layer is not None
  n,expectedql=self.context
  assert ql==expectedql==q.shape[1]
  assert q.shape==(1,ql,28,128) and k.shape==v.shape==(1,n+ql,28,128)
  assert q.device==k.device==v.device
  assert q.dtype==k.dtype==v.dtype==torch.bfloat16
  assert mask is None and kwargs.get("dropout",0)==0
  assert kwargs.get("sliding_window") is None and kwargs.get("is_causal")==True
  assert kwargs.get("use_top_left_mask")==False
  if ql>1:
   assert n==0 and ql==self.input_length
   self.prefill[layer]+=1
   return self.fa(q,k,v,mask,ql,*args,**kwargs)
  self.decode[layer]+=1
  assert n==self.input_length+self.decode[layer]-1
  if layer not in self.learned:return self.fa(q,k,v,mask,ql,*args,**kwargs)
  compact_k=k[:,:,::7,:].transpose(1,2)
  compact_v=v[:,:,::7,:].transpose(1,2)
  # Native repeat_kv creates seven consecutive query heads per KV head.
  if self.decode[layer]==1:
   assert torch.equal(k,hf.repeat_kv(compact_k,7).transpose(1,2))
   assert torch.equal(v,hf.repeat_kv(compact_v,7).transpose(1,2))
  ids=choose(self.c,layer,q,compact_k)
  audit=budget_check(ids,n);self.budgets.append(audit);self.selected[layer]+=1
  if layer not in self.first_examples:
   self.first_examples[layer]=dict(**audit,indices_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
      first_blocks_per_kv=[sorted(set(x//16 for x in row)) for row in ids.tolist()],device=str(q.device))
  selected_k=gather(compact_k,ids);selected_v=gather(compact_v,ids)
  result=self.fa(q,selected_k,selected_v,None,1,*args,**kwargs)
  assert torch.isfinite(result).all().item()
  update_digest(self.c,layer,compact_k,n)
  return result
 def finish(self,new_tokens):
  expected=new_tokens-1
  assert self.prefill==[1]*28 and self.decode==[expected]*28
  target=[expected if i in self.learned else 0 for i in range(28)]
  assert self.selected==target and self.scoring==target
  assert self.ranker_calls==[4*x for x in target]
  assert all(b["selected_blocks"]==66 and b["middle_blocks"]==32 for b in self.budgets)
  return dict(variant=self.variant,prefill_calls=self.prefill,decode_calls=self.decode,
   learned_selector_calls=self.selected,learned_score_calls=self.scoring,ranker_forward_calls=self.ranker_calls,
   fallback_calls=0,full_cache_preserved=True,post_rope_native=True,positions_checked=True,
   GQA_group=7,sink_blocks=2,local_blocks=32,middle_blocks=32,block_size=16,
   historical_tokens_min=min((b["historical_tokens"] for b in self.budgets),default=None),
   historical_tokens_max=max((b["historical_tokens"] for b in self.budgets),default=None),
   total_with_self_min=min((b["total_with_self"] for b in self.budgets),default=None),
   total_with_self_max=max((b["total_with_self"] for b in self.budgets),default=None),
   kv_union_min=min((b["kv_union"] for b in self.budgets),default=None),
   kv_union_max=max((b["kv_union"] for b in self.budgets),default=None),
   examples=self.first_examples)
