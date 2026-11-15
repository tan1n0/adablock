"""Inclusive host-wall/CUDA-stream event timers; original calls and values unchanged."""
import time,functools,collections,hashlib,torch
import rankloss_adapter as a
import models.block_ranker as br
import accelerate.hooks as ah
class Meter:
 def __init__(self):
  self.enabled=False;self.records=[];self.phase="outside";self.selections=[];self.svd=[];self.transfers=[]
 def start_sample(self,index):
  self.records=[];self.selections=[];self.svd=[];self.transfers=[];self.index=index;self.enabled=True
 def call(self,name,fn,args,kwargs,dev=None,meta=None):
  if not self.enabled:return fn(*args,**kwargs)
  if dev is not None:
   dev=torch.device(dev)
   if dev.type!="cuda":dev=None
  start=time.perf_counter();ev0=ev1=None
  if dev is not None:
   with torch.cuda.device(dev):
    ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
    ev0.record()
  try:return fn(*args,**kwargs)
  finally:
   if ev1 is not None:
    with torch.cuda.device(dev):ev1.record()
   self.records.append((name,time.perf_counter()-start,ev0,ev1,str(dev),meta))
 def wrap(self,obj,name,label,device_fn=None):
  orig=getattr(obj,name)
  @functools.wraps(orig)
  def fn(*args,**kwargs):
   label_now=label(*args,**kwargs) if callable(label) else label
   dev=device_fn(*args,**kwargs) if device_fn else tensor_device(args)
   return self.call(label_now,orig,args,kwargs,dev)
  setattr(obj,name,fn)
  return orig
 def finish(self):
  for d in range(4):torch.cuda.synchronize(d)
  self.enabled=False;stats={}
  for name,wall,e0,e1,dev,meta in self.records:
   r=stats.setdefault(name,dict(calls=0,host_wall_seconds=0.,cuda_stream_seconds=0.,devices={}))
   r["calls"]+=1;r["host_wall_seconds"]+=wall
   if e0 is not None:r["cuda_stream_seconds"]+=e0.elapsed_time(e1)/1000
   r["devices"][dev]=r["devices"].get(dev,0)+1
  return dict(timers=stats,selections=self.selections,svd_calls=self.svd,transfers=self.transfers,
   semantics="Timers are inclusive/nested, not additive. Host wall includes Python and blocking calls. CUDA event intervals include stream execution and idle/host-dispatch gaps, not pure kernel time. No synchronization added per component. Whole-model forward boundaries synchronize all four GPUs; original runner already synchronizes sample boundaries. Compare whole-sample wall to retained smoke to estimate perturbation plus ordinary run variation.")
def tensors(x):
 if isinstance(x,torch.Tensor):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from tensors(v)
 elif isinstance(x,(tuple,list)):
  for v in x:yield from tensors(v)
def tensor_device(x):
 return next((t.device for t in tensors(x) if t.is_cuda),None)
meter=Meter()
def install(observer):
 # Digests are initialized once per sample/layer and only current block updated thereafter.
 orig=a.build_qk_block_digests
 @functools.wraps(orig)
 def digest(keys,*args,**kwargs):
  phase="initial" if keys.shape[1]>16 else "incremental"
  old=meter.phase;meter.phase=phase
  try:return meter.call("digest."+phase,orig,(keys,*args),kwargs,keys.device)
  finally:meter.phase=old
 a.build_qk_block_digests=digest
 origsvd=torch.linalg.svd
 @functools.wraps(origsvd)
 def svd(x,*args,**kwargs):
  if meter.enabled:meter.svd.append(dict(phase=meter.phase,layer=observer.active_layer,shape=list(x.shape),device=str(x.device),driver=kwargs.get("driver","torch_default"),full_matrices=kwargs.get("full_matrices")))
  return meter.call("svd."+meter.phase,origsvd,(x,*args),kwargs,x.device)
 torch.linalg.svd=svd
 meter.wrap(a,"update_digest","digest.incremental_update_including_clone",lambda c,layer,k,n:k.device)
 origchoose=a.choose
 @functools.wraps(origchoose)
 def choose(c,layer,q,k):
  ids=meter.call("choose.total",origchoose,(c,layer,q,k),{},q.device)
  if meter.enabled:meter.selections.append(dict(layer=layer,history=k.shape[2]-1,sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
  return ids
 a.choose=choose
 meter.wrap(a,"gather","gather.indices_H2D_and_GPU_KV",lambda k,ids:k.device)
 meter.wrap(a,"budget_check","audit.budget_python",lambda *args:None)
 meter.wrap(a.legacy.LayerHeadQKSparseController,"_select_token_indices","selection.total",lambda c,l,q:q.device)
 meter.wrap(a.legacy.LayerHeadQKSparseController,"_learned_block_scores","ranker.scoring_plus_sample_gathers",lambda c,l,q,d,ids:q.device)
 meter.wrap(torch,"topk","selection.topk")
 meter.wrap(br.BlockRanker,"forward","ranker.forward",lambda m,s:s["query"].device)
 meter.wrap(br,"cobs_terms","features.cobs")
 meter.wrap(br,"position_features","features.position")
 # Hooks on copies inherited by deepcopy attach the same meter through global reference.
 for name in ("query_proj","mean_proj","interaction","output"):
  module=getattr(observer.ranker,name);original=module.forward
  # Use class hooks, not stored bound closures, so ranker deepcopy retains its own tensors.
  def before(mod,args,label=name):
   if meter.enabled:
    dev=tensor_device(args);start=time.perf_counter()
    with torch.cuda.device(dev):
     e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
    mod._profile_pending=(start,e0,e1,dev,label)
  def after(mod,args,result):
   if meter.enabled:
    start,e0,e1,dev,label=mod._profile_pending
    with torch.cuda.device(dev):e1.record()
    meter.records.append(("ranker.module."+label,time.perf_counter()-start,e0,e1,str(dev),None))
  module.register_forward_pre_hook(before);module.register_forward_hook(after)
 oldfa=observer.fa
 @functools.wraps(oldfa)
 def fa(q,k,v,*args,**kwargs):
  return meter.call("attention.fa2_prefill" if q.shape[1]>1 else "attention.fa2_decode",oldfa,(q,k,v,*args),kwargs,q.device)
 observer.fa=fa
 oldnative=observer.native
 @functools.wraps(oldnative)
 def native(module,*args,**kwargs):
  return meter.call("attention.native_including_selector",oldnative,(module,*args),kwargs,module.q_proj.weight.device)
 observer.native=native
 meter.wrap(observer,"flash","adapter.flash_including_audits")
 oldsend=ah.send_to_device
 @functools.wraps(oldsend)
 def send(x,device,*args,**kwargs):
  ts=list(tensors(x));dest=torch.device(device)
  moved=[t for t in ts if t.device!=dest]
  if not meter.enabled or not moved:return oldsend(x,device,*args,**kwargs)
  cross=any(t.is_cuda and dest.type=="cuda" for t in moved)
  label="transfer.cross_GPU" if cross else "transfer.CPU_GPU"
  meter.transfers.append(dict(kind=label,bytes=sum(t.numel()*t.element_size() for t in moved),sources=sorted({str(t.device) for t in moved}),destination=str(dest)))
  return meter.call(label,oldsend,(x,device,*args),kwargs,dest if dest.type=="cuda" else tensor_device(x))
 ah.send_to_device=send
def attach_model(model):
 orig=model.forward
 @functools.wraps(orig)
 def forward(*args,**kwargs):
  ids=kwargs.get("input_ids",args[0] if args else None)
  label="model.prefill" if ids.shape[-1]>1 else "model.decode"
  def sync_forward():
   for d in range(4):torch.cuda.synchronize(d)
   result=orig(*args,**kwargs)
   for d in range(4):torch.cuda.synchronize(d)
   return result
  return meter.call(label,sync_forward,(),{},None)
 model.forward=forward
