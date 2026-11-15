import json,time,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
import rankloss_adapter as a
W=Path(__file__).parent
torch.set_num_threads(4);torch.manual_seed(42)
ranker,payload=a.load_ranker(a.CKPT)
config=AutoConfig.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",local_files_only=True)
reports=[];start=time.perf_counter()
with torch.inference_mode():
 for n in [31,1056,1063]:
  # Same post-RoPE tensor inputs on all owning layer devices; fixed-seed synthetic tensors.
  kc=torch.randn(1,4,n+1,128,dtype=torch.float32).mul_(.4).bfloat16()
  vc=torch.randn_like(kc.float()).bfloat16()
  qc=torch.randn(1,1,28,128).mul_(.4).bfloat16()
  for device in [1,2,3]:
   dev=torch.device("cuda",device);k=kc.to(dev);v=vc.to(dev);q=qc.to(dev)
   layers=[NS(self_attn=NS(scaling=128**-.5,q_proj=NS(weight=torch.empty(0,device=dev)))) for _ in range(28)]
   fake=NS(model=NS(layers=layers),config=config)
   ref=a.legacy.LayerHeadQKSparseController(fake,[(k[:,:,:n,:],v[:,:,:n,:])],16,2,32,
         ranker=ranker,ranker_metadata=payload["data_metadata"])
   ref.begin_step(n,66,32,False)
   expected=ref._select_token_indices(0,q.transpose(1,2))
   c=a.state(layers,ranker);actual=a.choose(c,0,q,k)
   assert torch.equal(actual,expected),"Selection changed"
   budget=a.budget_check(actual,n)
   sk=a.gather(k,actual);sv=a.gather(v,actual)
   compact_sk=torch.cat([torch.gather(k[:,:,:n,:],2,actual.to(dev)[None,:,:,None].expand(1,4,actual.shape[1],128)),k[:,:,-1:,:]],2)
   assert torch.equal(sk,compact_sk.repeat_interleave(7,1).transpose(1,2))
   fa=a.hf._flash_attention_forward(q,sk,sv,None,1,dropout=0,sliding_window=None,is_causal=True,use_top_left_mask=False)
   query=q.transpose(1,2)
   logits=query.float()@sk.transpose(1,2).float().transpose(-1,-2)*(128**-.5)
   exact=(logits.softmax(-1)@sv.transpose(1,2).float()).transpose(1,2)
   sdpa=torch.nn.functional.scaled_dot_product_attention(query,sk.transpose(1,2),sv.transpose(1,2),is_causal=False,scale=128**-.5).transpose(1,2)
   err=(fa.float()-exact).abs().max().item()
   rel=((fa.float()-exact).norm()/exact.norm()).item()
   assert err<.02 and rel<.01 and (fa-sdpa).abs().max().item()<.02
   # Incremental summary update matches the prototype implementation exactly.
   a.update_digest(c,0,k,n);ref._append_current_kv(0,k[:,:,-1:,:],v[:,:,-1:,:])
   for field in c.qk_digests[0].__dataclass_fields__:
    torch.testing.assert_close(getattr(c.qk_digests[0],field),getattr(ref.qk_digests[0],field),rtol=0,atol=0)
   # Full native cache retains every key/value while gathering leaves it untouched.
   cache=DynamicCache();cache.update(k[:,:,:n,:].clone(),v[:,:,:n,:].clone(),0)
   cache.update(k[:,:,-1:,:],v[:,:,-1:,:],0)
   assert cache.get_seq_length(0)==n+1 and torch.equal(cache.key_cache[0],k) and torch.equal(cache.value_cache[0],v)
   reports.append(dict(history=n,device=str(dev),selection_exact=True,digest_update_exact=True,
    full_cache_preserved=True,GQA_exact=True,FA2_vs_fp32_max_abs=err,FA2_vs_fp32_rel_l2=rel,
    FA2_vs_SDPA_max_abs=(fa-sdpa).abs().max().item(),budget=budget,
    indices_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()))
   del c,ref,k,v,q,cache
 for n in [31,1056,1063]:
  assert len({r["indices_sha256"] for r in reports if r["history"]==n})==1
result=dict(passed=True,checkpoint_sha256=a.CKPT_SHA,adapter_sha256=hashlib.sha256((W/"rankloss_adapter.py").read_bytes()).hexdigest(),
 tests=reports,seconds=time.perf_counter()-start,
 note="Synthetic post-RoPE selector/kernel tests; native positions and 28 layer calls additionally gated by real 8-sample smoke.")
(W/"adapter_tests.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
