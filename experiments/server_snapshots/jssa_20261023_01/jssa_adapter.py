"""HF integration of pinned official JSSA store/score/indexed-attention kernels.

Execution adaptation, not the native vLLM performance benchmark.
"""
import sys, time, types
from pathlib import Path
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'kernel_adapter'))
from triton_jssa_store import jssa_project_and_store
from triton_jssa_score_topk import jssa_score_topk
from triton_jssa_indexed_attend import jssa_indexed_attend

class JSSA:
    def __init__(self,model,cpu_pairs,basis,capacity):
        self.model=model;self.prefix=cpu_pairs[0][0].shape[2]
        self.h=model.config.num_key_value_heads;self.d=model.config.hidden_size//model.config.num_attention_heads
        self.g=model.config.num_attention_heads//self.h
        self.nblocks=(capacity+15)//16
        self.bt=torch.arange(self.nblocks,device='cuda',dtype=torch.int32)[None]
        self.seq=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.slot=torch.zeros(1,device='cuda',dtype=torch.int64)
        self.basis={i:basis[i].to(device='cuda',dtype=torch.bfloat16).contiguous() for i in range(len(cpu_pairs))}
        self.cache=[]
        slots=torch.arange(self.prefix,device='cuda',dtype=torch.int64)
        for i,(k,v) in enumerate(cpu_pairs):
            cache=torch.empty((self.nblocks,16,self.h,2*self.d+32),device='cuda',dtype=torch.bfloat16)
            jssa_project_and_store(k[0].transpose(0,1).contiguous().cuda(),v[0].transpose(0,1).contiguous().cuda(),cache,slots,self.basis[i])
            self.cache.append(cache)
        self.original=[]
        self.scratch_buffers={}
        self.qkv=torch.empty((1,self.h,self.d),device='cuda',dtype=torch.bfloat16)
        self.projq=torch.empty((self.h,1,32),device='cuda',dtype=torch.bfloat16)
        self.scores=torch.empty((1,self.h,self.nblocks*16),device='cuda',dtype=torch.float32)
        self.output=torch.empty((1,self.h*self.g,self.d),device='cuda',dtype=torch.bfloat16)

    def scratch(self,name,shape,dtype,device):
        key=(name,tuple(shape),dtype)
        if key not in self.scratch_buffers:self.scratch_buffers[key]=torch.empty(shape,dtype=dtype,device=device)
        return self.scratch_buffers[key]

    def install(self):
        for i,layer in enumerate(self.model.model.layers):
            module=layer.self_attn;self.original.append(module.forward)
            def forward(module,hidden_states,position_embeddings,attention_mask=None,past_key_value=None,cache_position=None,_layer=i,**kwargs):
                return self.forward(module,_layer,hidden_states,position_embeddings)
            module.forward=types.MethodType(forward,module)

    def uninstall(self):
        for layer,f in zip(self.model.model.layers,self.original):layer.self_attn.forward=f
        self.original=[]

    def forward(self,module,layer,hidden,embeddings):
        shape=(1,1,-1,self.d)
        q=module.q_proj(hidden).view(shape).transpose(1,2)
        k=module.k_proj(hidden).view(shape).transpose(1,2)
        v=module.v_proj(hidden).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*embeddings)
        q=q[0,:,0].unsqueeze(0).contiguous()
        jssa_project_and_store(k[0,:,0].unsqueeze(0).contiguous(),v[0,:,0].unsqueeze(0).contiguous(),self.cache[layer],self.slot,self.basis[layer])
        torch.mean(q.view(1,self.h,self.g,self.d),dim=2,out=self.qkv)
        torch.bmm(self.qkv.transpose(0,1),self.basis[layer].transpose(1,2),out=self.projq)
        idx,_=jssa_score_topk(self.projq.transpose(0,1),self.cache[layer],self.bt,self.seq,min(self.budget,self.position+1),self.d,32,
            use_radix=True,scores_out=self.scores,top_idx_out=self.top_idx,top_scores_out=self.top_scores)
        output=jssa_indexed_attend(q,self.cache[layer],self.bt,idx,head_size=self.d,scale=float(module.scaling),out=self.output,scratch=self.scratch)
        return module.o_proj(output.reshape(1,1,-1)),None

    @torch.inference_mode()
    def decode(self,logits,eos,budget,max_new=128,ignore_eos=False,return_logits=False):
        self.budget=budget;ids=[int(logits.argmax())];trace=[logits.cpu()] if return_logits else None
        self.top_idx=self.scratch('top_idx',(1,self.h,min(budget,self.nblocks*16)),torch.int32,'cuda')
        self.top_scores=self.scratch('top_scores',self.top_idx.shape,torch.float32,'cuda')
        self.install();torch.cuda.synchronize();start=time.perf_counter()
        try:
            for step in range(1,max_new):
                if ids[-1] in eos and not ignore_eos:break
                self.position=self.prefix+step-1
                self.slot.fill_(self.position);self.seq.fill_(self.position+1)
                out=self.model.model(input_ids=torch.tensor([[ids[-1]]],device='cuda'),
                    position_ids=torch.tensor([[self.position]],device='cuda'),cache_position=torch.tensor([self.position],device='cuda'),use_cache=False)
                next_logits=self.model.lm_head(out.last_hidden_state[:,-1:])[0,-1].float()
                ids.append(int(next_logits.argmax()))
                if return_logits:trace.append(next_logits.cpu())
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
        finally:self.uninstall()
        steps=len(ids)-1
        full=sum(self.prefix+i+1 for i in range(steps));read=sum(min(budget,self.prefix+i+1) for i in range(steps))
        frac=read/full if full else None
        result=dict(ids=ids,decode_seconds=elapsed,decode_steps=steps,seconds_per_decode_step=elapsed/max(1,steps),
            mean_selected_tokens=read/max(1,steps),logical_fraction=frac,padded_fraction=frac,
            logical_kv_token_reads=read*self.h*len(self.cache),full_kv_token_reads=full*self.h*len(self.cache),
            projected_key_bytes_scanned=full*self.h*len(self.cache)*32*2,
            selected_kv_bytes=read*self.h*len(self.cache)*self.d*2*2,
            ended_with_eos=ids[-1] in eos,hit_generation_limit=len(ids)==max_new and ids[-1] not in eos)
        if return_logits:result['_logits']=trace
        return result
