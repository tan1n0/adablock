import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,json,time,hashlib,traceback,gc,statistics
from pathlib import Path
ROOT=Path('/root/adablock');sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/longbench_all28_attention_coverage8_20261004'
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from scripts import eval_longbench_adablock_sparse as ev
from models.experimental_all_layer_ranker_qwen import ExperimentalAllLayerRankerController as Base
import models.layer_head_sparse_qwen as orig
from models.block_ranker import load_ranker
from utils.longbench_eval import build_model_input_text
from utils.block_oracle import make_block_ranges,fixed_and_middle_blocks
RUN=ROOT/'results/longbench_ranker_all28_hotpotqa_20261004'
CKPT=ROOT/'checkpoints/ranker_ddp_fresh_global64_20261003/best.pt'
MODEL=Path('/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
state={};measurements=[];sample_audits=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def forbidden(*a,**kw):raise RuntimeError('COBS/Quest fallback forbidden')
orig.score_qk_block_digests=forbidden
oldrope=orig.apply_rotary_pos_emb
def rope(*a,**kw):
 q,k=oldrope(*a,**kw);state['self_key']=k.detach();return q,k
orig.apply_rotary_pos_emb=rope
class Diagnostic(Base):
 def __init__(self,*a,**kw):
  kw['all_layer_ranker']=True;super().__init__(*a,**kw)
 def _select_token_indices(self,layer_idx,query_states):
  ids=super()._select_token_indices(layer_idx,query_states)
  with torch.no_grad():
   start=time.perf_counter();hist=self._history_len
   step=state['layer_steps'].get(layer_idx,0)+1;state['layer_steps'][layer_idx]=step
   ranges=make_block_ranges(hist,self.block_size);sink,local,middle=fixed_and_middle_blocks(num_blocks=len(ranges),sink_blocks=self.sink_blocks,local_window_blocks=self.local_window_blocks,current_block=len(ranges)-1)
   fixed=sink|local
   fixed_ids=[i for b in sorted(fixed) for i in range(*ranges[b])]
   q=query_states[0,:,0,:].detach().float().reshape(4,7,-1)
   k=self.full_past[layer_idx][0][0].to(q.device).float()
   sk=state['self_key'][0,:,0,:].float()
   logits=torch.einsum('hgd,hnd->hgn',q,k)*float(self.layers[layer_idx].self_attn.scaling)
   selflog=torch.einsum('hgd,hd->hg',q,sk)*float(self.layers[layer_idx].self_attn.scaling)
   probs=torch.softmax(torch.cat([logits,selflog.unsqueeze(-1)],dim=-1),dim=-1)
   assert (probs.sum(-1)-1).abs().max().item()<2e-6
   selfmass=probs[:,:,-1];fixedmass=probs[:,:,fixed_ids].sum(-1)+selfmass
   blockmass=torch.nn.functional.pad(probs[:,:,:hist],(0,len(ranges)*16-hist)).reshape(4,7,len(ranges),16).sum(-1)
   coverage=[];oracle=[];chosen=[];oracle_blocks=[]
   for h in range(4):
    chosenblocks=sorted(set((ids[h]//16).tolist()));chosenmiddle=sorted(set(chosenblocks)-fixed)
    assert fixed<=set(chosenblocks) and set(chosenmiddle)<=middle
    assert len(chosenmiddle)==min(32,len(middle))
    cov=probs[h,:,ids[h].to(q.device)].sum(-1)+selfmass[h]
    mids=sorted(middle);mass=blockmass[h].mean(0)
    picks=torch.tensor(mids,device=q.device)[torch.topk(mass[mids],len(chosenmiddle)).indices]
    oraclecov=fixedmass[h]+blockmass[h,:,picks].sum(-1)
    assert oraclecov.mean().item()+2e-6>=cov.mean().item()
    coverage.append(cov);oracle.append(oraclecov);chosen.append(chosenmiddle);oracle_blocks.append(picks.tolist())
   cov=torch.stack(coverage);orc=torch.stack(oracle)
   assert ((cov>=fixedmass-2e-6)&(cov<=1+2e-6)).all()
   if step==1:
    ref=state['reference']['audit']['first_selections'][layer_idx]
    assert ref['layer']==layer_idx and hashlib.sha256(ids[0].numpy().tobytes()).hexdigest()==ref['head0_sha256'],('selection mismatch',state['sample'],layer_idx)
   rec={'sample':state['sample'],'step':step,'layer':layer_idx,'history_tokens':hist,'selected_history_tokens':ids.shape[1],'coverage':cov.flatten().tolist(),'fixed_self':fixedmass.flatten().tolist(),'self_mass':selfmass.flatten().tolist(),'middle_increment':(cov-fixedmass).flatten().tolist(),'oracle_gqa_same_budget_coverage':orc.flatten().tolist(),'selected_middle_blocks_per_kv_head':chosen,'oracle_middle_blocks_per_kv_head':oracle_blocks,'selected_all_heads_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
   measurements.append(rec)
   with (OUT/'coverage_steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   state['diagnostic_seconds']+=time.perf_counter()-start
  return ids
ev.LayerHeadQKSparseController=Diagnostic
def quant(vals,q):
 v=sorted(vals);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def stats(v):
 if not v:return {'n':0}
 return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p10':quant(v,.1),'below90_fraction':sum(x<.9 for x in v)/len(v),'below95_fraction':sum(x<.95 for x in v)/len(v)}
def aggregate(records,group=False):
 vals=[];fixed=[];inc=[];orc=[];selfs=[]
 for r in records:
  for dest,k in [(vals,'coverage'),(fixed,'fixed_self'),(inc,'middle_increment'),(orc,'oracle_gqa_same_budget_coverage'),(selfs,'self_mass')]:
   a=r[k];dest.extend([statistics.mean(a[i:i+7]) for i in range(0,28,7)] if group else a)
 non=[i for i,x in enumerate(fixed) if x<.95]
 return {'coverage':stats(vals),'fixed_self':stats(fixed),'middle_increment_mean':statistics.mean(inc),'self_mass_mean':statistics.mean(selfs),'oracle_gqa_same_budget':stats(orc),'oracle_minus_selected_mean':statistics.mean([o-v for o,v in zip(orc,vals)]),'fixed_already95_fraction':sum(x>=.95 for x in fixed)/len(fixed),'nontrivial_fixed_below95':{'coverage':stats([vals[i] for i in non]),'oracle_coverage':stats([orc[i] for i in non]),'oracle_gap_mean':statistics.mean([orc[i]-vals[i] for i in non]) if non else None}}
def main():
 start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(42)
 assert sha(CKPT)=='18113b5cd6f34b8a9b9925694bbc616787ad77c71d40d4d0b3eecf2aedaff640'
 ref={r['index']:r for r in map(json.loads,(RUN/'learned_all28.jsonl').read_text().splitlines())}
 rows=json.loads((RUN/'selected_rows.json').read_text());cfg=json.loads((RUN/'shard_0/config.json').read_text())
 write('config.json',{'samples':list(range(8)),'all_sparse_decode_steps':True,'parent_config':cfg,'oracle':'FP32 qk softmax over sparse-trajectory full historical keys plus self, each query head separately; 7 query heads averaged per shared KV head for group oracle','denominator_includes_self':True,'fixed_includes_sink_local_self':True,'no_dense_baseline_trajectory':True,'no_fallback':True,'original_results_sha256':sha(RUN/'learned_all28.jsonl')})
 sys.argv=[sys.argv[0]];args=ev.parse_args();args.max_input_length=32640;args.max_new_tokens_override=128;args.fixed_middle_blocks=32;args.ranker_checkpoint=str(CKPT);args.ranker_untrained_layers='cobs';args.debug_max_steps=128;args.debug_logits_topk=1;args.ranker_profile=False;args.selection_mode='learned_qk';args._ranker_bundle=load_ranker(CKPT,'cpu')
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 for i in range(8):
  state.clear();state.update(sample=i,reference=ref[i],layer_steps={},diagnostic_seconds=0)
  write('status.json',{'stage':'running','sample':i,'completed':i,'elapsed_seconds':time.perf_counter()-start})
  prompt=build_model_input_text(tok,cfg['prompt'].format(**rows[i]),True)
  inp=tok(prompt,return_tensors='pt')['input_ids']
  assert hashlib.sha256(inp.numpy().tobytes()).hexdigest()==ref[i]['input_ids_sha256']
  t=time.perf_counter();pred,runstats,trace=ev.run_layer_head_qk_decode(model,tok,None,torch.device('cuda:0'),prompt,args)
  assert [x['token_id'] for x in trace]==[x['token_id'] for x in ref[i]['trace']],('token mismatch',i)
  assert pred==ref[i]['prediction']
  steps=len(trace)-1;assert state['layer_steps']=={l:steps for l in range(28)}
  audit={'sample':i,'generated_tokens':len(trace),'sparse_steps':steps,'exact_original_token_parity':True,'first_step_all28_head0_selection_hash_parity':True,'seconds':time.perf_counter()-t,'diagnostic_seconds':state['diagnostic_seconds']}
  sample_audits.append(audit);print(json.dumps(audit),flush=True)
  write('sample_audits.json',sample_audits);gc.collect();torch.cuda.empty_cache()
 summary={'samples':8,'layer_steps':len(measurements),'query_head_observations':len(measurements)*28,'gqa_observations':len(measurements)*4,'query_head':aggregate(measurements),'gqa_mean':aggregate(measurements,True),'per_layer':{str(l):{'query_head':aggregate([r for r in measurements if r['layer']==l]),'gqa_mean':aggregate([r for r in measurements if r['layer']==l],True)} for l in range(28)},'per_sample':{str(i):aggregate([r for r in measurements if r['sample']==i]) for i in range(8)},'per_generation_step':{str(s):aggregate([r for r in measurements if r['step']==s]) for s in sorted(set(r['step'] for r in measurements))},'wall_seconds':time.perf_counter()-start}
 write('summary.json',summary)
 write('audit.json',{'all8_original_generated_token_sequences_identical':True,'all8_first_sparse_step_28layers_head0_selection_hashes_identical':True,'old_logs_do_not_contain_later_or_other_heads_selection_hashes':True,'all_steps_28layers':True,'checkpoint_unchanged':sha(CKPT)=='18113b5cd6f34b8a9b9925694bbc616787ad77c71d40d4d0b3eecf2aedaff640','original_results_unchanged':sha(RUN/'learned_all28.jsonl')==json.loads((OUT/'config.json').read_text())['original_results_sha256'],'fp32_oracle_does_not_feed_generation':True,'group_oracle_optimal_for_mean_7_query_heads_not_individually_optimal_heads':True})
 with (ROOT/'docs/ranker_full_training_20261003.md').open('a') as f:
  f.write('\n\n### All28 HotpotQA前8条实际decode attention质量诊断\n\n')
  f.write('原epoch7、32K/128、sink32+local512+middle512，无COBS回退。独立重放前8条全部稀疏decode步，完整生成token序列及首步28层head0选块哈希与原结果一致。原日志未保存其他步/其他head选择，无法逐项追溯比较；新诊断保留所有所选middle块。\n\n')
  f.write('分母为稀疏模型自身轨迹真实query对该轨迹完整历史K与self的FP32 softmax总质量；不是full模型独立轨迹。fixed含sink/local/self，middle增量为实际选中middle的质量。GQA按每7个queryhead质量均值，相同middle块预算oracle按组均值选top块。总体按head-step等权，另存分层/样本/生成步结果；仅8条不代表200条。\n\n')
  f.write(json.dumps({k:summary[k] for k in ['samples','layer_steps','query_head','gqa_mean','wall_seconds']},ensure_ascii=False,indent=2))
  f.write('\n\n独立产物：'+str(OUT)+'，含diagnose.py、coverage_steps.jsonl、summary.json、audit.json、sample_audits.json。\n')
 write('status.json',{'stage':'completed','completed':8,'wall_seconds':time.perf_counter()-start});(OUT/'COMPLETE').write_text('All8 coverage diagnostic completed and audited')
if __name__=='__main__':
 try:main()
 except Exception:
  write('status.json',{'stage':'failed','traceback':traceback.format_exc()});raise
