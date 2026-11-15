"""Bounded real-generation HotpotQA comparison; preserves existing evaluators."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import sys,json,time,hashlib,copy,traceback,gc
from pathlib import Path
from collections import Counter
import torch
ROOT=Path("/root/adablock");sys.path.insert(0,str(ROOT))
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.utils.hub import cached_file
from scripts import eval_longbench_adablock_sparse as ev
from utils.longbench_eval import build_model_input_text,normalize_answer
from models.block_ranker import load_ranker
from models.experimental_all_layer_ranker_qwen import ExperimentalAllLayerRankerController as Controller
from functools import partial
import models.layer_head_sparse_qwen as original_controller
def forbidden_fallback(*args,**kwargs):
 raise RuntimeError("Baseline selector invoked during all-layer experiment")
METHOD=os.environ["RANKER_EVAL_METHOD"]
assert METHOD in ["learned","cobs","quest"]
if METHOD=="learned":
 original_controller.score_qk_block_digests=forbidden_fallback
 ev.LayerHeadQKSparseController=partial(Controller,all_layer_ranker=True)
else:
 Controller=original_controller.LayerHeadQKSparseController
 ev.LayerHeadQKSparseController=Controller
RUN=ROOT/"results/hotpotqa_curve50k_threeway_20261005"
SHARD=int(os.environ["RANKER_EVAL_SHARD"]);assert SHARD in range(4)
PROBE_ONLY=os.environ.get("RANKER_PROBE_ONLY")=="1"
OUT=RUN/METHOD/("probe" if PROBE_ONLY else "shard_"+str(SHARD))
OUT.mkdir(parents=True,exist_ok=True)
CKPT=ROOT/"checkpoints/ranker_all28_curve_20261005/best.pt"
PROMPT="Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
def write(name,data):
 p=OUT/name
 text=json.dumps(data,indent=2,ensure_ascii=False)
 if name=="config.json" and p.exists():
  assert json.loads(p.read_text())==data,"Resume configuration mismatch"
 tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(text);tmp.replace(p)

def score(pred,answers):
 vals=[]
 for ans in answers:
  p=normalize_answer(pred).split();g=normalize_answer(ans).split()
  common=sum((Counter(p)&Counter(g)).values())
  vals.append(2*common/(len(p)+len(g)) if common else 0.)
 return 100*max(vals)
def sync():torch.cuda.synchronize()
def main():
 torch.set_num_threads(4);torch.manual_seed(42)
 rows=json.loads((RUN/"selected_rows.json").read_text());assert len(rows)==200
 assigned=[i for i in range(200) if i%4==SHARD]
 write("selected_rows.json",rows)
 source={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ["scripts/eval_longbench_adablock_sparse.py","models/layer_head_sparse_qwen.py","models/block_ranker.py","utils/qk_block_oracle.py","models/experimental_all_layer_ranker_qwen.py"]}
 write("config.json",dict(task="hotpotqa",samples=200,selection="complete official LongBench HotpotQA test in source order",method=METHOD,physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],parallel_shards=True,shard=SHARD,assigned_indices=assigned,probe_only=PROBE_ONLY,reuse_previous_results=False,requested_context_limit=32768,model_total_positions=32768,max_input_length=32640,max_new_tokens=128,block_size=16,sink_tokens=32,local_window_blocks=32,middle_blocks=32,seed=42,dtype="bfloat16",prefill="full SDPA all methods",decode="greedy raw logits, same manual decoder API",learned_layers=list(range(28)),checkpoint_training_layers=list(range(28)),other_learned_layers="none: baseline calls raise",checkpoint=str(CKPT),checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),source_sha256=source,prompt=PROMPT,metric="Official LongBench normalized QA token F1, max over references, macro average *100",metric_source="https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py"))
 sys.argv=[sys.argv[0]];args=ev.parse_args()
 args.max_input_length=32640;args.max_new_tokens_override=128;args.fixed_middle_blocks=32
 args.ranker_checkpoint=str(CKPT);args.ranker_untrained_layers="cobs"
 args.debug_max_steps=128;args.debug_logits_topk=1;args.ranker_profile=False
 args._ranker_bundle=load_ranker(CKPT,"cpu")
 assert args._ranker_bundle[1]["training"]["step"]==50000
 assert list(args._ranker_bundle[1]["data_metadata"]["layers"])==list(range(28))
 assert hashlib.sha256(CKPT.read_bytes()).hexdigest()=="d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae"
 path=str(Path(cached_file("Qwen/Qwen2.5-7B-Instruct","config.json",local_files_only=True)).parent)
 assert Path(path).name=="a09a35458c702b33eeacc393d103063234e8bc28"
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 from utils.longbench_eval import truncate_middle
 inputs=[]
 for i,row in enumerate(rows):
  ids=tok(build_model_input_text(tok,PROMPT.format(**row),use_chat_template=True),return_tensors="pt")["input_ids"]
  used=truncate_middle(ids,args.max_input_length)
  assert used.shape[-1]+128<=32768
  inputs.append(dict(index=i,original_prompt_tokens=ids.shape[-1],used_tokens=used.shape[-1],input_truncated=used.shape[-1]<ids.shape[-1],input_ids_sha256=hashlib.sha256(used.numpy().tobytes()).hexdigest()))
 assert inputs==json.loads((RUN/"reference_input_manifest.json").read_text()),"Historical Full input mismatch"
 write("input_manifest.json",inputs)
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",local_files_only=True).eval()
 probe_args=copy.copy(args);probe_args.max_new_tokens_override=3
 longest=max(inputs,key=lambda r:r["used_tokens"])["index"]
 probe_prompt=build_model_input_text(tok,PROMPT.format(**rows[longest]),use_chat_template=True)
 torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
 ev.run_full_decode_trace(model,tok,probe_prompt,probe_args)
 sync()
 write("memory_preflight.json",dict(sample=longest,prompt_tokens=inputs[longest]["used_tokens"],seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,device="cuda:0",passed=True))
 gc.collect();torch.cuda.empty_cache()
 original_learned=Controller._learned_block_scores;original_select=Controller._select_token_indices
 audit={}
 def learned(self,layer,*a,**kw):
  audit["learned_calls"][str(layer)]=audit["learned_calls"].get(str(layer),0)+1
  return original_learned(self,layer,*a,**kw)
 def select(self,layer,*a,**kw):
  ids=original_select(self,layer,*a,**kw)
  audit["attention_calls"]+=1
  audit["layer_calls"][str(layer)]=audit["layer_calls"].get(str(layer),0)+1
  audit["max_selected_history_tokens"]=max(audit["max_selected_history_tokens"],ids.shape[1])
  if len(audit["first_selections"])<28:
   audit["first_selections"].append({"layer":layer,"history":self._history_len,"tokens":ids.shape[1],"head0_sha256":hashlib.sha256(ids[0].numpy().tobytes()).hexdigest()})
  return ids
 Controller._learned_block_scores=learned;Controller._select_token_indices=select
 def reset():
  audit.clear();audit.update(learned_calls={},layer_calls={},attention_calls=0,max_selected_history_tokens=0,first_selections=[])
 prompt=build_model_input_text(tok,PROMPT.format(**rows[0]),use_chat_template=True)
 # All-candidate sparse attention should reproduce full decoding.
 small=copy.copy(args);small.max_input_length=1024;small.max_new_tokens_override=4
 fullpred,fulltrace,fullids=ev.run_full_decode_trace(model,tok,prompt,small)
 small.selection_mode="learned_qk" if METHOD=="learned" else "fixed_qk";small.qk_score_selector="quest_aabb" if METHOD=="quest" else "cobs";small.fixed_middle_blocks=1024;reset()
 pred,stats,trace=ev.run_layer_head_qk_decode(model,tok,None,torch.device("cuda:0"),prompt,small)
 assert fullids==[r["token_id"] for r in trace],(fullids,trace)
 parity=dict(full_ids=fullids,sparse_ids=[r["token_id"] for r in trace],all_candidate_token_parity=True,audit=copy.deepcopy(audit))
 small.max_input_length=2048;small.selection_mode="learned_qk" if METHOD=="learned" else "fixed_qk";small.qk_score_selector="quest_aabb" if METHOD=="quest" else "cobs";small.fixed_middle_blocks=32;reset()
 pred,stats,trace=ev.run_layer_head_qk_decode(model,tok,None,torch.device("cuda:0"),prompt,small)
 assert set(audit["layer_calls"])=={str(i) for i in range(28)} and audit["attention_calls"]>0
 if METHOD=="learned":assert set(audit["learned_calls"])=={str(i) for i in range(28)}
 assert audit["max_selected_history_tokens"]<2048
 parity["learned_active_probe"]=copy.deepcopy(audit);write("path_verification.json",parity)
 write("status.json",{"stage":"running","path_verification":"passed"})
 if PROBE_ONLY:
  write("status.json",{"stage":"probe_completed","all28_called":True,"baseline_calls":0});return
 results={}
 for method in [METHOD]:
  records=[]
  journal=OUT/(method+".jsonl")
  if journal.exists():
   raw=journal.read_bytes()
   if raw and not raw.endswith(b"\n"):
    valid=raw[:raw.rfind(b"\n")+1]
    backup=journal.with_name(journal.name+".incomplete_"+str(time.time_ns()))
    backup.write_bytes(raw)
    temp=journal.with_suffix(".repair.tmp");temp.write_bytes(valid);temp.replace(journal)
   records=[json.loads(line) for line in journal.read_text().splitlines()]
   assert len(records)<=len(assigned)
   for j,r in enumerate(records):
    assert r["index"]==assigned[j] and r["input_ids_sha256"]==inputs[assigned[j]]["input_ids_sha256"],"Resume journal mismatch"
  completed_indices={r["index"] for r in records}
  for i,row in enumerate(rows):
   if i not in assigned or i in completed_indices:continue
   prompt=build_model_input_text(tok,PROMPT.format(**row),use_chat_template=True)
   reset();gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats()
   start=time.perf_counter()
   if method=="full":
    pred,trace,ids=ev.run_full_decode_trace(model,tok,prompt,args)
    encoded=tok(prompt,return_tensors="pt")["input_ids"]
    stats={"generated_tokens":len(ids),"prompt_tokens":encoded.shape[-1],"used_prompt_tokens":min(32640,encoded.shape[-1])}
   else:
    args.selection_mode="learned_qk" if method=="learned" else "fixed_qk"
    args.qk_score_selector="quest_aabb" if method=="quest" else "cobs"
    pred,stats,trace=ev.run_layer_head_qk_decode(model,tok,None,torch.device("cuda:0"),prompt,args)
   sync();seconds=time.perf_counter()-start
   if method!="full":
    assert set(audit["layer_calls"])=={str(i) for i in range(28)}
    assert stats["avg_requested_middle_blocks"]==32
    assert audit["max_selected_history_tokens"]<=1056
   if method=="learned":assert set(audit["learned_calls"])=={str(i) for i in range(28)}
   record=dict(index=i,prediction=pred,answers=row["answers"],score=score(pred,row["answers"]),seconds=seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,stats=stats,audit=copy.deepcopy(audit),trace=trace)
   record.update(sample_id=row.get("_id",row.get("id")),exact_match=100*float(any(normalize_answer(pred)==normalize_answer(a) for a in row["answers"])),original_prompt_tokens=inputs[i]["original_prompt_tokens"],retained_prompt_tokens=inputs[i]["used_tokens"],input_truncated=inputs[i]["input_truncated"],input_ids_sha256=inputs[i]["input_ids_sha256"],output_hit_limit=stats["generated_tokens"]>=128,output_truncated=stats["generated_tokens"]>=128 and trace[-1]["token_id"]!=tok.eos_token_id,ended_with_eos=trace[-1]["token_id"]==tok.eos_token_id)
   records.append(record)
   with (OUT/(method+".jsonl")).open("a") as f:
    f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
   write("status.json",{"stage":"running","method":method,"completed_samples":len(records),"seconds":seconds})
   print(json.dumps({"method":method,"sample":i,"score":record["score"],"seconds":seconds}),flush=True)
  results[method]={"score":sum(r["score"] for r in records)/len(assigned),"seconds":sum(r["seconds"] for r in records),"peak_allocated_mib":max(r["peak_allocated_mib"] for r in records),"peak_reserved_mib":max(r["peak_reserved_mib"] for r in records),"generated_tokens":sum(r["stats"]["generated_tokens"] for r in records)}
  write("summary.json",results)
 write("status.json",{"stage":"completed","results":results});(OUT/"COMPLETE").write_text("done")
if __name__=="__main__":
 try:main()
 except Exception:
  write("status.json",{"stage":"failed","traceback":traceback.format_exc()});raise
