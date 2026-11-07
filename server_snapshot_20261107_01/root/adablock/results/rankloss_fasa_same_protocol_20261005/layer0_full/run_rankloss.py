"""Frozen RankLoss, native HF4.43 Full prefill and cache, official FASA generation protocol."""
import os
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["TOKENIZERS_PARALLELISM"]="false"
from pathlib import Path
import sys,json,hashlib,time,importlib.util,torch,transformers,flash_attn
W=Path(__file__).parent;ROOT=Path("/root/adablock")
PARENT=W.parent;LIMIT=int(os.environ.get("RANKLOSS_LIMIT","200"));assert LIMIT in (8,200)
sys.path.insert(0,str(PARENT))
import rankloss_adapter as adapter
assert json.loads((PARENT/"adapter_tests.json").read_text())["passed"]
assert json.loads((PARENT/"adapter_tests.json").read_text())["adapter_sha256"]==hashlib.sha256((PARENT/"rankloss_adapter.py").read_bytes()).hexdigest()
for rel,expected in json.loads((PARENT/"source_hashes.json").read_text()).items():assert hashlib.sha256((PARENT/"snapshot"/rel).read_bytes()).hexdigest()==expected
fullref=[json.loads(l) for l in (ROOT/"results/fasa_full_reproduction_20261005_multigpu/run/predictions.jsonl").read_text().splitlines()]
sys.path.insert(0,str(W/"patched_source"))
import longbench.longbench_pred as official
SNAP=Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
MODEL="Qwen2.5-7B-Instruct";OUT=W/"run";OUT.mkdir(exist_ok=True)
def write(n,x):
 p=OUT/n;t=p.with_suffix(".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert torch.cuda.is_available(),"GPU required; do not alter allocation"
assert transformers.__version__=="4.43.0" and flash_attn.__version__=="2.7.4.post1"
rows=official.read_jsonl(str(ROOT/"data/longbench/hotpotqa.jsonl"));assert len(rows)==200 and len({x["_id"] for x in rows})==200
oldroot=ROOT/"results/full_hf_B_calibration_20261005/output"
reference=[json.loads(l) for l in (oldroot/"predictions.jsonl").read_text().splitlines()]
originaljson=ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full/selected_rows.json"
assert rows==json.loads(originaljson.read_text())
official.seed_everything(42);torch.set_num_threads(4)
tokenizer=official.AutoTokenizer.from_pretrained(str(SNAP),trust_remote_code=False,local_files_only=True)
tokenizer.pad_token_id=tokenizer.eos_token_id
inputs=[]
for i,row in enumerate(rows):
 prompt=official.dataset2prompt["hotpotqa"].format(**row)
 raw=tokenizer(prompt,truncation=False,return_tensors="pt").input_ids[0]
 assert len(raw)<=official.model2maxlen[MODEL],"Unexpected truncation requires audit"
 text=official.build_chat(tokenizer,prompt,MODEL)
 ids=tokenizer(text,truncation=False,return_tensors="pt").input_ids
 h=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
 assert h==reference[i]["input_ids_sha256"],("Input hash differs from A/B",i)
 inputs.append(dict(index=i,id=row["_id"],raw_tokens=len(raw),input_tokens=ids.shape[-1],input_ids_sha256=h))
write("input_manifest.json",inputs)
identity=dict(mode="learned_rankloss",variant=W.name,block_size=16,sink_blocks=2,local_blocks=32,middle_blocks=32,checkpoint_sha256=adapter.CKPT_SHA,adapter_sha256=sha(PARENT/"rankloss_adapter.py"),source_hashes=json.loads((PARENT/"source_hashes.json").read_text()),official_commit="96c00c46db12ccee56104175470aa29579d66c58",model_revision=SNAP.name,model_path=str(SNAP),dataset_sha256=sha(ROOT/"data/longbench/hotpotqa.jsonl"),samples=200,backend="flash_attention_2",dtype="bfloat16",max_input=official.model2maxlen[MODEL],max_new_tokens=32,seed=42,threads=4,torch=torch.__version__,transformers=transformers.__version__,flash_attn=flash_attn.__version__,script_sha256=sha(__file__),patch_sha256=sha(W/"official_full_only.patch"))
identity["placement"]=json.loads((W/"placement_plan.json").read_text())
p=OUT/"config.json"
if p.exists():assert json.loads(p.read_text())==identity
else:write("config.json",identity)
journal=OUT/"predictions.jsonl";done=[]
if journal.exists():
 raw=journal.read_bytes();assert not raw or raw.endswith(b"\n"),"Partial journal: preserve and audit before repair"
 done=[json.loads(l) for l in raw.decode().splitlines()]
 assert [x["index"] for x in done]==list(range(len(done)))
 assert all(x["id"]==rows[i]["_id"] and x["input_ids_sha256"]==inputs[i]["input_ids_sha256"] for i,x in enumerate(done))
# Official per-call result file is separately retained. Ignore orphan records after a crash; canonical sidecar gates resume.
if len(done)>=LIMIT:
 print("Already completed requested prefix",LIMIT,flush=True);sys.exit(0)
observer=adapter.Observer(W.name);observer.prepare()
model,tokenizer=official.load_model_and_tokenizer(str(SNAP),MODEL)
observer.attach(model)
assert model.config._attn_implementation=="flash_attention_2"
assert len(model.model.layers)==28
native=[dict(layer=i,cls=type(x.self_attn).__name__,module=type(x.self_attn).__module__,forward_module=getattr(x.self_attn,"_old_forward",x.self_attn.forward).__module__,placement_hook=hasattr(x.self_attn,"_hf_hook")) for i,x in enumerate(model.model.layers)]
assert all(x["cls"]=="Qwen2FlashAttention2" and x["forward_module"]=="rankloss_adapter" for x in native),native
assert model.hf_device_map==json.loads((W/"placement_plan.json").read_text())["device_map"]
assert model.generation_config.to_dict()==json.loads((ROOT/"results/fasa_full_reproduction_20261005_multigpu/run/native_full_audit.json").read_text())["model_generation_config"]
write("rankloss_runtime_audit.json",dict(device_map=model.hf_device_map,layers=native,actual_dtype=str(next(model.parameters()).dtype),tokenizer_pad=tokenizer.pad_token_id,model_generation_config=model.generation_config.to_dict(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
orig_generate=model.generate;orig_processors=model._get_logits_processor;orig_stopping=model._get_stopping_criteria
capture={}
def simple(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
 return x
class FiniteProcessorList(transformers.LogitsProcessorList):
 def __call__(self,input_ids,scores,**kwargs):
  assert torch.isfinite(scores).all().item(),"Nonfinite raw logits"
  return super().__call__(input_ids,scores,**kwargs)
def processors(*a,**kw):
 ps=orig_processors(*a,**kw);names=[type(x).__name__ for x in ps]
 assert names==["RepetitionPenaltyLogitsProcessor","MinLengthLogitsProcessor"],names
 assert ps[0].penalty==1.05
 capture["processors"]=[dict(name=type(x).__name__,**{k:simple(getattr(x,k)) for k in ["penalty","min_length","eos_token_id"] if hasattr(x,k)}) for x in ps]
 return FiniteProcessorList(ps)
def stopping(*a,**kw):
 ss=orig_stopping(*a,**kw)
 capture["stopping"]=[dict(name=type(x).__name__,**{k:simple(getattr(x,k)) for k in ["max_length","eos_token_id"] if hasattr(x,k)}) for x in ss]
 return ss
def generate(*a,**kw):
 assert not a and kw["do_sample"]==False and kw["num_beams"]==1 and kw["max_new_tokens"]==32
 ids=kw["input_ids"];n=ids.shape[-1]
 assert kw["min_length"]==n+1
 capture["generate_args"]={k:v for k,v in kw.items() if k not in ["input_ids","attention_mask"]}
 capture["input_sha256"]=hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()
 out=orig_generate(*a,**kw)
 capture["token_ids"]=out[0,n:].detach().cpu().tolist()
 return out
model._get_logits_processor=processors;model._get_stopping_criteria=stopping;model.generate=generate
for device in range(4):torch.cuda.reset_peak_memory_stats(device)
start_all=time.perf_counter()
for i in range(len(done),LIMIT):
 row=rows[i];capture.clear();observer.begin(i,inputs[i]["input_tokens"]);[torch.cuda.synchronize(d) for d in range(4)];start=time.perf_counter()
 # Call the official function unchanged, one row at a time for crash-safe checkpointing.
 result=official.get_pred_single_gpu([row],official.model2maxlen[MODEL],official.dataset2maxlen["hotpotqa"],official.dataset2prompt["hotpotqa"],"hotpotqa",MODEL,str(OUT/"official_predictions.jsonl"),tokenizer,model)
 [torch.cuda.synchronize(d) for d in range(4)];elapsed=time.perf_counter()-start
 assert len(result)==1 and capture["input_sha256"]==inputs[i]["input_ids_sha256"]
 tids=capture["token_ids"];assert 1<=len(tids)<=32
 assert tids[0]==fullref[i]["token_ids"][0],("Full prefill first token mismatch",i)
 for x in capture["processors"]:
  if x["name"]=="MinLengthLogitsProcessor":assert x["min_length"]==inputs[i]["input_tokens"]+1 and set(x["eos_token_id"])=={151645,151643}
 score=100*max(official.scorer.__globals__["dataset2metric"]["hotpotqa"](result[0]["pred"],a) for a in row["answers"])
 from longbench.metrics import normalize_answer
 em=100*any(normalize_answer(result[0]["pred"])==normalize_answer(a) for a in row["answers"])
 record=dict(index=i,id=row["_id"],prediction=result[0]["pred"],answers=row["answers"],F1=score,EM=em,seconds=elapsed,input_ids_sha256=inputs[i]["input_ids_sha256"],input_tokens=inputs[i]["input_tokens"],token_ids=tids,ended_eos=tids[-1] in [151645,151643],output_truncated=len(tids)==32 and tids[-1] not in [151645,151643],audit=dict(capture),budget_audit=observer.finish(len(tids)))
 with journal.open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
 done.append(record)
 if i==7:
  write("smoke8_passed.json",dict(indices=list(range(8)),native_full_prefill=True,first_tokens_match_full=True,finite_logits=True,variant=W.name,actual_layer_calls_passed=True,no_cobs_fallback=True,input_hashes_match=True,expected_processors=True,peak_reserved_mib={str(d):torch.cuda.max_memory_reserved(d)/2**20 for d in range(4)},continue_same_group_to=200))
 write("status.json",dict(stage="running",completed=len(done),sample_seconds=elapsed))
 print(json.dumps(dict(index=i,F1=score,seconds=elapsed)),flush=True)
if LIMIT==8:
 write("status.json",dict(stage="smoke8_completed",completed=len(done)));print("SMOKE8_COMPLETE",flush=True);sys.exit(0)
assert len(done)==200
summary=dict(samples=200,F1=sum(x["F1"] for x in done)/200,official_rounded_F1=official.scorer("hotpotqa",[x["prediction"] for x in done],[x["answers"] for x in done],rows[0]["all_classes"]),EM=sum(x["EM"] for x in done)/200,input_truncated=0,output_truncated=sum(x["output_truncated"] for x in done),ended_eos=sum(x["ended_eos"] for x in done),sample_seconds=sum(x["seconds"] for x in done),loop_wall_seconds=time.perf_counter()-start_all,peak_reserved_mib={str(d):torch.cuda.max_memory_reserved(d)/2**20 for d in range(4)})
write("summary.json",summary);write("status.json",dict(stage="completed",summary=summary));(OUT/"COMPLETE").write_text("RankLoss same-protocol200 complete.")
print(json.dumps(summary),flush=True)
