"""Prepared only: frozen-input HF Full calibration. Defaults to print plan; --execute required."""
import os
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
import argparse,json,sys,time,hashlib,ast,re,string
from pathlib import Path
from collections import Counter
ROOT=Path("/root/adablock");sys.path.insert(0,str(ROOT))
W=Path(__file__).parent
FULL=ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full"
SNAP=Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
p=argparse.ArgumentParser()
p.add_argument("--variant",choices=["A"],required=True)
p.add_argument("--limit",type=int,default=200)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--execute",action="store_true")
args=p.parse_args()
assert 1<=args.limit<=200
config=json.loads((W/("proposed_hf_"+args.variant+"_generation_config.json")).read_text())
config["min_new_tokens"]=None # Avoid inert MinLength processor; min_length remains0.
config["disable_compile"]=True
config["top_k"]=50 # inactive when do_sample=False; avoid HF unused-top_k warning.
plan=dict(variant=args.variant,model=str(SNAP),generation_config=config,inputs="exact historical Full token hashes",limit=args.limit,output=str(args.out),dtype="bfloat16",attention_backend="sdpa",no_sparse_controller=True,torch_seed=42,stop_on_A_token_mismatch=True)
if not args.execute:
 print(json.dumps(dict(status="prepared_not_executed",plan=plan),indent=2));sys.exit(0)
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from utils.longbench_eval import build_model_input_text
assert torch.cuda.is_available(),"No CUDA device available; do not download or start substitute resources."
torch.set_num_threads(4);torch.manual_seed(42)
rows=json.loads((FULL/"selected_rows.json").read_text());old=[json.loads(l) for l in (FULL/"full.jsonl").read_text().splitlines()]
manifest=json.loads((FULL/"input_manifest.json").read_text())
prompt=json.loads((W/"upstream_longbench/config/dataset2prompt.json").read_text())["hotpotqa"]
tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True)
inputs=[]
for i,row in enumerate(rows):
 ids=tok(build_model_input_text(tok,prompt.format(**row),True),return_tensors="pt")["input_ids"]
 assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==manifest[i]["input_ids_sha256"]
 assert ids.shape[-1]+32<=32768
 inputs.append(ids)
sc=ast.parse((W/"upstream_longbench/metrics.py").read_text())
scope=dict(re=re,string=string,Counter=Counter)
exec(compile(ast.Module(body=[n for n in sc.body if isinstance(n,ast.FunctionDef) and n.name in ["normalize_answer","f1_score","qa_f1_score"]],type_ignores=[]),"official_pure_scores","exec"),scope)
args.out.mkdir(parents=True,exist_ok=True)
identity={k:v for k,v in plan.items() if k!="limit"}
cf=args.out/"config.json"
if cf.exists():assert json.loads(cf.read_text())==identity,"Output identity mismatch"
else:cf.write_text(json.dumps(identity,indent=2))
journal=args.out/"predictions.jsonl";done=[]
if journal.exists():
 raw=journal.read_bytes()
 assert not raw or raw.endswith(b"\n"),"Partial journal requires explicit repair preserving original."
 done=[json.loads(l) for l in raw.decode().splitlines()]
 assert [r["index"] for r in done]==list(range(len(done)))
 assert all(r["input_ids_sha256"]==manifest[r["index"]]["input_ids_sha256"] for r in done)
 assert all(r.get("archive_token_parity",True) for r in done) if args.variant=="A" else True
if len(done)>=args.limit:
 print("Requested prefix already complete");sys.exit(0)
model=AutoModelForCausalLM.from_pretrained(str(SNAP),local_files_only=True,torch_dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
assert len(model.model.layers)==28
assert model.config._attn_implementation=="sdpa"
g=GenerationConfig.from_dict(config)
original_processors=model._get_logits_processor
original_stopping=model._get_stopping_criteria
audit_calls=[]
def checked_processors(*aa,**kk):
 result=original_processors(*aa,**kk)
 names=[type(x).__name__ for x in result]
 assert names==[],("Unexpected logits processors",names)
 audit_calls.append(dict(kind="logits_processors",names=names))
 return result
def checked_stopping(*aa,**kk):
 result=original_stopping(*aa,**kk)
 names=[type(x).__name__ for x in result]
 assert set(names)=={"MaxLengthCriteria","EosTokenCriteria"},names
 eos=[x for x in result if type(x).__name__=="EosTokenCriteria"][0].eos_token_id
 assert eos.reshape(-1).tolist()==[151645],eos
 audit_calls.append(dict(kind="stopping_criteria",names=names,eos=eos.reshape(-1).tolist()))
 return result
model._get_logits_processor=checked_processors
model._get_stopping_criteria=checked_stopping
torch.cuda.reset_peak_memory_stats()
(args.out/"runtime_model_audit.json").write_text(json.dumps(dict(model_revision=SNAP.name,layers=len(model.model.layers),dtype=str(next(model.parameters()).dtype),backend=model.config._attn_implementation,generation_config=g.to_dict()),indent=2))
for i in range(len(done),args.limit):
 audit_calls.clear()
 ids=inputs[i].to(model.device);mask=torch.ones_like(ids)
 torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  out=model.generate(input_ids=ids,attention_mask=mask,generation_config=g,return_dict_in_generate=True,output_scores=False)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 tids=out.sequences[0,ids.shape[-1]:].tolist()
 pred=tok.decode(tids,skip_special_tokens=True).strip()
 f1=100*max(scope["qa_f1_score"](pred,a) for a in rows[i]["answers"])
 em=100*any(scope["normalize_answer"](pred)==scope["normalize_answer"](a) for a in rows[i]["answers"])
 reference=[t["token_id"] for t in old[i]["trace"]][:32]
 record=dict(index=i,id=rows[i]["_id"],prediction=pred,answers=rows[i]["answers"],token_ids=tids,input_ids_sha256=manifest[i]["input_ids_sha256"],F1=f1,EM=em,seconds=elapsed,archive_token_parity=tids==reference if args.variant=="A" else None,eos=tids[-1]==151645,contains_endoftext=151643 in tids,output_truncated=len(tids)==32 and tids[-1]!=151645)
 record["generation_path_audit"]=list(audit_calls)
 with journal.open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
 done.append(record)
 if args.variant=="A" and tids!=reference:
  (args.out/"PARITY_MISMATCH.json").write_text(json.dumps(dict(index=i,hf=tids,archived=reference,note="Stop and diagnose; do not tune to paper score."),indent=2))
  raise RuntimeError("HF/manual prefix mismatch; stopped before further evaluation")
 if i==7:
  assert all(x["archive_token_parity"] for x in done[:8])
  (args.out/"pilot8_passed.json").write_text(json.dumps(dict(indices=list(range(8)),all_tokens_equal=True,continue_same_process_to=200),indent=2))
 print(json.dumps(dict(index=i,F1=f1,seconds=elapsed,token_parity=record["archive_token_parity"])),flush=True)
 (args.out/"status.json").write_text(json.dumps(dict(completed=len(done),requested=args.limit)))
summary=dict(all_token_parity=all(x["archive_token_parity"] for x in done),input_truncated=0,output_truncated=sum(x["output_truncated"] for x in done),eos=sum(x["eos"] for x in done),contains_endoftext=sum(x["contains_endoftext"] for x in done),sample_seconds=sum(x["seconds"] for x in done),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,samples=len(done),F1=sum(x["F1"] for x in done)/len(done),EM=sum(x["EM"] for x in done)/len(done),generation_config=g.to_dict())
(args.out/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
