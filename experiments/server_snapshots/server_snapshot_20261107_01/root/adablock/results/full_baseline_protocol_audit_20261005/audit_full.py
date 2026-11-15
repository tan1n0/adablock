"""CPU-only Full baseline protocol audit. No model load or inference."""
import os
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
import ast,json,sys,hashlib,zipfile,inspect,platform
from pathlib import Path
from collections import Counter
import re,string
import torch,transformers,numpy as np
from transformers import AutoTokenizer,GenerationConfig,AutoConfig
from transformers.models.qwen2 import modeling_qwen2
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
ROOT=Path("/root/adablock");sys.path.insert(0,str(ROOT))
from utils.longbench_eval import build_model_input_text,truncate_middle,TASK_PROMPTS
W=Path(__file__).parent
FULL=ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full"
SNAP=Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(W/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
rows=json.loads((FULL/"selected_rows.json").read_text())
records=[json.loads(l) for l in (FULL/"full.jsonl").read_text().splitlines()]
cfg=json.loads((FULL/"config.json").read_text())
manifest=json.loads((FULL/"input_manifest.json").read_text())
tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True)
gc=GenerationConfig.from_pretrained(str(SNAP),local_files_only=True)
mc=AutoConfig.from_pretrained(str(SNAP),local_files_only=True)
official=W/"upstream_longbench"
prompt=json.loads((official/"config/dataset2prompt.json").read_text())["hotpotqa"]
cap=json.loads((official/"config/dataset2maxlen.json").read_text())["hotpotqa"]
assert cfg["prompt"]==prompt and cap==32
# Execute only the three pure scoring functions from the reviewed fixed upstream source.
src=(official/"metrics.py").read_text();tree=ast.parse(src)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ["normalize_answer","f1_score","qa_f1_score"]]
scope=dict(re=re,string=string,Counter=Counter)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(official/"metrics.py"),"exec"),scope)
official_f1=scope["qa_f1_score"]
checks=[];caprows=[];template_cases={}
for i,(row,r) in enumerate(zip(rows,records)):
 assert r["index"]==i and r["answers"]==row["answers"]
 text=build_model_input_text(tok,prompt.format(**row),True)
 ids=tok(text,return_tensors="pt")["input_ids"]
 used=truncate_middle(ids,32640)
 h=hashlib.sha256(used.numpy().tobytes()).hexdigest()
 assert h==manifest[i]["input_ids_sha256"]==r["input_ids_sha256"]
 assert used.shape[-1]==ids.shape[-1]
 tokenids=[x["token_id"] for x in r["trace"]]
 assert len(tokenids)==r["stats"]["generated_tokens"]
 pred=tok.decode(tokenids,skip_special_tokens=True).strip()
 assert pred==r["prediction"]
 f=100*max(official_f1(pred,a) for a in row["answers"])
 assert abs(f-r["score"])<1e-10
 short=tok.decode(tokenids[:32],skip_special_tokens=True).strip()
 f32=100*max(official_f1(short,a) for a in row["answers"])
 caprows.append(dict(index=i,F1_128=f,F1_32=f32,pred128=pred,pred32=short,length=len(tokenids)))
 checks.append(dict(index=i,id=row["_id"],input_tokens=ids.shape[-1],input_sha256=h,gen_tokens=len(tokenids),first_eos=tokenids[0]==151645,has_endoftext=151643 in tokenids,eos_positions=[j for j,t in enumerate(tokenids) if t==151645],ends_eos=tokenids[-1]==151645,selector_calls=r["audit"]["attention_calls"],learned_calls=r["audit"]["learned_calls"],score=f))
 if i==0:template_cases=dict(prefix=text[:180],suffix=text[-180:],special_token_map=tok.special_tokens_map,tokenizer_class=tok.__class__.__name__,clean_up_tokenization_spaces=tok.clean_up_tokenization_spaces)
assert len(rows)==len(records)==200 and len({r["_id"] for r in rows})==200
datafile=ROOT/"data/longbench/hotpotqa.jsonl"
zipfile_path=ROOT/".cache/longbench/datasets--zai-org--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip"
with zipfile.ZipFile(zipfile_path) as z:
 names=[n for n in z.namelist() if n.endswith("/hotpotqa.jsonl") or n=="hotpotqa.jsonl"];assert len(names)==1;b=z.read(names[0])
assert b==datafile.read_bytes()
assert [json.loads(l) for l in b.decode().splitlines()]==rows
source_status={f:sha(ROOT/f)==h for f,h in cfg["source_sha256"].items()};assert all(source_status.values())
dump("per_sample_audit.json",checks)
dump("cap32_recheck.json",dict(F1_128=np.mean([r["F1_128"] for r in caprows]),F1_32=np.mean([r["F1_32"] for r in caprows]),text_changed=[r["index"] for r in caprows if r["pred128"]!=r["pred32"]],score_changed=[r for r in caprows if abs(r["F1_128"]-r["F1_32"])>1e-10],scope="Exact saved raw-greedy trajectory prefix, not HF-generate equivalence."))
dump("template_audit.json",template_cases)
dump("cached_model_config.json",mc.to_dict());dump("cached_generation_config.json",gc.to_dict())
# Inspect actual code, never instantiate model weights.
snapshot=W/"implementation_snapshot";snapshot.mkdir(exist_ok=True)
files=[ROOT/"scripts/run_ranker_hotpotqa_full_20261004.py",ROOT/"scripts/eval_longbench_adablock_sparse.py",ROOT/"models/layer_head_sparse_qwen.py",ROOT/"utils/longbench_eval.py",Path(modeling_qwen2.__file__),Path(inspect.getfile(sdpa_attention_forward)),Path(inspect.getfile(GenerationMixin)),Path(inspect.getfile(RepetitionPenaltyLogitsProcessor))]
sources={}
for j,p in enumerate(files):
 out=snapshot/(str(j)+"_"+p.name);out.write_bytes(p.read_bytes());sources[str(p)]=dict(sha256=sha(p),snapshot=str(out.relative_to(W)))
weights=[dict(name=p.name,bytes=p.stat().st_size,resolved_cache_blob=p.resolve().name) for p in SNAP.glob("*.safetensors")]
# Config for two proposed same-input HF runs: penalty is the ONLY A/B difference.
ga=GenerationConfig(max_new_tokens=32,do_sample=False,num_beams=1,repetition_penalty=1.0,eos_token_id=151645,pad_token_id=151643,bos_token_id=151643,use_cache=True,temperature=1.0,top_p=1.0,top_k=0,min_length=0,min_new_tokens=0,no_repeat_ngram_size=0,encoder_repetition_penalty=1.0,length_penalty=1.0,renormalize_logits=False,remove_invalid_values=False)
gb=GenerationConfig.from_dict(ga.to_dict());gb.repetition_penalty=1.05
dump("proposed_hf_A_generation_config.json",ga.to_dict());dump("proposed_hf_B_generation_config.json",gb.to_dict())
audit=dict(status="offline_audit_complete_no_inference",model_revision=SNAP.name,model_revision_evidence="Current resolved snapshot and historical cap32 tokenizer audit; original Full config did not record revision/weight hashes, so no retroactive full-byte proof.",cached_files={p.name:sha(p) for p in SNAP.iterdir() if p.is_file() and p.suffix==".json"},weights_inventory=weights,weights_full_bytes_rehashed=False,environment=dict(python=platform.python_version(),transformers=transformers.__version__,torch=torch.__version__),dataset=dict(path=str(datafile),sha256=sha(datafile),zip_snapshot="5e628be450b7e67fb7ae6e201bd6d8f7056f7672",zip_member=names[0],exact_bytes_equal=True,unique_ids=200,exact_selected_rows=True,paper_ids_unknown=True),input=dict(official_prompt_exact_match=True,current_generic_prompt_is_different=TASK_PROMPTS["hotpotqa"]!=prompt,all200_archived_input_hashes_match=True,min_tokens=min(x["input_tokens"] for x in checks),max_tokens=max(x["input_tokens"] for x in checks),mean_tokens=np.mean([x["input_tokens"] for x in checks]),truncated=0,template=template_cases),decoding=dict(manual="raw argmax,temperature0,top_p1; no logits processors",native_cached_generation_config=json.loads((SNAP/"generation_config.json").read_text()),official_max_new_tokens=32,historical_max_new_tokens=128,first_token_eos_check_missing=True,first_token_eos_cases=[x["index"] for x in checks if x["first_eos"]],endoftext_cases=[x["index"] for x in checks if x["has_endoftext"]],all200_eos_only_last=all(x["eos_positions"]==[x["gen_tokens"]-1] for x in checks),all200_prediction_trace_match=True,topk_logits_saved=1),dense_path=dict(all200_selector_calls_zero=all(x["selector_calls"]==0 and not x["learned_calls"] for x in checks),full_cache_no_pruning=True,causal_mask_positions_static_review_pass=True,backend="HF SDPA; exact dispatched CUDA kernel not logged",dtype="bfloat16",rope_theta=mc.rope_theta,rope_scaling=getattr(mc,"rope_scaling",None),use_sliding_window=mc.use_sliding_window,limitation="Static implementation audit and archived traces; no numerical native generate equivalence run."),scoring=dict(upstream_pure_functions_executed=True,per_question_scores_match=True,F1=np.mean([x["score"] for x in checks]),official_rounded=round(float(np.mean([x["score"] for x in checks])),2),no_hotpot_first_line_postprocess=True),historical_source_hashes_match=source_status,sources=sources)
dump("AUDIT.json",audit)
print(json.dumps({k:v for k,v in audit.items() if k in ["dataset","input","decoding","dense_path","scoring"]},ensure_ascii=False,indent=2))
print("CAP32",(W/"cap32_recheck.json").read_text())
