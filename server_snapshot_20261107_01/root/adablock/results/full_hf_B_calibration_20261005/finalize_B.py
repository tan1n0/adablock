import json,sys,ast,re,string,hashlib
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
W=Path(__file__).parent
A=Path("/root/adablock/results/full_hf_A_calibration_20261005_v2/output")
B=W/"output"
def dump(n,x):(W/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
aa=[json.loads(l) for l in (A/"predictions.jsonl").read_text().splitlines()]
bb=[json.loads(l) for l in (B/"predictions.jsonl").read_text().splitlines()]
assert [x["index"] for x in aa]==[x["index"] for x in bb]==list(range(200))
ca=json.loads((A/"config.json").read_text())["generation_config"]
cb=json.loads((B/"config.json").read_text())["generation_config"]
assert {k:(ca.get(k),cb.get(k)) for k in set(ca)|set(cb) if ca.get(k)!=cb.get(k)}=={"repetition_penalty":(1.,1.05)}
src=ast.parse((W/"upstream_longbench/metrics.py").read_text());ns=dict(re=re,string=string,Counter=Counter)
exec(compile(ast.Module(body=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in ["normalize_answer","f1_score","qa_f1_score"]],type_ignores=[]),"official","exec"),ns)
tok=AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",local_files_only=True)
rows=json.loads(Path("/root/adablock/results/longbench_ranker_hotpotqa_full_20261004/full/selected_rows.json").read_text())
diffs=[];dF=[];dE=[];dual=[];scores=[];ems=[]
for i,(a,b) in enumerate(zip(aa,bb)):
 assert a["id"]==b["id"]==rows[i]["_id"]
 assert a["input_ids_sha256"]==b["input_ids_sha256"] and a["answers"]==b["answers"]
 assert b["generation_path_audit"][0]==dict(kind="logits_processors",names=["RepetitionPenaltyLogitsProcessor"],penalty=1.05)
 pred=tok.decode(b["token_ids"],skip_special_tokens=True).strip();assert pred==b["prediction"]
 f=100*max(ns["qa_f1_score"](pred,x) for x in b["answers"])
 e=100*any(ns["normalize_answer"](pred)==ns["normalize_answer"](x) for x in b["answers"])
 assert abs(f-b["F1"])<1e-10 and e==b["EM"]
 scores.append(f);ems.append(e);dF.append(f-a["F1"]);dE.append(e-a["EM"])
 x,y=a["token_ids"],b["token_ids"]
 first=next((j for j in range(max(len(x),len(y))) if (x[j] if j<len(x) else None)!=(y[j] if j<len(y) else None)),None)
 if first is not None:
  diffs.append(dict(index=i,id=b["id"],question=rows[i]["input"],answers=b["answers"],first_divergence_step=first,A_token=x[first] if first<len(x) else None,B_token=y[first] if first<len(y) else None,A_token_text=tok.decode([x[first]]) if first<len(x) else None,B_token_text=tok.decode([y[first]]) if first<len(y) else None,A_prediction=a["prediction"],B_prediction=pred,A_F1=a["F1"],B_F1=f,A_EM=a["EM"],B_EM=e,F1_difference=f-a["F1"],A_ids=x,B_ids=y))
 stop=next((j+1 for j,t in enumerate(y) if t in [151645,151643]),len(y))
 dp=tok.decode(y[:stop],skip_special_tokens=True).strip()
 dual.append(dict(index=i,changed=stop<len(y),prediction=dp,F1=100*max(ns["qa_f1_score"](dp,z) for z in b["answers"])))
def counts(ds):return dict(wins=sum(v>1e-10 for v in ds),ties=sum(abs(v)<=1e-10 for v in ds),losses=sum(v<-1e-10 for v in ds))
summary=dict(samples=200,A_F1=sum(x["F1"] for x in aa)/200,B_F1=sum(scores)/200,A_EM=sum(x["EM"] for x in aa)/200,B_EM=sum(ems)/200,B_minus_A_F1=sum(dF)/200,B_minus_A_EM=sum(dE)/200,F1_paired=counts(dF),EM_paired=counts(dE),token_changed=len(diffs),prediction_changed=sum(a["prediction"]!=b["prediction"] for a,b in zip(aa,bb)),F1_changed=sum(abs(v)>1e-10 for v in dF),EM_changed=sum(abs(v)>1e-10 for v in dE),first_token_divergences=sum(x["first_divergence_step"]==0 for x in diffs),eos_count=sum(b["eos"] for b in bb),output_truncated_indices=[b["index"] for b in bb if b["output_truncated"]],endoftext_indices=[b["index"] for b in bb if b["contains_endoftext"]],no_input_truncation=True,official_scores_recomputed=True,only_expected_processor=True,paper_full=55.84,B_minus_paper=sum(scores)/200-55.84,A_minus_paper=sum(x["F1"] for x in aa)/200-55.84,dual_eos_diagnostic=dict(new_inference=False,F1=sum(x["F1"] for x in dual)/200,changed_indices=[x["index"] for x in dual if x["changed"]]))
dump("comparison_A_B.json",summary);dump("changed_cases.json",diffs);dump("dual_eos_offline.json",dual)
print(json.dumps(summary,indent=2))
