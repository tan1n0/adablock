from pathlib import Path
import json,os,sys,subprocess,time,hashlib,traceback
W=Path(__file__).parent;R=Path("/root/adablock")
def write(n,x):(W/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
try:
 audit=json.loads((W/"dependency_audit.json").read_text());assert audit["ada_unchanged"]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="0,1,2,3",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONNOUSERSITE="1")
 start=time.perf_counter()
 with (W/"run.log").open("a") as log:
  p=subprocess.Popen([str(R/".venvs/fasa_full_20261005/bin/python"),str(W/"run_official_full.py")],cwd=str(W),env=env,stdout=log,stderr=subprocess.STDOUT)
  write("status.json",dict(stage="running",manager_pid=os.getpid(),worker_pid=p.pid))
  rc=p.wait()
 assert rc==0,dict(worker_exit=rc)
 out=W/"run";new=[json.loads(l) for l in (out/"predictions.jsonl").read_text().splitlines()]
 assert len(new)==200 and (out/"COMPLETE").exists()
 summary=json.loads((out/"summary.json").read_text());comparisons={}
 for name,folder in [("A","full_hf_A_calibration_20261005_v2"),("B","full_hf_B_calibration_20261005")]:
  old=[json.loads(l) for l in (R/"results"/folder/"output/predictions.jsonl").read_text().splitlines()]
  assert len(old)==len(new)
  assert all(a["id"]==b["id"] and a["input_ids_sha256"]==b["input_ids_sha256"] for a,b in zip(new,old))
  changed=[]
  for a,b in zip(new,old):
   if a["token_ids"]!=b["token_ids"]:
    first=next((i for i,(x,y) in enumerate(zip(a["token_ids"],b["token_ids"])) if x!=y),min(len(a["token_ids"]),len(b["token_ids"])))
    changed.append(dict(index=a["index"],id=a["id"],first_different_token=first,official_prediction=a["prediction"],previous_prediction=b["prediction"],official_F1=a["F1"],previous_F1=b["F1"],answers=a["answers"]))
  comparisons[name]=dict(old_F1=sum(x["F1"] for x in old)/200,delta_F1=summary["F1"]-sum(x["F1"] for x in old)/200,token_changed=len(changed),F1_wins=sum(a["F1"]>b["F1"]+1e-9 for a,b in zip(new,old)),F1_ties=sum(abs(a["F1"]-b["F1"])<=1e-9 for a,b in zip(new,old)),F1_losses=sum(a["F1"]<b["F1"]-1e-9 for a,b in zip(new,old)))
  write("changed_vs_"+name+".json",changed)
 comparisons["paper"]=dict(reported_F1=55.9,delta_F1=summary["F1"]-55.9,limitation="Paper data ordering/hash and exact model revision are not published in inspected repository; this is pinned official-code Full reproduction on verified cached LongBench 200.")
 write("comparison.json",comparisons)
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv"],text=True)
 write("final_audit.json",dict(worker_exit=rc,manager_wall_seconds=time.perf_counter()-start,summary=summary,gpus_after_worker=gpu,dependencies=audit))
 report="# FASA official Full reproduction\n\n"+json.dumps(dict(summary=summary,comparison=comparisons),ensure_ascii=False,indent=2)+"\n\nOfficial commit 96c00c46db12ccee56104175470aa29579d66c58. Qwen2.5-7B-Instruct cached revision a09a35458c702b33eeacc393d103063234e8bc28. Native FA2 Full, BF16, transformers 4.43.0, torch 2.5.1+cu121, flash-attn 2.7.4.post1; seed42, greedy, max_new_tokens32, inherited repetition_penalty1.05, dualEOS, min_length=input_length+1. All 200 input hashes match A/B; no input truncation. Official prediction function AST unchanged. Adaptations: remove unused sparse imports; local cache only and native trust_remote_code=False; per-row journal/resume and smoke8 checkpoint; fixed four-GPU device map with full prefill logits retained, placement_plan.json records budgets. No FASA sparse or training executed. Timings include four-GPU placement and are not comparable to single-GPU A/B as a speed result. Paper exact dataset/model provenance unavailable; score difference cannot identify one cause.\n"
 (W/"REPORT.md").write_text(report)
 with (R/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n\n"+report+"\nArtifacts: "+str(W)+"\n")
 write("status.json",dict(stage="completed",summary=summary,comparison=comparisons))
 print(report,flush=True)
except Exception:
 write("status.json",dict(stage="failed",traceback=traceback.format_exc()))
 traceback.print_exc();sys.exit(1)
