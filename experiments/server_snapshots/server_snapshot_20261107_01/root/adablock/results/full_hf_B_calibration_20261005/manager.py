import json,os,time,subprocess,hashlib
from pathlib import Path
W=Path(__file__).parent;PY="/root/miniconda3/envs/ada/bin/python";DOC=Path("/root/adablock/docs/ranker_full_training_20261003.md")
def write(n,x):
 p=W/n;t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2));t.replace(p)
def note(s):
 with DOC.open("a") as f:f.write("\n\n"+s+"\n")
note("## Authorized B-only Full calibration\nDirectory: "+str(W)+". Exactly A generation settings and inputs except repetition_penalty1.05; GPU0 only; no other experiments. Actual processor must be RepetitionPenaltyLogitsProcessor1.05 only. Config/source hashes and per-question fsync journals saved.")
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0"
start=time.perf_counter();log=(W/"run.log").open("a")
p=subprocess.Popen([PY,str(W/"run_B.py"),"--variant","B","--limit","200","--out",str(W/"output"),"--execute"],cwd="/root/adablock",env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
while p.poll() is None:
 journal=W/"output/predictions.jsonl";n=len(journal.read_text().splitlines()) if journal.exists() else 0
 write("manager_status.json",dict(stage="running",worker_pid=p.pid,completed=n,elapsed_seconds=time.perf_counter()-start))
 time.sleep(5)
result=dict(stage="completed" if p.returncode==0 else "failed",returncode=p.returncode,wall_seconds=time.perf_counter()-start,worker_pid=p.pid)
if p.returncode==0:
 log=(W/"comparison.log").open("a")
 q=subprocess.run([PY,str(W/"finalize_B.py")],stdout=log,stderr=subprocess.STDOUT);log.close()
 result["analysis_returncode"]=q.returncode
 if q.returncode==0:result["comparison"]=json.loads((W/"comparison_A_B.json").read_text())
 else:result["stage"]="analysis_failed"
if (W/"output/summary.json").exists():result["run_summary"]=json.loads((W/"output/summary.json").read_text())
source=json.loads((W/"source_hashes.json").read_text())
result["sources_unchanged"]=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in source.items())
gpu=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True)
apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],capture_output=True,text=True)
write("gpu_release_receipt.json",dict(returncode=gpu.returncode,gpus=gpu.stdout,compute_apps=apps.stdout))
write("manager_status.json",result);write("final_audit.json",result)
note("B-only calibration ended: "+json.dumps(result)+". GPU receipt: "+gpu.stdout+"\nA/B isolates local penalty effect; no claim that authors used this setting and no score-target configuration selection.")
if result["stage"]=="completed":(W/"COMPLETE").write_text("B200 and offline paired analysis complete.")
