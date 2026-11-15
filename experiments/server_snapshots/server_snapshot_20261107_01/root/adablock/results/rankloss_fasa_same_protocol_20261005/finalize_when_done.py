from pathlib import Path
import os,json,time,subprocess,fcntl,hashlib
W=Path(__file__).parent
lock=(W/"finalizer.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(x):
 p=W/"finalizer_status.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2));t.replace(p)
status(dict(stage="waiting",pid=os.getpid()))
while True:
 s=json.loads((W/"manager_status.json").read_text())
 if s["stage"] in ("completed","failed"):break
 time.sleep(20)
if s["stage"]=="failed":
 status(dict(stage="blocked_by_worker_failure",manager=s));raise SystemExit(1)
with (W/"analysis.log").open("a") as log:
 p=subprocess.run(["/root/adablock/.venvs/fasa_full_20261005/bin/python",str(W/"analyze.py")],cwd=W,stdout=log,stderr=subprocess.STDOUT)
if p.returncode:
 status(dict(stage="analysis_failed",exit_code=p.returncode));raise SystemExit(p.returncode)
gpu=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True)
proc=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],capture_output=True,text=True)
(W/"gpu_after_completion.json").write_text(json.dumps(dict(gpus=gpu.stdout,compute_processes=proc.stdout,manager=s),indent=2))
doc=Path("/root/adablock/docs/ranker_full_training_20261003.md")
heading="## RankLoss versus FASA under the same generation protocol (2026-10-05)"
old=doc.read_text()
if heading not in old:
 report=(W/"REPORT.md").read_text()
 with doc.open("a") as f:f.write("\n\n"+heading+"\n\n"+report.replace("# RankLoss and FASA under the same generation protocol\n","",1)+"\n\nExperiment directory: "+str(W)+".\n")
status(dict(stage="completed",report=str(W/"REPORT.md"),comparison=str(W/"comparison.json"),
 report_sha256=hashlib.sha256((W/"REPORT.md").read_bytes()).hexdigest(),gpu_check=str(W/"gpu_after_completion.json")))
