import os,json,time,subprocess,hashlib,traceback
from pathlib import Path
W=Path(__file__).parent;ROOT=Path("/root/adablock");PY="/root/miniconda3/envs/ada/bin/python"
DOC=ROOT/"docs/ranker_full_training_20261003.md"
def write(n,x):
 p=W/n;t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2));t.replace(p)
def note(s):
 with DOC.open("a") as f:f.write("\n\n"+s+"\n")
start=time.perf_counter()
note("## Authorized A-only native HF Full calibration started\nDirectory: "+str(W)+". GPU0 only; no B/16-block/training. Same200token hashes, cached Qwen revision a09a35458c702b33eeacc393d103063234e8bc28,BF16/SDPA,greedy,penalty1.0,EOS151645,maxnew32. First8 fixed questions gated by token parity; stop at any mismatch. Per-sample fsync resume journal; no source result overwrite.")
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0"
log=(W/"run.log").open("a")
p=subprocess.Popen([PY,str(W/"run_A.py"),"--variant","A","--limit","200","--out",str(W/"output"),"--execute"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
log.close();pilot_noted=False
while p.poll() is None:
 out=W/"output";count=0
 if (out/"predictions.jsonl").exists():count=len((out/"predictions.jsonl").read_text().splitlines())
 if (out/"pilot8_passed.json").exists() and not pilot_noted:
  note("A-only pilot8 token parity passed; continuing existing process to200.");pilot_noted=True
 write("manager_status.json",dict(stage="running",worker_pid=p.pid,completed=count,pilot8_passed=pilot_noted,elapsed_seconds=time.perf_counter()-start))
 time.sleep(5)
out=W/"output";mismatch=out/"PARITY_MISMATCH.json"
result=dict(stage="completed" if p.returncode==0 else "stopped_parity_mismatch" if mismatch.exists() else "failed",returncode=p.returncode,worker_pid=p.pid,wall_seconds=time.perf_counter()-start,pilot8_passed=(out/"pilot8_passed.json").exists())
if (out/"summary.json").exists():result["summary"]=json.loads((out/"summary.json").read_text())
if mismatch.exists():result["mismatch"]=json.loads(mismatch.read_text())
original=json.loads((W/"source_hashes.json").read_text())
result["source_hashes_unchanged"]=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in original.items())
q=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True)
apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],capture_output=True,text=True)
write("gpu_release_receipt.json",dict(returncode=q.returncode,gpus=q.stdout,compute_apps=apps.stdout))
write("manager_status.json",result);write("final_audit.json",result)
note("A-only calibration ended: "+json.dumps(result)+". GPU receipt: "+q.stdout)
if p.returncode==0:(W/"COMPLETE").write_text("A200 complete; no B executed.")
