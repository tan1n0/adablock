"""Read-only training observer plus authorized final document-weighted evaluation."""
from pathlib import Path
import json,time,os,sys,subprocess,fcntl,traceback,hashlib
import numpy as np
import torch
R=Path("/root/adablock");W=R/"results/ranker_all28_curve_20261005";O=R/"checkpoints/ranker_all28_curve_20261005";PY=sys.executable
lock=(W/"observer.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((R/"data/ranker_cobs_train_all28_20261004/manifest.json").read_text())
ends=np.cumsum([e["scenes"] for e in manifest["shards"]]);starts=np.r_[0,ends[:-1]]
docs=np.array([e["document_index"] for e in manifest["shards"]]);layers=np.array([e["layer_id"] for e in manifest["shards"]])
plan=np.load(W/"train_plan.npy",mmap_mode="r")
def status(stage,**kw):
 row={"stage":stage,"pid":os.getpid(),"time":time.time(),**kw};tmp=W/"observer_status.tmp";tmp.write_text(json.dumps(row,indent=2));tmp.replace(W/"observer_status.json");print(json.dumps(row),flush=True)
def audit(step):
 target=W/f"actual_coverage_step_{step:06d}.json"
 if target.exists():return
 ckpt=O/f"step_{step:06d}_resume.pt"
 if not ckpt.exists():return
 x=torch.load(ckpt,weights_only=True,map_location="cpu");assert x["step"]==step
 batch=x["next_batch"];ids=np.asarray(plan[:batch]).reshape(-1);si=np.searchsorted(ends,ids,side="right")
 d=docs[si];l=layers[si];rec=(ids-starts[si])//4;heads=(ids-starts[si])%4
 report={"optimizer_steps":step,"actual_batches":batch,"presentations":len(ids),"unique_scenes":len(np.unique(ids)),
         "unique_query_positions":len(np.unique(d.astype(np.int64)*1048576+rec)),"query_identity":"document plus aligned record index; all28 layer record positions were previously audited identical",
         "documents":len(np.unique(d)),"doc_layer_pairs":len(np.unique(d*28+l)),
         "layer_counts":np.bincount(l,minlength=28).tolist(),"head_counts":np.bincount(heads,minlength=4).tolist(),
         "plan_sha256":x["plan_sha256"],"checkpoint_bytes":ckpt.stat().st_size,"rank_rng_states":len(x["rng"])}
 target.write_text(json.dumps(report,indent=2))
 with (R/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n### Actual coverage at"+str(step)+" optimizer steps\n\n"+json.dumps(report)+"\n")
def gpu_apps():
 return subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
try:
 while True:
  for step in [5000,20000,50000]:audit(step)
  manager=json.loads((W/"status.json").read_text());progress=json.loads((O/"progress.json").read_text())
  if manager["stage"]=="failed":
   status("training_failed",manager=manager);sys.exit(1)
  if manager["stage"]=="complete":break
  status("observing",training_stage=progress["stage"],step=progress.get("step"))
  time.sleep(30)
 # Wait until original trainer processes exit and release GPU contexts; never stop another job.
 while gpu_apps():
  status("waiting_for_gpu_release_before_final_metrics",apps=gpu_apps());time.sleep(15)
 if not (W/"final_validation_both_weights.json").exists():
  command=[PY,"-m","torch.distributed.run","--standalone","--nproc_per_node=4","scripts/eval_ranker_curve_document_weights.py"]
  for attempt in range(3):
   with (W/f"final_document_eval_attempt_{attempt}.log").open("a") as f:
    p=subprocess.Popen(command,cwd=R,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
    status("final_document_metrics",child_pid=p.pid,command=command,attempt=attempt);code=p.wait()
   if code==0:break
   if attempt==2:raise RuntimeError("Final document evaluation failed repeatedly")
   time.sleep(10)
 assert (W/"final_validation_both_weights.json").exists()
 time.sleep(5);apps=gpu_apps()
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],text=True)
 receipt={"compute_apps_after_completion":apps,"gpu_snapshot":gpu,"our_train_and_eval_processes_exited":True,"all_gpu_compute_contexts_released":not bool(apps)}
 (W/"gpu_release_receipt.json").write_text(json.dumps(receipt,indent=2))
 with (R/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n### GPU release after all curve work\n\n"+json.dumps(receipt)+"\n")
 status("complete",receipt=receipt,best_step=json.loads((O/"selection.json").read_text())["best_step"])
except Exception:
 status("failed",error=traceback.format_exc());raise
