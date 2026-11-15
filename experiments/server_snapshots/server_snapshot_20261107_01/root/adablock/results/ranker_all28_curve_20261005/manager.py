import os,sys,json,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path("/root/adablock");W=ROOT/"results/ranker_all28_curve_20261005";PY="/root/miniconda3/envs/ada/bin/python"
OUT=ROOT/"checkpoints/ranker_all28_curve_20261005"
lock=(W/"manager.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 row={"stage":stage,"time":time.time(),"pid":os.getpid(),**kw};(W/"status.json").write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def run(cmd,name):
 status(name,command=cmd)
 with (W/(name+".log")).open("a") as f:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
  status(name,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise RuntimeError(name+" exit "+str(code))
def launch(out,extra,name):
 cmd=[PY,"-m","torch.distributed.run","--standalone","--nproc_per_node=4","scripts/train_ranker_curve.py","--work",str(W),"--out",str(out)]+extra
 run(cmd,name)
try:
 if not (W/"TESTS_PASSED.json").exists():
  run([PY,"-m","unittest","tests.test_ranking_batch","tests.test_block_ranker"],"unit_tests")
  run([PY,"-m","torch.distributed.run","--standalone","--nproc_per_node=4","scripts/check_ranker_curve_ddp.py"],"ddp_normalization_test")
  launch(W/"test_continuous_balanced",["--targets","64","--test"],"test_continuous_balanced")
  launch(W/"test_resumed_balanced",["--targets","64","--test","--stop-after","32"],"test_stop32_balanced")
  launch(W/"test_resumed_balanced",["--targets","64","--test","--resume"],"test_resume64_balanced")
  import torch,numpy as np
  a=torch.load(W/"test_continuous_balanced/resume_latest.pt",weights_only=True,map_location="cpu");b=torch.load(W/"test_resumed_balanced/resume_latest.pt",weights_only=True,map_location="cpu")
  def equal(x,y):
   if isinstance(x,torch.Tensor):return torch.equal(x,y)
   if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
   if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
   return x==y
  assert a["step"]==b["step"]==64 and a["next_batch"]==b["next_batch"]
  exact={k:equal(a[k],b[k]) for k in ["model","optimizer","rng"]}
  if not all(exact.values()):raise AssertionError("resume not bitwise equal: "+str(exact))
  sys.path.insert(0,str(ROOT))
  from utils.ranking_curve import MMapDataset
  from utils.ranking_data import RankingDataset
  plan=np.load(W/"train_plan.npy",mmap_mode="r")
  new=MMapDataset(ROOT/"data/ranker_cobs_train_all28_20261004",True);old=RankingDataset(new.directory)
  for index in plan[0,[0,16,32,48]]:
   x=new[int(index)];y=old[int(index)]
   assert all(equal(x[k],y[k]) for k in y)
  proof={"unit_tests":"passed","DDP_normalization":"passed","mmap_scene_identity":"bitwise equal four length buckets","resume_exact":exact,"all28_embedding_update":"passed at64 steps"}
  (W/"TESTS_PASSED.json").write_text(json.dumps(proof,indent=2))
  with (ROOT/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n### All28 curve tests passed\n\n"+json.dumps(proof)+"\n")
 for attempt in range(4):
  if (OUT/"COMPLETE").exists():break
  try:
   launch(OUT,["--resume"] if (OUT/"resume_latest.pt").exists() else [],"training_attempt_"+str(attempt))
  except Exception:
   status("training_retry",attempt=attempt,error=traceback.format_exc())
   if attempt==3 or not (OUT/"resume_latest.pt").exists():raise
   time.sleep(10)
 if not (OUT/"COMPLETE").exists():raise RuntimeError("Training returned without COMPLETE")
 status("complete",output=str(OUT),result=json.loads((OUT/"COMPLETE").read_text()))
except Exception:
 status("failed",error=traceback.format_exc());raise
