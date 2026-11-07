import os,sys,json,time,subprocess,fcntl
from pathlib import Path
W=Path(__file__).parent
lock=(W/"manager.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY="/root/adablock/.venvs/fasa_full_20261005/bin/python"
def write(x):
 p=W/"manager_status.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2));t.replace(p)
assert json.loads((W/"adapter_tests.json").read_text())["passed"]
start=time.perf_counter()
try:
 for limit in (8,200):
  for variant in ("layer0_full","all_layers"):
   if limit==200:
    assert all((W/v/"run/smoke8_passed.json").exists() for v in ("layer0_full","all_layers"))
   env=dict(os.environ,RANKLOSS_LIMIT=str(limit),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
   log=(W/variant/f"worker_limit{limit}.log").open("a")
   p=subprocess.Popen([PY,"-u",str(W/variant/"run_rankloss.py")],cwd=W/variant,env=env,stdout=log,stderr=subprocess.STDOUT)
   write(dict(stage="running",variant=variant,limit=limit,pid=p.pid,manager_pid=os.getpid()))
   rc=p.wait();log.close()
   if rc:raise RuntimeError(f"{variant} limit{limit} failed with exit {rc}; inspect worker log")
   expected="smoke8_passed.json" if limit==8 else "COMPLETE"
   assert (W/variant/"run"/expected).exists()
 write(dict(stage="completed",seconds=time.perf_counter()-start,manager_pid=os.getpid()))
except Exception as e:
 write(dict(stage="failed",error=str(e),seconds=time.perf_counter()-start,manager_pid=os.getpid()))
 raise
