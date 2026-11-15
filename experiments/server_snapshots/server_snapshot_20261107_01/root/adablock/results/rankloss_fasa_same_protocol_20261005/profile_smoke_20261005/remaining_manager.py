from pathlib import Path
import json,subprocess,os,fcntl,time
W=Path(__file__).parent
lock=(W/"lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(s):
 p=W/"status.json";tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(s,indent=2));tmp.replace(p)
assert json.loads((W.parent/"manager_status.json").read_text())["stage"]=="paused_after_smoke"
try:
 for v in ("layer0_full","all_layers"):
  with (W/v/"worker_remaining.log").open("x") as log:
   p=subprocess.Popen(["/root/adablock/.venvs/fasa_full_20261005/bin/python","-u",str(W/v/"run_profile.py")],cwd=W/v,stdout=log,stderr=subprocess.STDOUT)
   status(dict(stage="running",variant=v,worker_pid=p.pid,manager_pid=os.getpid(),max_samples=3))
   rc=p.wait()
  if rc:raise RuntimeError(f"{v} failed {rc}; no automatic repeat")
 status(dict(stage="completed",max_samples=3,manager_pid=os.getpid()))
except Exception as e:
 status(dict(stage="failed",error=str(e)));raise
