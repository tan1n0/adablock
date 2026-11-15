"""Authorized full200 continuation; reuse eight audited rows; frozen inference sources."""
from pathlib import Path
import os,json,time,subprocess,fcntl,hashlib
W=Path(__file__).parent
lock=(W/"manager.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY="/root/adablock/.venvs/fasa_full_20261005/bin/python"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(s):
 p=W/"manager_status.json";tmp=p.with_suffix(".resume.tmp");tmp.write_text(json.dumps(s,indent=2));tmp.replace(p)
test=json.loads((W/"adapter_tests.json").read_text())
assert test["passed"] and test["adapter_sha256"]==sha(W/"rankloss_adapter.py")
assert test["checkpoint_sha256"]==sha(Path("/root/adablock/checkpoints/ranker_all28_curve_20261005/best.pt"))
for v in ("layer0_full","all_layers"):
 out=W/v/"run";cfg=json.loads((out/"config.json").read_text())
 assert cfg["script_sha256"]==sha(W/v/"run_rankloss.py")
 assert cfg["adapter_sha256"]==test["adapter_sha256"]
 assert (out/"smoke8_passed.json").exists()
 raw=(out/"predictions.jsonl").read_bytes()
 assert raw.endswith(b"\n")
 rows=[json.loads(l) for l in raw.decode().splitlines()]
 assert 8<=len(rows)<=200 and [x["index"] for x in rows]==list(range(len(rows)))
start=time.perf_counter()
try:
 for variant in ("layer0_full","all_layers"):
  out=W/variant/"run"
  if (out/"COMPLETE").exists():continue
  completed=len((out/"predictions.jsonl").read_text().splitlines())
  env=dict(os.environ,RANKLOSS_LIMIT="200",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
  with (W/variant/"worker_limit200.log").open("a") as log:
   p=subprocess.Popen([PY,"-u",str(W/variant/"run_rankloss.py")],cwd=W/variant,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
   write(dict(stage="running",variant=variant,limit=200,pid=p.pid,manager_pid=os.getpid(),resumed_from=completed,full200_on_hold=False,algorithm_unchanged=True,ssh_independent=True))
   rc=p.wait()
  if rc:raise RuntimeError(f"{variant} failed {rc}; preserved per-sample journal, no automatic repeat")
  assert (out/"COMPLETE").exists()
 write(dict(stage="completed",seconds=time.perf_counter()-start,manager_pid=os.getpid(),resumed_from_per_variant=8,algorithm_unchanged=True))
except Exception as e:
 write(dict(stage="failed",error=str(e),seconds=time.perf_counter()-start,manager_pid=os.getpid()))
 raise
