from pathlib import Path
import os,time,json,subprocess,fcntl,signal
W=Path(__file__).parent
lock=(W/"smoke_only_guard.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
hold=json.loads((W/"hold_after_smoke.json").read_text());oldpid=hold["stopped_manager_pid"]
assert hold["state_before"]["variant"]=="layer0_full" and hold["state_before"]["limit"]==8
def write(x):
 p=W/"manager_status.json";tmp=p.with_suffix(".guard.tmp");tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def running(pid):
 p=Path(f"/proc/{pid}/stat")
 return p.exists() and p.read_text().split()[2] not in ("Z","X")
write(dict(stage="running_smoke_only",variant="layer0_full",limit=8,pid=hold["current_worker"],guard_pid=os.getpid(),stopped_manager_pid=oldpid,full200_on_hold=True))
while running(hold["current_worker"]):time.sleep(5)
if not (W/"layer0_full/run/smoke8_passed.json").exists():
 write(dict(stage="failed",error="first smoke worker exited without smoke gate",full200_on_hold=True));raise SystemExit(1)
env=dict(os.environ,RANKLOSS_LIMIT="8",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
with (W/"all_layers/worker_limit8.log").open("a") as log:
 p=subprocess.Popen(["/root/adablock/.venvs/fasa_full_20261005/bin/python","-u",str(W/"all_layers/run_rankloss.py")],cwd=W/"all_layers",env=env,stdout=log,stderr=subprocess.STDOUT)
 write(dict(stage="running_smoke_only",variant="all_layers",limit=8,pid=p.pid,guard_pid=os.getpid(),stopped_manager_pid=oldpid,full200_on_hold=True))
 rc=p.wait()
 if rc or not (W/"all_layers/run/smoke8_passed.json").exists():
  write(dict(stage="failed",error=f"all_layers smoke failed exit {rc}",full200_on_hold=True));raise SystemExit(1)
summary={}
for v in ("layer0_full","all_layers"):
 rows=[json.loads(l) for l in (W/v/"run/predictions.jsonl").read_text().splitlines()]
 assert len(rows)==8
 summary[v]=dict(samples=8,sample_seconds=sum(x["seconds"] for x in rows),mean_seconds=sum(x["seconds"] for x in rows)/8,
  rows=[dict(index=x["index"],seconds=x["seconds"],input_tokens=x["input_tokens"],generated_tokens=len(x["token_ids"])) for x in rows])
(W/"smoke_timing.json").write_text(json.dumps(summary,indent=2))
gpu=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout
write(dict(stage="paused_after_smoke",completed_per_variant=8,full200_on_hold=True,stopped_manager_pid=oldpid,guard_pid=os.getpid(),gpu=gpu,timing=summary))
# The original orchestrator remains stopped; never resume it without replacing its automatic 200-stage loop.
