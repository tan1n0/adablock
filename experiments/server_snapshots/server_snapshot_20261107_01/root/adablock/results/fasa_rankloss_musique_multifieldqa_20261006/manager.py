from pathlib import Path
import os,json,time,subprocess,fcntl,hashlib,traceback
W=Path(__file__).parent
lock=(W/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='/root/adablock/.venvs/fasa_full_20261005/bin/python'
tasks={'musique':200,'multifieldqa_en':150}
def write(name,obj):
 p=W/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def rows(task,method):return [json.loads(l) for l in (W/task/method/'run/predictions.jsonl').read_text().splitlines()]
def run(task,method,limit):
 d=W/task/method
 env=dict(os.environ,EVAL_LIMIT=str(limit),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 with (d/f'worker_limit{limit}.log').open('a') as log:
  p=subprocess.Popen([PY,'-u',str(d/'run_eval.py')],cwd=d,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
  write('manager_status.json',dict(stage='smoke' if limit==2 else 'full',task=task,method=method,limit=limit,pid=p.pid,manager_pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
  rc=p.wait()
 if rc:raise RuntimeError(f'{task}/{method} exit{rc}; inspect worker log; no automatic retry')
 assert len(rows(task,method))==limit
def pair_check(task,n):
 a,b=rows(task,'fasa'),rows(task,'all_layers')
 assert len(a)>=n and len(b)>=n
 for x,y in zip(a[:n],b[:n]):
  assert x['index']==y['index'] and x['id']==y['id'] and x['input_ids_sha256']==y['input_ids_sha256']
  assert x['token_ids'][0]==y['token_ids'][0]
start=time.time()
try:
 for task in tasks:
  for method in ['fasa','all_layers']:run(task,method,2)
  pair_check(task,2)
 estimate={}
 for task,n in tasks.items():
  estimate[task]={m:sum(x['seconds'] for x in rows(task,m))/2*n for m in ['fasa','all_layers']}
 write('smoke_gate.json',dict(passed=True,samples_per_task_method=2,paired_input_and_dense_first_token=True,estimated_total_sample_seconds=estimate,estimate_note='First-two extrapolation; input-length and output-length distribution can differ.'))
 for task,n in tasks.items():
  for method in ['fasa','all_layers']:run(task,method,n)
  pair_check(task,n)
 subprocess.run([PY,str(W/'analyze.py')],cwd=W,check=True)
 write('manager_status.json',dict(stage='completed',wall_seconds=time.time()-start,manager_pid=os.getpid()))
except Exception as e:
 write('manager_status.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc(),wall_seconds=time.time()-start,manager_pid=os.getpid()))
 raise
