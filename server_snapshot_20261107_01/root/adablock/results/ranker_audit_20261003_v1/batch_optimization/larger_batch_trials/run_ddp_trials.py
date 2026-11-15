import pathlib,subprocess,json,time,os,signal
root=pathlib.Path('/root/adablock');out=pathlib.Path(__file__).parent
while json.loads((out/'isolated_status.json').read_text())['state']!='completed':time.sleep(3)
for local in [8,16,32]:
 tag=f'ddp_local{local}'
 cmd=['/root/miniconda3/envs/ada/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4','scripts/benchmark_ranker_ddp.py','--local-batch',str(local),'--tag',tag]
 with (out/(tag+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
  (out/'ddp_trial_status.json').write_text(json.dumps({'state':'running','local_batch':local,'global_batch':4*local,'pid':p.pid,'argv':cmd},indent=2))
  try:rc=p.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   (out/'ddp_trial_status.json').write_text(json.dumps({'state':'failed','reason':'own short-test process group timed out after300s','pid':p.pid},indent=2))
   raise
 if rc:
  (out/'ddp_trial_status.json').write_text(json.dumps({'state':'failed','exit_code':rc,'tag':tag},indent=2))
  raise RuntimeError(f'{tag} failed')
(out/'ddp_trial_status.json').write_text(json.dumps({'state':'completed','local_batches':[8,16,32]},indent=2))
