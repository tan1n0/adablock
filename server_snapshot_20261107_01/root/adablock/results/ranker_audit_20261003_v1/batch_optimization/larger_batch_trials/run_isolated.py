import pathlib,subprocess,json
out=pathlib.Path(__file__).parent
for batch in [8,32,64]:
 tag=f'isolated_b{batch}'
 cmd=['/root/miniconda3/envs/ada/bin/python',str(out/'trial_batch.py'),'--batch',str(batch),'--device','cuda:1','--tag',tag]
 with (out/(tag+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,cwd='/root/adablock',stdout=log,stderr=subprocess.STDOUT)
  (out/'isolated_status.json').write_text(json.dumps({'state':'running','batch':batch,'pid':p.pid},indent=2))
  rc=p.wait()
 if rc:raise RuntimeError(f'batch {batch} failed: {rc}')
(out/'isolated_status.json').write_text(json.dumps({'state':'completed','batches':[8,32,64]},indent=2))
