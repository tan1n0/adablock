import pathlib,subprocess,json,time,traceback
OUT=pathlib.Path(__file__).parent;ROOT=pathlib.Path('/root/adablock');PY='/root/miniconda3/envs/ada/bin/python'
try:
 while pathlib.Path('/proc/1441').exists():time.sleep(2)
 assert (OUT/'multistep_verification.json').exists(),(OUT/'multistep_verification.log').read_text()[-5000:]
 checks=['tests/test_ranking_batch.py','tests/test_block_ranker.py']
 for test in checks:
  name=pathlib.Path(test).name
  with (OUT/(name+'.log')).open('x') as log:
   rc=subprocess.call([PY,'-m','unittest','discover','-s','tests','-p',name,'-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert rc==0,name
 cmd=[PY,'-u','scripts/train_block_ranker_batched.py','--train-dir','data/ranker_cobs_train_full_32k_20261003','--val-dir','data/ranker_cobs_validation16_20261003','--output-dir','checkpoints/ranker_batched_entry_probe_20261003','--epochs','1','--batch-size','8','--lr','2e-4','--coverage','0.95','--max-train-scenes','32768','--seed','42','--cpu-threads','1','--device','cuda:0']
 (OUT/'entry_probe_command.json').write_text(json.dumps(cmd,indent=2))
 start=time.perf_counter()
 with (OUT/'entry_probe.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  (OUT/'entry_probe_status.json').write_text(json.dumps({'state':'running','pid':p.pid,'argv':cmd},indent=2))
  rc=p.wait()
 status={'state':'completed' if rc==0 else 'failed','exit_code':rc,'seconds':time.perf_counter()-start}
 (OUT/'entry_probe_status.json').write_text(json.dumps(status,indent=2))
 assert rc==0,'Entry probe failed'
 h=json.loads((ROOT/'checkpoints/ranker_batched_entry_probe_20261003/history.json').read_text())[-1]
 b=json.loads((OUT/'benchmark_v2.json').read_text())
 estimate=b['estimated_10_epoch_training_hours']['batched_threads1']+10*h['validation']['evaluation_seconds']/3600
 result={'entry_training_scenes':h['scenes'],'entry_training_seconds':h['training_seconds'],'entry_train_loss':h['train_loss'],'entry_validation_scenes':h['validation']['scenes'],'entry_validation_seconds':h['validation']['evaluation_seconds'],'entry_scenes_per_second':h['scenes']/h['training_seconds'],'estimated_10_epochs_hours_length_weighted_plus_validation':estimate,'note':'Entry probe sees a seeded prefix of shuffled shards; use length-weighted benchmark for forecast, not prefix throughput alone.'}
 (OUT/'entry_cost_review.json').write_text(json.dumps(result,indent=2))
except Exception:
 (OUT/'ENTRY_PROBE_BLOCKED').write_text(traceback.format_exc());raise
