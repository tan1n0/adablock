"""Controlled full train -> frozen held-out test build -> comparison."""
import pathlib,json,subprocess,time,datetime,traceback,hashlib,sys,os
ROOT=pathlib.Path('/root/adablock');OUT=pathlib.Path(__file__).parent;PY='/root/miniconda3/envs/ada/bin/python'
CKPT=ROOT/'checkpoints/ranker_ddp_fresh_global64_20261003'
TEST=ROOT/'data/ranker_cobs_test16_ddp_fresh_20261003'
def status(stage,**extra):
 data={'stage':stage,'runner_pid':os.getpid(),'remote_time':datetime.datetime.now().astimezone().isoformat(),**extra}
 p=OUT/'formal_pipeline_status.tmp';p.write_text(json.dumps(data,indent=2));p.replace(OUT/'formal_pipeline_status.json')
 print(json.dumps(data),flush=True)
def run(name,cmd):
 start=time.perf_counter()
 with (OUT/(name+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  status(name,child_pid=p.pid,argv=cmd)
  rc=p.wait()
 (OUT/(name+'_result.json')).write_text(json.dumps({'argv':cmd,'exit_code':rc,'seconds':time.perf_counter()-start},indent=2))
 if rc:raise RuntimeError(name+' exit '+str(rc))
def idle_gpus():
 # Permit only the already paused predecessor's small CUDA context.
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 unexpected=[int(x) for x in active.split() if x.isdigit() and int(x) not in [2190,3358921]]
 if unexpected:raise RuntimeError('Unexpected GPU processes '+str(unexpected))
 if pathlib.Path('/proc/2190/status').exists():
  assert 'T (stopped)' in pathlib.Path('/proc/2190/status').read_text()
try:
 assert json.loads((OUT.parent/'ddp_resume_verification/result.json').read_text())['passed']
 assert not CKPT.exists() and not TEST.exists()
 idle_gpus()
 cmd=[PY,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4','scripts/train_block_ranker_ddp.py','--train-dir','data/ranker_cobs_train_full_32k_20261003','--val-dir','data/ranker_cobs_validation16_20261003','--output-dir',str(CKPT),'--epochs','10','--local-batch','16','--lr','2e-4','--coverage','0.95','--seed','42','--cpu-threads','1']
 (OUT/'formal_train_command.json').write_text(json.dumps(cmd,indent=2))
 hashes={}
 for folder in ['models','utils','scripts']:
  for p in (ROOT/folder).glob('*.py'):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 (OUT/'source_sha256_at_formal_launch.json').write_text(json.dumps(hashes,indent=2))
 run('formal_training',cmd)
 assert (CKPT/'TRAINING_COMPLETE').exists()
 history=json.loads((CKPT/'history.json').read_text())
 assert len(history)==10 and all(r['scenes']==2691192 for r in history)
 idle_gpus()
 # Existing base model only; no download/API use.
 cmd=json.loads((OUT.parent.parent/'full_build_command.json').read_text())['argv'][:]
 cmd[cmd.index('--input-jsonl')+1]=str(OUT.parent/'final_test_selected.jsonl')
 cmd[cmd.index('--output-dir')+1]=str(TEST)
 run('final_test_build',cmd)
 sys.path.insert(0,str(ROOT))
 import torch
 from utils.ranking_data import RankingDataset,assert_disjoint,check_metadata
 from models.block_ranker import load_ranker
 from utils.ranking_batch_eval import evaluate_batched
 torch.set_num_threads(1)
 train=RankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003')
 val=RankingDataset(ROOT/'data/ranker_cobs_validation16_20261003')
 test=RankingDataset(TEST)
 assert_disjoint(train,test);assert_disjoint(val,test)
 assert len(test.manifest['documents'])==16
 model,payload=load_ranker(CKPT/'best.pt','cuda:0')
 check_metadata(payload['data_metadata'],test.metadata)
 status('final_evaluation',best_epoch=payload['training']['epoch'],test_scenes=len(test))
 outputs={}
 for name,data in [('validation',val),('heldout_test',test)]:
  result=evaluate_batched(model,data,'cuda:0')
  result.update(checkpoint=str(CKPT/'best.pt'),checkpoint_epoch=payload['training']['epoch'],
                split=name,selection_note='Validation selected checkpoint; heldout test frozen before formal training and not used for selection.')
  (OUT/(name+'_comparison.json')).write_text(json.dumps(result,indent=2))
  outputs[name]={'scenes':result['scenes'],'overall':result['groups']['overall'],'fixed_below_rho':result['groups'].get('fixed_below_rho')}
 (OUT/'final_comparison_summary.json').write_text(json.dumps(outputs,indent=2))
 status('completed',checkpoint=str(CKPT/'best.pt'),comparison=str(OUT/'final_comparison_summary.json'))
 (OUT/'PIPELINE_COMPLETE').write_text('Full 10 epochs and independent held-out comparison completed.')
except Exception as e:
 status('blocked',error=str(e),traceback=traceback.format_exc())
 (OUT/'FORMAL_PIPELINE_BLOCKED').write_text(traceback.format_exc())
 raise
