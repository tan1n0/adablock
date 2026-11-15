import pathlib,subprocess,json,time,datetime,os,sys,hashlib,traceback,collections
ROOT=pathlib.Path('/root/adablock')
OUT=pathlib.Path(__file__).parent
PY='/root/miniconda3/envs/ada/bin/python'
TRAIN=ROOT/'data/ranker_cobs_train_full_32k_20261003'
VAL=ROOT/'data/ranker_cobs_validation16_20261003'
SANITY=ROOT/'checkpoints/ranker_full_20261003_sanity'
def status(stage,**kw):
 payload={'stage':stage,'pid':os.getpid(),'remote_time':datetime.datetime.now().astimezone().isoformat(),**kw}
 tmp=OUT/'continuation_status.tmp'
 tmp.write_text(json.dumps(payload,indent=2)); tmp.replace(OUT/'continuation_status.json')
 print(json.dumps(payload),flush=True)
def run(label,argv):
 start=time.monotonic()
 with (OUT/(label+'.log')).open('x') as log:
  p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  status(label,child_pid=p.pid,argv=argv)
  rc=p.wait()
 result={'argv':argv,'exit_code':rc,'seconds':time.monotonic()-start}
 (OUT/(label+'_result.json')).write_text(json.dumps(result,indent=2))
 if rc: raise RuntimeError(label+' failed with exit code '+str(rc))
 return result
try:
 status('waiting_for_existing_full_build',build_pid=6814,poll_seconds=300)
 while True:
  s=json.loads((OUT/'full_build_status.json').read_text())
  if s['state']=='completed': break
  if s['state']=='failed': raise RuntimeError('Full build failed; see full_build.log')
  if not pathlib.Path('/proc/6814').exists(): raise RuntimeError('Full build PID absent without successful status')
  time.sleep(300)
 status('checking_full_manifest')
 m=json.loads((TRAIN/'manifest.json').read_text())
 assert len(m['documents'])==1751, len(m['documents'])
 expected={(i,l) for i in range(1751) for l in [12,20,27]}
 assert {(e['document_index'],e['layer_id']) for e in m['shards']}==expected
 assert len(m['shards'])==5253
 assert all((TRAIN/e['file']).stat().st_size>0 for e in m['shards'])
 assert not list(TRAIN.glob('*.tmp'))
 assert {e['file'] for e in m['shards']}=={p.name for p in TRAIN.glob('*.pt')}
 actual={}
 with (ROOT/'data/train_strict_packed_tokens.jsonl').open() as f:
  for row in map(json.loads,f): actual[(row['source'],str(row.get('doc_id','')))]=hashlib.sha256(row['text'].encode()).hexdigest()
 assert all(actual[(d['source'],d['doc_id'])]==d['text_sha256'] for d in m['documents'])
 summary={'documents':len(m['documents']),'shards':len(m['shards']),'scenes':sum(e['scenes'] for e in m['shards']),'bytes':sum((TRAIN/e['file']).stat().st_size for e in m['shards']),'manifest_source_hashes_match':True}
 (OUT/'full_manifest_audit.json').write_text(json.dumps(summary,indent=2))
 # Keep user's active source edits intact; do not execute a silently changed pipeline.
 launch=json.loads((OUT/'source_sha256_at_full_launch.json').read_text())
 for name in ['scripts/build_ranking_dataset.py','scripts/build_oracle_dataset.py','scripts/train_block_ranker.py','utils/ranking_data.py','utils/ranking_eval.py','utils/ranking_loss.py','utils/block_oracle.py','utils/qk_block_oracle.py','models/block_ranker.py']:
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==launch[name], 'source changed: '+name
 # Do not contend with a GPU job started after full build.
 while True:
  usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if all(int(x)<256 for x in usage.split()): break
  status('waiting_for_idle_gpus',memory_mib=usage.strip()); time.sleep(300)
 args=json.loads((OUT/'full_build_command.json').read_text())['argv'][:]
 args[args.index('--input-jsonl')+1]=str(OUT/'validation_selected.jsonl')
 args[args.index('--output-dir')+1]=str(VAL)
 assert not VAL.exists(),str(VAL)
 run('validation_build',args)
 sys.path.insert(0,str(ROOT))
 from utils.ranking_data import RankingDataset,assert_disjoint
 import torch
 torch.set_num_threads(4)
 train,val=RankingDataset(TRAIN),RankingDataset(VAL)
 assert_disjoint(train,val)
 maxerr=0.; scene_count=0; positions=[]
 for e in val.shards:
  shard=torch.load(VAL/e['file'],map_location='cpu',weights_only=True)
  assert len(shard['records'])==e['records']
  for r in shard['records']:
   mass=r['teacher_mass']; fixed=r['fixed_mass']
   assert torch.isfinite(mass).all() and torch.isfinite(fixed).all()
   assert (mass>=0).all()
   assert ((r['block_ids']+1)*val.metadata['block_size']<=r['token_position']).all()
   err=float((fixed+mass.sum(-1)-1).abs().max()); maxerr=max(maxerr,err)
   assert err<2e-4,err
   scene_count+=len(fixed); positions.append(r['token_position'])
 (OUT/'validation_audit.json').write_text(json.dumps({'documents':len(val.manifest['documents']),'shards':len(val.shards),'scenes':scene_count,'mass_sum_max_error':maxerr,'sample_position_minmax':[min(positions),max(positions)],'train_validation_disjoint':True},indent=2))
 # Sanity is only a training health/cost probe, never the final quality result.
 assert not SANITY.exists(),str(SANITY)
 cmd=[PY,'-u','scripts/train_block_ranker.py','--train-dir',str(TRAIN),'--val-dir',str(VAL),'--output-dir',str(SANITY),'--epochs','2','--batch-size','8','--lr','2e-4','--coverage','0.95','--max-train-scenes','512','--max-val-scenes','256','--seed','42','--device','cuda:0']
 result=run('training_sanity',cmd)
 h=json.loads((SANITY/'history.json').read_text())
 assert len(h)==2 and (SANITY/'best.pt').exists() and (SANITY/'last.pt').exists()
 assert all(x['valid_scenes']>0 and x['pairs']>0 for x in h)
 review={'sanity_seconds':result['seconds'],'history_summary':[{k:v for k,v in row.items() if k!='validation'} for row in h],'full_train_scenes':len(train),'validation_scenes':len(val),'note':'Sanity includes validation and startup; do not extrapolate exact full cost from 1024 scenes. Review length mix and throughput before default 10-epoch run. No final training/evaluation launched.'}
 (OUT/'sanity_cost_review.json').write_text(json.dumps(review,indent=2))
 status('sanity_complete_pending_full_training_cost_review',**review)
 (OUT/'SANITY_COMPLETE').write_text('Full construction + validation + sanity complete; formal training pending cost review.')
except Exception as e:
 status('blocked',error=str(e),traceback=traceback.format_exc())
 (OUT/'CONTINUATION_BLOCKED').write_text(traceback.format_exc())
 raise
