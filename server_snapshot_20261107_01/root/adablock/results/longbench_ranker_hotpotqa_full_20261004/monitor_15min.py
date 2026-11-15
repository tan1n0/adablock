from pathlib import Path
import json,time,subprocess,hashlib,sys
R=Path('/root/adablock');O=R/'results/longbench_ranker_hotpotqa_full_20261004'
sys.path.insert(0,str(R))
from utils.longbench_eval import normalize_answer
from collections import Counter
names=['full','cobs','quest','learned']
def f1(pred,answers):
 values=[]
 for ans in answers:
  p=normalize_answer(pred).split();g=normalize_answer(ans).split()
  hit=sum((Counter(p)&Counter(g)).values())
  values.append(2*hit/(len(p)+len(g)) if hit else 0.)
 return max(values)*100
def inspect():
 manifest=json.loads((O/'input_manifest.json').read_text())
 source=json.loads((O/'selected_rows.json').read_text())
 assert len(source)==len(manifest)==200 and len({x['_id'] for x in source})==200
 state=json.loads((O/'status.json').read_text())
 report={'observed_unix_time':time.time(),'pipeline_stage':state['stage'],'methods':{},'anomalies':[]}
 for n in names:
  d=O/n;p=d/(n+'.jsonl');rows=[]
  if p.exists():
   for line in p.read_text().splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:break
  errors=[]
  for i,r in enumerate(rows):
   if r['index']!=i or r['input_ids_sha256']!=manifest[i]['input_ids_sha256']:errors.append('identity_or_input_mismatch_'+str(i))
   if abs(f1(r['prediction'],source[i]['answers'])-r['score'])>1e-9:errors.append('score_mismatch_'+str(i))
  report['methods'][n]={'rows':len(rows),'prefix_identity_and_score_valid':not errors,'official_f1_if_complete':sum(r['score'] for r in rows)/200 if len(rows)==200 else None,'input_truncated':sum(r['input_truncated'] for r in rows),'output_truncated':sum(r['output_truncated'] for r in rows),'errors':errors}
  report['anomalies']+=errors
 try:report['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu','--format=csv'],text=True,timeout=20)
 except Exception as e:report['gpu_snapshot_error']=str(e)
 with (O/'monitor_15min.jsonl').open('a') as f:f.write(json.dumps(report)+'\n');f.flush()
 (O/'monitor_latest.json').write_text(json.dumps(report,indent=2))
 if state['stage']=='completed':
  assert all(v['rows']==200 and v['prefix_identity_and_score_valid'] for v in report['methods'].values()),report
  report['unique_dataset_ids']=200
  report['checkpoint_sha256']=hashlib.sha256((R/'checkpoints/ranker_ddp_fresh_global64_20261003/best.pt').read_bytes()).hexdigest()
  (O/'independent_final_verification.json').write_text(json.dumps(report,indent=2))
  return True
 return state['stage']=='blocked'
while True:
 if inspect():break
 # Passive observer only; no restart, configuration changes or GPU work.
 for _ in range(30):
  time.sleep(30)
  try:
   if json.loads((O/'status.json').read_text())['stage'] in ['completed','blocked']:break
  except (OSError,json.JSONDecodeError):pass
