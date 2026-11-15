from pathlib import Path
import json,subprocess,hashlib
import numpy as np
W=Path(__file__).parent
result={};lines=['# FASA versus all-layer RankLoss: MuSiQue and MultiFieldQA-en','',
'Qwen2.5-7B-Instruct; frozen 50k ranker; existing verified HF4.43/FA2 environment and four-GPU map.',
'FASA: native256 per query head, FC16, layer0 dense. RankLoss: learned all28, block16/sink2/local32/middle32 per KV head, no COBS fallback. Budgets are unequal; full cache retained. Timing includes auditing and digest preprocessing.',
'Official task generation limits: MuSiQue32, MultiFieldQA-en64. Historical Full used128 and is not a same-protocol baseline; no new Full run. This experiment does not resolve the paper Full discrepancy.','',
'|Task|Method|N|F1|EM|Output capped|Seconds|','|---|---|---:|---:|---:|---:|---:|']
for task,n in [('musique',200),('multifieldqa_en',150)]:
 rr={}
 for m in ['fasa','all_layers']:
  d=W/task/m/'run';r=[json.loads(l) for l in (d/'predictions.jsonl').read_text().splitlines()]
  assert len(r)==n and [x['index']for x in r]==list(range(n))
  assert len({x['id']for x in r})==n and (d/'COMPLETE').exists()
  rr[m]=r
  s=json.loads((d/'summary.json').read_text());lines.append(f"|{task}|{m}|{n}|{s['F1']:.4f}|{s['EM']:.2f}|{s['output_truncated']}|{s['sample_seconds']:.1f}|")
 a,b=rr['fasa'],rr['all_layers']
 assert all(x['input_ids_sha256']==y['input_ids_sha256'] and x['token_ids'][0]==y['token_ids'][0] for x,y in zip(a,b))
 dif=np.array([y['F1']-x['F1'] for x,y in zip(a,b)])
 rng=np.random.default_rng(42);boot=np.mean(dif[rng.integers(0,n,size=(10000,n))],axis=1)
 result[task]={'N':n,'rankloss_minus_fasa_F1':float(dif.mean()),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'wins_ties_losses':[int((dif>1e-9).sum()),int((abs(dif)<=1e-9).sum()),int((dif< -1e-9).sum())],'changed_token_sequences':sum(x['token_ids']!=y['token_ids']for x,y in zip(a,b)),'all_inputs_match':True,'all_dense_first_tokens_match':True,'summaries':{m:json.loads((W/task/m/'run/summary.json').read_text())for m in rr}}
 lines+=['',f"{task}: RankLoss-FASA F1 {dif.mean():+.4f} pp; paired bootstrap95 {result[task]['paired_bootstrap95']}.",'']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
(W/'gpu_after_completion.json').write_text(json.dumps({'gpus':gpu,'compute_processes':proc},indent=2))
(W/'comparison.json').write_text(json.dumps(result,indent=2))
report='\n'.join(lines)+'\n\nArtifacts: '+str(W)+'\n'
(W/'REPORT.md').write_text(report)
doc=Path('/root/adablock/docs/ranker_full_training_20261003.md')
with doc.open('a')as f:f.write('\n\n## Two additional LongBench tasks (2026-10-06)\n\n'+report)
print(report)
