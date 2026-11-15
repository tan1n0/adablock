import json
from pathlib import Path
import subprocess
import sys

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
sys.path.insert(0,str(root))
from rankloss_diagnostic.data import read_json,sha256
out=root/'results/full_validation_gpu_v1'
s=read_json(out/'summary.json');integrity=read_json(out/'integrity_audit.json')
precision=read_json(root/'results/gpu_pilot_v1/precision_border_audit.json')
launch=read_json(root/'launch.json')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
assert not (Path('/proc')/str(launch['pid'])).exists()
assert not gpu
quality={(r['weighting'],r['method']):r for r in s['quality'] if r['group']=='overall' and r['target_kind']=='absolute' and r['absolute_target']==.95}
savings={}
for weighting in ('scene','document_equal'):
    rank=quality[(weighting,'RankLoss')]['required_k']
    savings[weighting]={baseline:1-rank/quality[(weighting,baseline)]['required_k'] for baseline in ('COBS','Quest')}
terminal={'status':'completed_and_verified','analysis_split':'validation_used_for_checkpoint_selection',
          'pid':launch['pid'],'pid_present':False,'gpu_compute_processes':[],
          'exit_code':None,'exit_code_note':'Detached launcher did not retain numeric exit code; complete sentinel/log, absent PID and successful integrity audit confirm terminal completion.',
          'scenes':s['scenes'],'shards':s['shards'],'documents':s['dataset_documents'],
          'run_seconds':s['wall_seconds'],'integrity_seconds':integrity['verification_seconds'],
          'tests_passed':17,'checkpoint_sha256':s['checkpoint_sha256'],'manifest_sha256':s['manifest_sha256'],
          'integrity_audit_sha256':sha256(out/'integrity_audit.json'),
          'FP32_precision_audit_sha256':sha256(root/'results/gpu_pilot_v1/precision_border_audit.json'),
          'TF32_enabled':precision['TF32_matmul'] or precision['TF32_cudnn'],
          'prefix_tolerance_unchanged':precision['prefix_tolerance_unchanged'],
          'quality95_middle_budget_saving_fraction':savings,
          'output_error':'unavailable_no_attention_and_V','confidence_intervals':'not_computed; scenes are not independent documents',
          'per_head_coverage32_scene':{r['group']:r['coverage'] for r in s['coverage'] if r['group'].startswith('head/') and r['method']=='RankLoss' and r['budget']==32 and r['weighting']=='scene'}}
with (out/'FINAL_STATE.json').open('x') as stream:json.dump(terminal,stream,indent=2)
with (out/'REPORT.md').open('a',encoding='utf-8') as stream:
    stream.write('\n\n## Terminal and strict FP32 border check\n\n')
    stream.write('PID1316 no longer exists; no GPU compute processes remain. FINAL_STATE.json records completion evidence; the detached launcher did not retain a numeric exit code.\n')
    stream.write('TF32 and autocast were disabled. Strict FP32 CPU-single/CPU-batch/GPU-single/GPU-batch recomputation reproduced the original GPU coverage exactly.\n')
    stream.write('At the border: CPU score(block64)-score(block31)=3.0517578125e-5; GPU-batch gap=-6.103515625e-5, changing which block ranks64th. Teacher mass(block31)-mass(block64)=6.956560537219048e-5.\n')
    stream.write('This is near-equal FP32 score ordering, not altered teacher or metric. Prefix hit tolerance remains1e-7. Full proof is ../gpu_pilot_v1/precision_border_audit.json and NUMERICAL_AUDIT.md.\n')
    stream.write('COBS refers to the actual repository cobs_legacy_mean_logits baseline (legacy mean raw GQA scores).\n')
    stream.write('No confidence interval is newly computed or claimed; 16 packed documents are the document units.\n')
    stream.write('At95% actual teacher coverage, mean middle budget savings vs COBS: '+json.dumps(savings)+'. These are descriptive selection-validation results, not generalization or latency claims.\n')
print(json.dumps(terminal))
