import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=Path(__file__).resolve().parent
while True:
    if (p/'driver.log').exists() and 'Traceback' in (p/'driver.log').read_text():raise RuntimeError('Driver failed')
    state=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
    if state.get('stage')=='failed':raise RuntimeError(state)
    if state.get('stage')=='evaluations_complete':break
    time.sleep(10)
(p/'status.json').write_text(json.dumps(dict(stage='independent_qa_controls')))
(p/'qa_controls_frozen.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'qa_fixed.py',p/'summarize.py']},indent=2))
jobs=[];handles=[]
for i in range(4):
    out=(p/f'qa_fixed_{i}.log').open('x');handles.append(out)
    jobs.append(subprocess.Popen([sys.executable,'-u',str(p/'qa_fixed.py'),str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=out,stderr=subprocess.STDOUT))
codes=[j.wait() for j in jobs]
for h in handles:h.close()
assert not any(codes),codes
with (p/'summarize.log').open('x') as out:subprocess.run([sys.executable,str(p/'summarize.py')],stdout=out,stderr=subprocess.STDOUT,check=True)
(p/'status.json').write_text(json.dumps(dict(stage='complete',training_questions=48,validation_questions=12,test_questions=12,hotpot_regression=50)))
print('COMPLETE',flush=True)
