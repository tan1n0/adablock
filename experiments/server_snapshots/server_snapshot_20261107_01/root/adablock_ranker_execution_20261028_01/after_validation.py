import time,json,subprocess,os
from pathlib import Path
r=Path('/root/adablock_ranker_execution_20261028_01')
pid=json.loads((r/'decode_v2_pid.json').read_text())['pid']
while Path(f'/proc/{pid}').exists():time.sleep(10)
s=json.loads((r/'status.json').read_text())
assert s['stage']=='complete',s
subprocess.run(['/root/miniconda3/envs/ada/bin/python','-u',str(r/'profile_decode_stages.py')],check=True)
