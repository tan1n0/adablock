import json,time,subprocess,os
from pathlib import Path
r=Path('/root/adablock_ranker_execution_20261028_01')
while True:
    s=json.loads((r/'status.json').read_text())
    if s['stage'].startswith('failed'):raise RuntimeError(s)
    if s.get('stages_profiled'):break
    time.sleep(10)
marker=Path('/root/adablock_small_ranker_20261028_01/PAUSE_KEEPALIVE')
while marker.exists():time.sleep(2)
marker.write_text(str(r)+' v3 validation')
time.sleep(8)
try:
    subprocess.run(['/root/miniconda3/envs/ada/bin/python','-u',str(r/'validate_decode_v3.py')],check=True)
finally:
    if marker.exists() and str(r) in marker.read_text():marker.unlink()
