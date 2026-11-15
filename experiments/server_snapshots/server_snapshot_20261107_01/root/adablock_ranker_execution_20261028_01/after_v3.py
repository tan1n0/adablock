import json,time,subprocess
from pathlib import Path
r=Path('/root/adablock_ranker_execution_20261028_01')
while True:
    s=json.loads((r/'status.json').read_text())
    if s['stage'].startswith('failed'):raise RuntimeError(s)
    if (r/'verification_v3.json').exists() and s['stage']=='complete':break
    time.sleep(10)
marker=Path('/root/adablock_small_ranker_20261028_01/PAUSE_KEEPALIVE')
while marker.exists():time.sleep(2)
marker.write_text(str(r)+' long decode check')
time.sleep(8)
try:
    subprocess.run(['/root/miniconda3/envs/ada/bin/python','-u',str(r/'long_decode_check.py')],check=True)
    subprocess.run(['/root/miniconda3/envs/ada/bin/python',str(r/'report.py')],check=True)
finally:
    if marker.exists() and str(r) in marker.read_text():marker.unlink()
