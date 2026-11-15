import os,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
p=Path('/proc/2565/stat')
while p.exists():
 try:
  if p.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(10)
assert (r/'worker_3.log').read_text().rstrip().endswith('COMPLETE')
with (r/'localize.log').open('x') as f:
 subprocess.run(['/root/miniconda3/envs/ada/bin/python','-u',str(r/'localize.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='3'),stdout=f,stderr=subprocess.STDOUT,check=True)
