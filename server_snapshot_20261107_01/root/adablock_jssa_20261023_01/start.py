import json,hashlib,subprocess,os
import time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'status.json').exists():
    assert json.loads((ROOT/'status.json').read_text())['stage']=='failed'
    archive=ROOT/'attempts'/str(int(time.time()));archive.mkdir(parents=True)
    for pattern in ['status.json','protocol.json','launch.log','preflight.log','smoke_*.log','failure_*.json']:
        for p in ROOT.glob(pattern):shutil.move(str(p),str(archive/p.name))
manifest={}
for p in (ROOT/'kernel_adapter').glob('triton_jssa_*.py'):
    src=ROOT/'sources/vllm/vllm/v1/attention/ops'/p.name
    s=src.read_text();expected=s.replace('from vllm.triton_utils import tl, triton','import triton\nimport triton.language as tl')
    expected=expected.replace('from vllm.v1.worker.workspace import (\n    current_workspace_manager,\n    is_workspace_manager_initialized,\n)', 'def is_workspace_manager_initialized():\n    return False')
    assert p.read_text()==expected,p.name
    manifest[p.name]=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        change='Import adapters only; gather unit-test reference uses official allocation fallback. No kernel math modified.')
(ROOT/'kernel_adapter_manifest.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'compatibility.json').write_text(json.dumps(dict(
    official_commit='34ce5abd82453b16ca6d5f16a1a8ae50fac5b920',vllm_commit='39910f2b25aacc09f5e7f166cdf0030b19f8b9e8',
    hardware='RTX3090 SM86',torch='2.5.1+cu121',transformers='4.55',triton='3.7.0 isolated PYTHONPATH',
    native_vllm='Not installed: remote package index connections stall. Attempt canceled; installation logs retained.',
    fp8='Official float8_e4m3fn cache kernel unsupported on SM86, including Triton3.7.0. BF16 official configuration used.',
    indexed_attention='Official unmodified kernel passed G=1 and G=4 reference tests on Triton3.7.0.',
    radix_topk='Native extension unavailable; official torch.topk fallback used.',
    scope='Algorithm reproduction in common HF harness, not native vLLM systems speed reproduction.'),indent=2))
with (ROOT/'launch.log').open('w') as log:
    p=subprocess.Popen(['/root/miniconda3/envs/ada/bin/python','-u',str(ROOT/'evaluate.py'),'launch'],
        env=dict(os.environ,PYTHONPATH=str(ROOT/'deps')),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'launcher_pid.json').write_text(json.dumps({'pid':p.pid}))
print(p.pid)
