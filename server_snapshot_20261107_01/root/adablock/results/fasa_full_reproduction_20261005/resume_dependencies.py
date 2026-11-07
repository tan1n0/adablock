import os,sys,json,subprocess,hashlib,time,urllib.request
from pathlib import Path
ROOT=Path("/root/adablock");W=ROOT/"results/fasa_full_reproduction_20261005";V=ROOT/".venvs/fasa_full_20261005"
PY="/root/miniconda3/envs/ada/bin/python"
def write(n,x):(W/n).write_text(json.dumps(x,indent=2))
def run(argv,env=None):
 print("RUN",argv,flush=True);subprocess.run(argv,env=env,check=True)
start=time.perf_counter()
before=(W/"ada_packages_before.json").read_text()
assert V.exists()
vp=str(V/"bin/python")
env={k:v for k,v in os.environ.items() if not k.lower().endswith("_proxy") and not k.startswith("PIP_")}
env.update(PIP_CONFIG_FILE="/dev/null",PIP_DISABLE_PIP_VERSION_CHECK="1",PYTHONNOUSERSITE="1")
url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
d=W/"dependency_downloads";d.mkdir(exist_ok=True)
p=d/"flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
write("dependency_status.json",dict(stage="downloading_fa2_direct",expected_bytes=187797312,venv=str(V)))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
part=p.with_suffix(".part");total=part.stat().st_size if part.exists() else 0;last=total
request=urllib.request.Request(url,headers={"Range":f"bytes={total}-"}) if total else urllib.request.Request(url)
with opener.open(request,timeout=60) as response:
 if total:assert response.status==206,(response.status,"Expected partial response")
 with part.open("ab" if total else "wb") as out:
  while True:
   b=response.read(1024*1024)
   if not b:break
   out.write(b);total+=len(b)
   if total-last>=16*1024*1024:print("FA2_BYTES",total,flush=True);last=total
h=hashlib.sha256(part.read_bytes())
assert total==187797312,total
p.with_suffix(".part").replace(p)
write("fa2_wheel_provenance.json",dict(url=url,bytes=total,sha256=h.hexdigest(),network="server direct; ProxyHandler({})"))
run([vp,"-m","pip","install","--no-deps","--report",str(W/"fa2_install_report.json"),str(p)],env)
code="import torch,transformers,flash_attn,flash_attn_2_cuda,tokenizers,datasets,jieba,rouge,fuzzywuzzy,json;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,flash_attn=flash_attn.__version__,tokenizers=tokenizers.__version__,datasets=datasets.__version__,torch_path=torch.__file__,cuda_available=torch.cuda.is_available(),cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI)))"
audit=json.loads(subprocess.check_output([vp,"-c",code],env=env,text=True))
assert audit["torch"]=="2.5.1+cu121" and audit["transformers"]=="4.43.0" and audit["flash_attn"]=="2.7.4.post1"
after=subprocess.check_output([PY,"-m","pip","list","--format=json"],text=True)
(W/"ada_packages_after.json").write_text(after);assert json.loads(before)==json.loads(after),"ada packages changed"
audit.update(ada_unchanged=True,venv=str(V),elapsed_seconds=time.perf_counter()-start)
write("dependency_audit.json",audit);write("dependency_status.json",dict(stage="completed",audit=audit))
print("DEPENDENCIES_COMPLETE",json.dumps(audit),flush=True)
