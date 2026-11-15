import json, random, sys, time
from pathlib import Path
sys.path.insert(0, '/root/adablock_sab90_20261019_01')
import core
import baseline_controller as base
import gpu_controller as gpu
import fast_digest
import torch

torch.set_num_threads(4)
torch.manual_seed(20261022)
rng=random.Random(20261022)
report={}
errors=[]
for n in [1,2,4,8,16,32,128,479,991,2015]:
    scores=torch.randn(8,n,device='cuda')
    query=torch.randn(8,4,128,device='cuda')
    got,order=gpu.features_gpu(query,scores,32760,17)
    expected=torch.stack([base.online_features(dict(query=query[h].cpu(),token_position=32760,
                 layer_id=17,kv_head_id=h),scores[h].cpu()) for h in range(8)])
    errors.append(float((got.cpu()-expected).abs().max()))
    assert torch.allclose(got.cpu(),expected,atol=3e-6,rtol=3e-6),errors
report['feature_max_abs']=max(errors)
for i in range(1000):
    cap=rng.randrange(0,2049); target=rng.randrange(8*cap+1)
    raw=[rng.randrange(cap+1) for _ in range(8)]
    expected=base.allocate(raw,target,cap)
    got=gpu.allocate_gpu(torch.tensor(raw,device='cuda'),target,cap).tolist()
    assert got==expected,(raw,target,cap,got,expected)
report['allocation_cases']=1000
for position in [20,543,544,545,8191,8192,32767]:
    hist=(position+15)//16
    sink,local,middle=base.fixed_and_middle_blocks(hist,2,32,hist-1)
    fixed=sink|local;ids=torch.tensor(sorted(middle),device='cuda',dtype=torch.long);n=len(ids)
    order=torch.rand(8,n,device='cuda').argsort(dim=-1,descending=True,stable=True)
    takes=torch.tensor([rng.randrange(n+1) for _ in range(8)],device='cuda')
    ix,valid,lengths=gpu.selected_gpu(ids,order,takes,fixed,position)
    for h in range(8):
        blocks=fixed|set(ids[order[h,:int(takes[h])]].tolist())
        expected=[t for b in sorted(blocks) for t in range(b*16,min((b+1)*16,position))]+[position]
        assert ix[h][valid[h]].tolist()==expected
report['selection_cases']=7
keys=torch.randn(8,4096,128,device='cuda',dtype=torch.bfloat16)
baseline=base.compact_digest(keys)
for method in ['svd','gram']:
    torch.cuda.synchronize();start=time.perf_counter()
    d=fast_digest.compact_digest(keys,method=method);torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    q=torch.randn(8,256,4,128,device='cuda')
    def quadratic(d):return ((q@d['basis'].transpose(-1,-2)).square()*d['eigenvalues'][:,:,None]).sum(-1)
    x,y=quadratic(baseline),quadratic(d)
    err=float((x-y).abs().max());rel=float((x-y).norm()/x.norm())
    assert rel<1e-4,(method,err,rel)
    report[method]=dict(seconds=seconds,quadratic_max_abs=err,quadratic_relative_norm=rel,
                        mean_max_abs=float((d['mean']-baseline['mean']).abs().max()))
Path(__file__).with_name('ops_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
