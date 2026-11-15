"""Isolated, deferred profiling; source evaluation is read-only."""
import os, sys, time, json, statistics, subprocess, gc
from pathlib import Path
from contextlib import nullcontext

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_autonomous_20261022_01')
ACTIVE=False
OWN_GPU_PID=None

def dump(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(value,indent=2));t.replace(p)

def scope(name):
    return torch.profiler.record_function('stage::'+name) if ACTIVE else nullcontext()

def wrapped(fn,name):
    def call(*args,**kwargs):
        with scope(name):return fn(*args,**kwargs)
    return call

def selected_attention(query,keys,values,indices,valid,scale):
    # Same operations and dispatch as the evaluated implementation.
    with scope('kv_gather_copy'):
        heads,width=indices.shape
        index=indices[...,None].expand(heads,width,keys.shape[-1])
        k=keys[0].gather(1,index)[None]
        v=values[0].gather(1,index)[None]
        group=query.shape[1]//heads
    with scope('mask_dispatch_sync'):
        unmasked=bool(valid.all())
    if unmasked:
        with scope('kv_gather_copy'):
            q,k,v=query.contiguous(),k.contiguous(),v.contiguous()
        with scope('attention'):
            return torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,scale=scale,enable_gqa=True)
    with scope('kv_gather_copy'):
        k,v=k.repeat_interleave(group,1),v.repeat_interleave(group,1)
        mask=valid.repeat_interleave(group,0)[None,:,None,:]
        q,k,v=query.contiguous(),k.contiguous(),v.contiguous()
    with scope('attention'):
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale)

def other_gpu_processes():
    text=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(x.strip()) for x in text.splitlines() if x.strip().isdigit() and int(x.strip())!=OWN_GPU_PID]

def await_ready():
    while True:
        status=json.loads((SOURCE/'status.json').read_text())
        if status['stage']=='failed':raise RuntimeError('Autonomous evaluation failed; profiling not started')
        if status['stage']=='complete' and not other_gpu_processes():return
        dump('status.json',dict(stage='waiting',evaluation_stage=status['stage'],waiting_for='evaluation completion and GPU 0 exclusively idle'))
        time.sleep(30)

def report():
    rows=[json.loads(p.read_text()) for p in ROOT.glob('case_*.json')]
    dump('summary.json',dict(completed_cases=len(rows),expected_cases=27,cases=rows))
    lines=['Decode stage profiling','GPU percentages are summed kernel duration shares, NOT wall-clock shares.',
           'CPU range durations overlap GPU work and cannot be added to GPU times.',
           'Padding overhead is embedded in KV/attention; no causal padding-only timing is claimed.','']
    for r in rows:
        lines.append(f"{r['task']} {r['length']} {r['branch']}: {r['wall_ms_per_token_median']:.3f} ms/token; profiler slowdown {r['profiler_slowdown']:.2f}x")
        for k,v in r['stages'].items():
            lines.append(f"  {k}: GPU {v['gpu_ms_per_token']:.3f} ms/token ({v['gpu_percent']:.2f}%); CPU range {v['cpu_ms_per_token']:.3f} ms/token")
        lines.append(f"  other GPU: {r['other_gpu_percent']:.2f}%; logical/padded slots {r['logical_fraction']:.4f}/{r['padded_fraction']:.4f}")
    (ROOT/'report.txt').write_text('\n'.join(lines))

def run():
    global torch,e,ACTIVE,OWN_GPU_PID
    await_ready()
    import torch
    import experiment as e
    torch.set_num_threads(4)
    # NVML reports host PIDs; os.getpid() is a container PID on this server.
    # Establish our sole CUDA context while the device is verified empty.
    assert not other_gpu_processes()
    probe=torch.zeros(1,device='cuda:0')
    gpu_pids=other_gpu_processes()
    assert len(gpu_pids)==1,gpu_pids
    OWN_GPU_PID=gpu_pids[0]
    dump('gpu_identity.json',dict(container_pid=os.getpid(),nvml_host_pid=OWN_GPU_PID))
    del probe
    protected=list(SOURCE.glob('*.py'))+[SOURCE/'selected.json',SOURCE/'comparison.json',SOURCE/'status.json']
    protected += list(SOURCE.glob('partial_*.json'))
    protected += [e.OLD/'branches'/n/'best_pair.pt' for n in ['S','A','B']]
    hashes={str(p):e.c.sha(p) for p in protected}
    allrows=json.loads((SOURCE/'selected.json').read_text())
    chosen=[]
    for length in [8192,16384,32768]:
        for task in ['niah_single_1','niah_multivalue','vt']:
            chosen.append(next(r for r in allrows if r['length']==length and r['task']==task))
    dump('selected.json',chosen)
    dump('protocol.json',dict(gpu=0,n_prompts=9,n_cases=27,decode_steps=16,warmups=1,wall_repeats=3,
        profiler_repeats=1,autonomous_budget=True,quantile=.95,summary='FP32 Gram without SVD fallback',
        trajectory='Per-branch deterministic greedy continuation; ignore EOS for fixed 16-step windows, not quality evaluation',
        timing='CUDA kernel duration attribution from torch.profiler; uninstrumented wall time reported separately',
        protected_sha256=hashes,source_sha256={p.name:e.c.sha(p) for p in ROOT.glob('*.py')}))
    dump('status.json',dict(stage='running',pid=os.getpid(),gpu=0))
    model,tok,models,eos=e.load()
    e.c.run.AdaptiveLlama=e.gpu.Controller;e.c.run.DynamicCache=e.BufferCache
    e.baseline.compact_digest=e.fast_digest.compact_digest
    original_attention=e.baseline.selected_attention
    original_rank=e.baseline.forward_packed
    original_features=e.gpu.features_gpu
    original_select=e.gpu.selected_gpu
    original_budget={n:b.forward for n,(_,b) in models.items()}

    def instrument(enabled):
        e.baseline.selected_attention=selected_attention if enabled else original_attention
        e.baseline.forward_packed=wrapped(original_rank,'ranker') if enabled else original_rank
        e.gpu.features_gpu=wrapped(original_features,'budget_features') if enabled else original_features
        e.gpu.selected_gpu=wrapped(original_select,'block_selection') if enabled else original_select
        for n,(_,b) in models.items():b.forward=wrapped(original_budget[n],'budget_network') if enabled else original_budget[n]

    @torch.inference_mode()
    def decode(p,l,d,rank,budget):
        return e.c.run.decode(model,p,l,'adaptive',rank,budget,d,eos,max_new=17,ignore_eos=True,capture=True,quantile=.95)

    for ix,row in enumerate(chosen):
        if other_gpu_processes():raise RuntimeError('GPU exclusivity lost')
        with torch.inference_mode():
            p,l,_=e.prefill(model,row['input_ids']);d,_=e.build(p,'gpu_digest')
        for branch,(rank,budget) in models.items():
            name=f'case_{ix}_{branch}.json'
            if (ROOT/name).exists():continue
            if other_gpu_processes():raise RuntimeError('GPU exclusivity lost')
            instrument(False);ACTIVE=False
            warm=decode(p,l,d,rank,budget);expected=warm['ids'];del warm
            wall=[]
            for repeat in range(3):
                result=decode(p,l,d,rank,budget)
                assert result['ids']==expected
                wall.append(result['decode_seconds']*1000/16)
                del result
            instrument(True);ACTIVE=True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as prof:
                result=decode(p,l,d,rank,budget)
            ACTIVE=False
            assert result['ids']==expected,'Instrumentation changed generation'
            assert result['decode_steps']==16
            if other_gpu_processes():raise RuntimeError('GPU exclusivity lost; discard contaminated case')
            events=prof.events()
            # CUDA user-annotation intervals include gaps and overlap kernels.
            # Exclude them from the denominator and use CPU ranges for attribution.
            total_gpu=sum(v.device_time_total for v in events if v.device_type==torch.autograd.DeviceType.CUDA and not v.name.startswith('stage::') and not v.is_user_annotation)
            stages={}
            for v in prof.key_averages():
                if v.key.startswith('stage::') and v.device_type==torch.autograd.DeviceType.CPU:
                    stages[v.key[7:]]=dict(gpu_ms_per_token=v.device_time_total/16000,
                        gpu_percent=100*v.device_time_total/total_gpu,cpu_ms_per_token=v.cpu_time_total/16000,calls=v.count)
            attributed=sum(v['gpu_percent'] for v in stages.values())
            assert 0<attributed<=100.1,(attributed,total_gpu)
            obj=result.pop('_controller');denom=sum(c['full'] for c in obj.costs)
            record=dict(task=row['task'],length=row['length'],branch=branch,decode_steps=16,
                wall_ms_per_token=wall,wall_ms_per_token_median=statistics.median(wall),
                profiler_slowdown=result['decode_seconds']*1000/16/statistics.median(wall),
                total_gpu_ms_per_token=total_gpu/16000,stages=stages,other_gpu_percent=max(0,100-attributed),
                logical_fraction=float(sum(c['read'] for c in obj.costs))/denom,
                padded_fraction=float(sum(c['padded'] for c in obj.costs))/denom,
                output_parity=True,gpu_exclusive_at_start_and_end=True)
            # One full trace per context length; all cases retain operator aggregates.
            if row['task']=='niah_single_1' and branch=='B':prof.export_chrome_trace(str(ROOT/f'trace_{row["length"]}_B.json'))
            ops=[dict(name=v.key,calls=v.count,cpu_self_us=v.self_cpu_time_total,gpu_self_us=v.self_device_time_total) for v in prof.key_averages()]
            dump(f'operators_{ix}_{branch}.json',ops);dump(name,record);report()
            print(json.dumps(record),flush=True)
            del result,obj,prof,events
        del p,l,d;gc.collect();torch.cuda.empty_cache()
    assert all(e.c.sha(Path(p))==h for p,h in hashes.items()),'Protected file changed'
    dump('status.json',dict(stage='complete',cases=27,old_results_unchanged=True))

if __name__=='__main__':
    try:run()
    except Exception as exc:
        dump('status.json',dict(stage='failed',error=str(exc)));raise
