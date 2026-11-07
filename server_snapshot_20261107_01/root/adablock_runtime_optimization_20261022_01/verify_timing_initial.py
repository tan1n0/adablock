"""Production-length decision parity and interleaved fixed-trajectory timing."""
import experiment as e
import json, sys, time, statistics
import torch

@torch.inference_mode()
def timed(model,p,d,rank,budget,version,tokens):
    cls=e.baseline.Controller if version=='baseline' else e.gpu.Controller
    cls.ratio=.25
    e.baseline.compact_digest=e.fast_digest.compact_digest if version=='gpu_digest' else e.ORIGINAL_DIGEST
    controller=cls(model,p,d,rank,budget,'adaptive',quantile=.95,capture=False)
    controller.install();torch.cuda.synchronize();start=time.perf_counter()
    try:
        for step,token in enumerate(tokens):
            position=p[0][0].shape[2]+step
            controller.begin_step()
            out=model.model(input_ids=torch.tensor([[token]],device='cuda'),
                position_ids=torch.tensor([[position]],device='cuda'),cache_position=torch.tensor([position],device='cuda'),use_cache=False)
            controller.finish_step()
            int(model.lm_head(out.last_hidden_state[:,-1:])[0,-1].float().argmax())
        torch.cuda.synchronize();seconds=time.perf_counter()-start
    finally:
        controller.uninstall();e.baseline.compact_digest=e.ORIGINAL_DIGEST
    return seconds

@torch.inference_mode()
def main(length):
    model,tok,models,eos=e.load()
    rows=json.loads((e.ROOT/'selected.json').read_text())
    row=next(r for r in rows if r['length']==length and r['task']=='cwe')
    p,l,prefill=e.prefill(model,row['input_ids'])
    d,original_seconds=e.build(p,'baseline')
    fast,fast_seconds=e.build(p,'gpu_digest')
    report=dict(length=length,prompt_tokens=len(row['input_ids']),index_seconds=dict(baseline=original_seconds,gpu_digest=fast_seconds),decisions={},timing={})
    report['digest_numerics']={}
    torch.manual_seed(20261022)
    for layer in [0,15,31]:
        a,b=d[layer],fast[layer]
        q=torch.randn(8,a['mean'].shape[1],4,128,device='cuda')
        def quadratic(x):return ((q@x['basis'].transpose(-1,-2)).square()*x['eigenvalues'][:,:,None]).sum(-1)
        x,y=quadratic(a),quadratic(b)
        relative=float((x-y).norm()/x.norm().clamp_min(1e-15))
        report['digest_numerics'][layer]=dict(mean_max_abs=float((a['mean']-b['mean']).abs().max()),
            quadratic_max_abs=float((x-y).abs().max()),quadratic_relative_norm=relative)
        assert relative<1e-3,report['digest_numerics']
        del q,x,y
    for name,(rank,budget) in models.items():
        for ratio in [.1,.25]:
            original=e.decode(model,p,l,d,rank,budget,eos,'baseline',ratio,2,True,True)
            chosen=e.indices(original['_controller'])
            for version in ['gpu','gpu_digest']:
                result=e.decode(model,p,l,fast if version=='gpu_digest' else d,rank,budget,eos,version,ratio,2,True,True)
                selected=e.indices(result['_controller'])
                equal=sum(a==b for a,b in zip(chosen,selected))
                classes=[r['classes'].tolist() for r in result['_controller'].selected]
                old_classes=original['steps'][0]['budget_classes']
                same_classes=old_classes==[x for r in classes for x in r]
                report['decisions'][f'{name}/{ratio}/{version}']=dict(equal_layers=equal,total_layers=len(chosen),classes_equal=same_classes,
                    next_token_equal=original['ids']==result['ids'])
                if version=='gpu':assert equal==len(chosen) and same_classes,(length,name,ratio,report['decisions'])
            del original,result
    e.dump(f'verification_{length}_partial.json',report)
    e.c.run.DynamicCache=e.BufferCache
    full=e.c.run.decode(model,p,l,'full',None,None,None,eos,max_new=17,ignore_eos=True)
    tokens=full['ids'][:16]
    rank,budget=models['B']
    versions=['baseline','gpu','gpu_digest'];times={v:[] for v in versions}
    for v in versions:timed(model,p,fast if v=='gpu_digest' else d,rank,budget,v,tokens[:2])
    for repeat in range(3):
        order=versions[repeat:]+versions[:repeat]
        for v in order:
            seconds=timed(model,p,fast if v=='gpu_digest' else d,rank,budget,v,tokens)
            times[v].append(seconds)
    report['timing']={v:dict(seconds=values,median_seconds=statistics.median(values),
                            median_ms_per_step=statistics.median(values)/len(tokens)*1000) for v,values in times.items()}
    report['timing_protocol']=dict(branch='B',ratio=.25,teacher_forced=True,steps=16,repeats=3,order_rotated=True,capture=False)
    e.dump(f'verification_{length}.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main(int(sys.argv[1]))
