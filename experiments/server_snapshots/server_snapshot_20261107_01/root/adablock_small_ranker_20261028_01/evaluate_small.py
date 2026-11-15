import os,sys,json,time,gc,statistics,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_autonomous_20261022_01')
sys.path.insert(0,str(SOURCE))
import experiment as e
import torch
NAMES=['original_S','original_B','H64_S','H64_B','P32H64_S','P32H64_B']

def dump(path,value):
    p=ROOT/path;t=p.with_suffix('.tmp');t.write_text(json.dumps(e.clean(value),indent=2));t.replace(p)
def clear():gc.collect();torch.cuda.empty_cache()
def setup():
    model,tok,old,eos=e.load();models={'original_'+n:old[n] for n in ['S','B']}
    for arch in ['H64','P32H64']:
        for branch in ['S','B']:
            models[arch+'_'+branch]=e.c.load_pair(ROOT/arch/'branches'/branch/'best_pair.pt','cuda')[:2]
    e.c.run.DynamicCache=e.BufferCache;e.c.run.AdaptiveLlama=e.gpu.Controller
    e.baseline.compact_digest=e.fast_digest.compact_digest
    return model,tok,models,eos

@torch.inference_mode()
def preflight():
    model,tok,models,eos=setup();rows=json.loads((ROOT/'selected.json').read_text())
    p,l,_=e.prefill(model,rows[0]['input_ids'][:1024]);d,_=e.build(p,'gpu_digest')
    report={'models':{}}
    for name,(rank,budget) in models.items():
        for mode in ['adaptive','fixed32','fixed128']:
            out=e.c.run.decode(model,p,l,mode,rank,budget,d,eos,max_new=3,ignore_eos=True,capture=True,quantile=.95)
            obj=out.pop('_controller')
            if mode=='adaptive':
                for row in obj.step_rows:
                    n=row['candidate_blocks'];choices=torch.tensor([min(n,k) for k in e.baseline.BUDGETS[:-1]]+[n],device='cuda')
                    assert torch.equal(row['middle_blocks'],choices[row['classes']])
            else:
                k=int(mode[5:])
                assert all(bool((row['middle_blocks']==min(k,row['candidate_blocks'])).all()) for row in obj.step_rows)
            del out,obj
        report['models'][name]={'parameters':sum(p.numel() for p in rank.parameters()),'projection_size':rank.config.projection_size,'hidden_size':rank.config.hidden_size}
    del p,l,d;e.reset_registry();clear()
    row=max(rows,key=lambda r:len(r['input_ids']))
    p,l,_=e.prefill(model,row['input_ids'])
    full=e.c.run.decode(model,p,l,'full',None,None,None,eos,max_new=3,ignore_eos=True)
    allread=e.c.run.decode(model,p,l,'all',None,None,None,eos,max_new=3,ignore_eos=True)
    assert full['ids']==allread['ids']
    report.update(passed=True,long_full_all_equal=True,prompt_tokens=len(row['input_ids']))
    dump('preflight.json',report)

@torch.inference_mode()
def worker(shard,stage):
    model,tok,models,eos=setup();rows=json.loads((ROOT/'selected.json').read_text())
    selection=json.loads((ROOT/'selection.json').read_text())
    indices=selection['smoke'] if stage=='smoke' else list(range(len(rows)))
    folder=ROOT/stage;folder.mkdir(exist_ok=True)
    warmed=set()
    for order_index,i in enumerate(indices):
        if order_index%2!=shard:continue
        row=rows[i];path=folder/f'partial_{i}.json'
        assert not path.exists(),path
        p,l,prefill=e.prefill(model,row['input_ids']);d,build=e.build(p,'gpu_digest')
        record=dict(index=i,task=row['task'],length=row['length'],prompt_tokens=len(row['input_ids']),answers=row['answers'],methods={})
        warm=row['length'] not in warmed
        specs=[('full','full')]+[(n,'adaptive') for n in NAMES]
        if stage=='smoke' or i in selection['fixed']:
            specs += [(n,m) for m in ['fixed32','fixed128'] for n in NAMES]
        shift=i%len(specs);specs=specs[shift:]+specs[:shift]
        previous=json.loads((SOURCE/f'partial_{i}.json').read_text())
        for name,mode in specs:
            rank,budget=models[name] if name!='full' else (None,None)
            def run(cap,ignore=False):
                return e.c.run.decode(model,p,l,mode,rank,budget,d if mode!='full' else None,eos,
                    max_new=cap,ignore_eos=ignore,capture=True,quantile=.95)
            if warm:
                tmp=run(3,True);del tmp
            torch.cuda.reset_peak_memory_stats()
            result=run(row['max_new']);obj=result.pop('_controller')
            if obj:
                full_count=sum(x['full'] for x in obj.costs)
                result.update(logical_fraction=float(sum(x['read'] for x in obj.costs))/full_count if full_count else None,
                    padded_fraction=float(sum(x['padded'] for x in obj.costs))/full_count if full_count else None)
                if mode.startswith('fixed'):
                    k=int(mode[5:]);assert all(bool((r['middle_blocks']==min(k,r['candidate_blocks'])).all()) for r in obj.step_rows)
            else:result.update(logical_fraction=1.,padded_fraction=1.)
            del obj
            prediction=e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
            result.pop('steps');result.pop('audit')
            result.update(prediction=prediction,score=e.official.metric_for(row['task'])([prediction],[row['answers']]),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,prefill_seconds=prefill,index_seconds=build if name!='full' else 0)
            if name=='full' or (name.startswith('original_') and mode=='adaptive'):
                oldname='full' if name=='full' else name[-1]
                assert result['ids']==previous['methods'][oldname]['ids'],('original output changed',i,name)
            key=name if mode in ['full','adaptive'] else name+'/'+mode
            record['methods'][key]=e.clean(result);dump(str(path.relative_to(ROOT)),record)
            print(json.dumps(dict(stage=stage,index=i,method=key,score=result['score'],read=result['logical_fraction'])),flush=True)
            del result
        if stage=='smoke':
            record['fixed128_profile']={n:[] for n in NAMES}
            # Three rotating passes at equal block count isolate architecture cost.
            for repeat in range(3):
                order=NAMES[repeat:]+NAMES[:repeat]
                for name in order:
                    rank,budget=models[name]
                    torch.cuda.reset_peak_memory_stats()
                    result=e.c.run.decode(model,p,l,'fixed128',rank,budget,d,eos,max_new=17,ignore_eos=True,quantile=.95)
                    record['fixed128_profile'][name].append(dict(ms_per_token=1000*result['decode_seconds']/16,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
                    del result
            dump(str(path.relative_to(ROOT)),record)
        warmed.add(row['length']);del p,l,d,record;e.reset_registry();clear()

def summarize():
    rr=[json.loads(p.read_text()) for p in (ROOT/'evaluation').glob('partial_*.json')]
    selection=json.loads((ROOT/'selection.json').read_text())
    expected=set(['full']+NAMES)
    complete=[r for r in rr if expected<=set(r['methods']) and (r['index'] not in selection['fixed'] or len(r['methods'])==19)]
    report={'completed':len(complete),'expected':450,'groups':{},'overall':{},'fixed128_profile':{}}
    def stats(rows,name):
        vals=[r['methods'][name] for r in rows if name in r['methods']]
        if not vals:return None
        return dict(n=len(vals),score=statistics.mean(v['score'] for v in vals),
            logical_fraction=statistics.mean(v['logical_fraction'] for v in vals if v['logical_fraction'] is not None),
            padded_fraction=statistics.mean(v['padded_fraction'] for v in vals if v['padded_fraction'] is not None),
            ms_per_token=1000*sum(v['decode_seconds'] for v in vals)/max(1,sum(v['decode_steps'] for v in vals)),
            mean_decode_seconds=statistics.mean(v['decode_seconds'] for v in vals),cap_hits=sum(v['hit_generation_limit'] for v in vals))
    keys=['full']+NAMES+[n+'/'+m for m in ['fixed32','fixed128'] for n in NAMES]
    report['overall']={n:stats(complete,n) for n in keys}
    for task,length in sorted({(r['task'],r['length']) for r in complete}):
        group=[r for r in complete if (r['task'],r['length'])==(task,length)]
        report['groups'][f'{task}_{length}']={n:stats(group,n) for n in keys}
    smoke=[json.loads(p.read_text()) for p in (ROOT/'smoke').glob('partial_*.json')]
    for length in [8192,16384,32768]:
        group=[r for r in smoke if r['length']==length and 'fixed128_profile' in r]
        if group:
            report['fixed128_profile'][length]={n:dict(ms_per_token=statistics.mean(statistics.median(x['ms_per_token'] for x in r['fixed128_profile'][n]) for r in group),
                peak_allocated_MiB=statistics.mean(statistics.median(x['peak_allocated_MiB'] for x in r['fixed128_profile'][n]) for r in group)) for n in NAMES}
    report['paired']={}
    for name in NAMES[2:]:
        ref='original_'+name[-1];diff=[r['methods'][name]['score']-r['methods'][ref]['score'] for r in complete]
        report['paired'][name]=dict(wins=sum(x>0 for x in diff),losses=sum(x<0 for x in diff),ties=sum(x==0 for x in diff))
    dump('comparison.json',report)
    if len(complete)==450:
        lines=['小排序器实验完成汇总','',
            '自主预算：450 道配对题；固定预算：其中分层抽取的 90 道题，每任务每长度 10 道。',
            'original=213→128→128，H64=213→64→64，P32H64=117→64→64。每种结构分别比较 S/B。',
            'fixed32/fixed128 是每 KV 头的中间候选块数，另加相同 sink/local/current；表中的读取量包含这些固定部分。',
            '解码计时不含预填充、摘要构建；mean_proj 缓存保持关闭。', '',
            '方法 | 自主预算得分 | 逻辑 KV 读取% | 补齐后% | 解码 ms/token']
        for name in ['full']+NAMES:
            v=report['overall'][name];lines.append(f"{name} | {v['score']:.4f} | {100*v['logical_fraction']:.3f} | {100*v['padded_fraction']:.3f} | {v['ms_per_token']:.3f}")
        lines+=['','方法 | 固定32块得分 | 固定128块得分 | 相对对应原版的自主预算得分变化']
        for name in NAMES:
            a=report['overall'][name+'/fixed32'];b=report['overall'][name+'/fixed128']
            delta=report['overall'][name]['score']-report['overall']['original_'+name[-1]]['score']
            lines.append(f"{name} | {a['score']:.4f} | {b['score']:.4f} | {delta:+.4f}")
        lines+=['','等读取预算的解码计时：固定128块、16步、3次重复，每个任务长度一道题。',
            '长度 | 方法 | ms/token | 相对对应原版加速倍数 | 峰值已分配显存 MiB']
        for length,group in report['fixed128_profile'].items():
            for name,v in group.items():
                speed=group['original_'+name[-1]]['ms_per_token']/v['ms_per_token']
                lines.append(f"{length} | {name} | {v['ms_per_token']:.3f} | {speed:.3f} | {v['peak_allocated_MiB']:.1f}")
        lines+=['','自主预算的逐题比较：']
        for name,v in report['paired'].items():lines.append(f"{name}: 得分更高 {v['wins']} 题，更低 {v['losses']} 题，相同 {v['ties']} 题。")
        lines+=['','预训练和适配的数据、轮数、学习率沿用原配置，小模型从头预训练，不截取旧权重。',
            '本轮只有一个训练种子，细小得分差异不代表统计显著。各任务/长度的详细结果见 comparison.json；最终校验状态见 status.json。']
        (ROOT/'report.txt').write_text('\n'.join(lines))
    return len(complete)==450

if __name__=='__main__':
    try:
        if sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='summarize':summarize()
        else:worker(int(sys.argv[2]),sys.argv[1])
    except Exception:
        dump('failure_'+sys.argv[1]+'_'+str(os.getpid())+'.json',dict(error=traceback.format_exc()));raise
