import json,collections,bisect,gc
from pathlib import Path
SOURCE=Path('/root/adablock_decode_profile_20261022_03')
OUT=Path(__file__).resolve().parent
result=[]
for length in [8192,16384,32768]:
    events=json.loads((SOURCE/f'trace_{length}_B.json').read_text())['traceEvents']
    cpu=sorted([e for e in events if e.get('ph')=='X' and e.get('cat') in ['cpu_op','user_annotation','cuda_runtime']],key=lambda e:(e['ts'],-e['dur']))
    stacks={};mapping={};linear_counts={};rank_calls=0;root_counts=collections.Counter()
    for ev in cpu:
        stack=stacks.setdefault((ev['pid'],ev['tid']),[])
        while stack and stack[-1]['ts']+stack[-1]['dur']<=ev['ts']+0.0001:stack.pop()
        if ev['name']=='stage::ranker':
            ev['_rank']=rank_calls;linear_counts[rank_calls]=0;rank_calls+=1
        rank=next((a for a in stack if a['name']=='stage::ranker'),None)
        if rank is not None:
            rid=rank['_rank']
            ancestors=stack[stack.index(rank)+1:]
            root=next((a for a in ancestors if a.get('cat')=='cpu_op'),None)
            if ev.get('cat')=='cpu_op' and root is None:
                root_counts[ev['name']]+=1
                label=ev['name']
                if label=='aten::linear':
                    order=linear_counts[rid];linear_counts[rid]+=1
                    label=['query_projection','mean_projection'][order] if order<2 else ['interaction_linear_1','interaction_linear_2','output_linear'][(order-2)%3]
                ev['_label']=label
            if ev.get('cat')=='cuda_runtime' and 'correlation' in ev.get('args',{}):
                mapping[ev['args']['correlation']]=root.get('_label',root['name']) if root else 'unattributed_runtime'
        stack.append(ev)
    sums=collections.Counter();counts=collections.Counter();kernels=collections.Counter()
    for ev in events:
        if ev.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        label=mapping.get(ev.get('args',{}).get('correlation'))
        if label is not None:
            sums[label]+=ev['dur'];counts[label]+=1;kernels[ev['name']]+=ev['dur']
    total=sum(sums.values())
    assert total>0 and rank_calls==512
    case=next(c for c in json.loads((SOURCE/'summary.json').read_text())['cases'] if c['length']==length and c['task']=='niah_single_1' and c['branch']=='B')
    expected=case['stages']['ranker']['gpu_ms_per_token']*16000
    assert abs(total-expected)/expected<.02,(length,total,expected)
    row=dict(length=length,rank_calls=rank_calls,total_gpu_ms_per_token=total/16000,expected_gpu_ms_per_token=expected/16000,
        groups=[dict(name=k,gpu_ms_per_token=v/16000,percent=100*v/total,kernel_calls=counts[k]) for k,v in sums.most_common()],
        root_operator_counts=dict(root_counts),linear_calls_per_ranker=sorted(set(linear_counts.values())),
        top_kernels=[dict(name=k,gpu_ms_per_token=v/16000,percent=100*v/total) for k,v in kernels.most_common(12)])
    result.append(row)
    (OUT/'ranker_breakdown.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in row.items() if k not in ['root_operator_counts','top_kernels']}),flush=True)
    del events,cpu,stacks,mapping;gc.collect()
