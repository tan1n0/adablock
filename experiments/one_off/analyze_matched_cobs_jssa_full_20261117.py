"""Paired quality, timing, index, and memory analysis for matched backend run."""
from __future__ import annotations
import argparse,json,random,statistics
from pathlib import Path

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
def mean(v):return statistics.mean(v)
def boot(values,seed=20261117,n=10000):
    rng=random.Random(seed);a=list(values)
    out=sorted(sum(a[rng.randrange(len(a))] for _ in a)/len(a) for _ in range(n))
    return [out[int(.025*n)],out[int(.975*n)-1]]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--limit',type=int,default=100)
    args=parser.parse_args();root=args.root
    rows=[]
    for i in range(args.limit):
        full=json.loads((root/'full'/f'case_{i:03d}.json').read_text())
        jssa=json.loads((root/'jssa_512'/f'case_{i:03d}.json').read_text())
        cobs=json.loads((root/'cobs'/f'case_{i:03d}.json').read_text())
        assert len({full['input_sha256'],jssa['input_sha256'],cobs['input_sha256']})==1
        assert full['forced_input_ids']==jssa['forced_input_ids']
        assert full['forced_input_ids']==cobs['profile']['forced_input_ids']
        for case,method in ((full,'full'),(jssa,'jssa_512')):
            reads=case['natural'].get('reads_per_kv_head_step')
            if reads is not None:
                expected=[(case['prompt_tokens']+step+1 if method=='full' else 512+(case['prompt_tokens']+step)%16+1) for step in range(case['natural']['decode_steps'])]
                assert reads==expected,(i,method,'KV read budget mismatch')
        assert cobs['budget']==512
        converted=[]
        for name,case in [('full',full),('jssa_512',jssa)]:
            n=case['natural']
            converted.append((name,{'score':n['ruler_score'],'all_hit':n['all_hit'],
              'decode_steps':n['decode_steps'],'decode_seconds':n['decode_seconds'],
              'cap_reached':n['cap_reached'],'matches_historical_ids':n['matches_historical_ids'],
              'forced_ms_per_step':case['forced_median_ms_per_step'],
              'index_seconds':case['index_build_seconds'],
              'postprefill16_seconds':case['postprefill_forced16_seconds'],
              'prefill_seconds':case['prefill_seconds'],
              'peak_allocated_mib':case['peak_allocated_mib'],
              'peak_reserved_mib':case['peak_reserved_mib'],
              'index_cache_mib':case['index_cache_mib'],
              'total_sparse_cache_mib':case['total_sparse_cache_mib']}))
        n=cobs['methods']['cobs'];p=cobs['profile']['cobs']
        converted.append(('cobs',{'score':n['ruler_score'],'all_hit':n['all_hit'],
              'decode_steps':n['decode_steps'],'decode_seconds':n['decode_seconds'],
              'cap_reached':n['cap_reached'],'matches_historical_ids':None,
              'forced_ms_per_step':p['median_forced_ms_per_step'],
              'index_seconds':n['index_build_seconds'],
              'postprefill16_seconds':n['index_build_seconds']+16*p['median_forced_ms_per_step']/1000,
              'prefill_seconds':cobs['prefill_seconds'],
              'peak_allocated_mib':n['peak_allocated_mib'],
              'peak_reserved_mib':n['peak_reserved_mib'],
              'index_cache_mib':n['index_cache_mib'],
              'total_sparse_cache_mib':None}))
        rows.append({'index':i,'task':full['task'],'input_sha256':full['input_sha256'],
                     'prompt_tokens':full['prompt_tokens'],'forced_input_ids':full['forced_input_ids'],
                     'methods':dict(converted)})
    warm=[r for r in rows if r['methods']['cobs']['index_seconds']<1.0]
    def group(sub):
        m={}
        for method in ('full','jssa_512','cobs'):
            x=[r['methods'][method] for r in sub]
            m[method]={'score':mean(t['score'] for t in x),'all_hit':sum(t['all_hit'] for t in x),
              'cap_reached':sum(t['cap_reached'] for t in x),
              'mean_decode_steps':mean(t['decode_steps'] for t in x),
              'prefill_seconds_mean':mean(t['prefill_seconds'] for t in x),
              'natural_postprefill_seconds_mean':mean(t['index_seconds']+t['decode_seconds'] for t in x),
              'natural_decode_ms_per_step':1000*sum(t['decode_seconds'] for t in x)/sum(t['decode_steps'] for t in x),
              'forced_ms_per_step_mean':mean(t['forced_ms_per_step'] for t in x),
              'forced_ms_per_step_median':statistics.median(t['forced_ms_per_step'] for t in x),
              'index_seconds_mean':mean(t['index_seconds'] for t in x),
              'index_seconds_median':statistics.median(t['index_seconds'] for t in x),
              'postprefill16_seconds_mean':mean(t['postprefill16_seconds'] for t in x),
              'peak_allocated_mib_median':statistics.median(t['peak_allocated_mib'] for t in x),
              'peak_reserved_mib_median':statistics.median(t['peak_reserved_mib'] for t in x),
              'peak_allocated_mib_max':max(t['peak_allocated_mib'] for t in x),
              'index_cache_mib_mean':mean(t['index_cache_mib'] for t in x),
              'total_sparse_cache_mib_mean':mean(t['total_sparse_cache_mib'] for t in x if t['total_sparse_cache_mib'] is not None) if method=='jssa_512' else None}
            ids=[t['matches_historical_ids'] for t in x if t['matches_historical_ids'] is not None]
            if ids:m[method]['historical_ids_matching']=sum(ids)
        comparisons={}
        for left,right in [('jssa_512','cobs'),('full','cobs'),('jssa_512','full')]:
            scores=[r['methods'][left]['score']-r['methods'][right]['score'] for r in sub]
            forced=[r['methods'][left]['forced_ms_per_step']-r['methods'][right]['forced_ms_per_step'] for r in sub]
            post=[r['methods'][left]['postprefill16_seconds']-r['methods'][right]['postprefill16_seconds'] for r in sub]
            hits=[int(r['methods'][left]['all_hit'])-int(r['methods'][right]['all_hit']) for r in sub]
            comparisons[f'{left}_minus_{right}']={'score_mean':mean(scores),'score_bootstrap95':boot(scores),
              'score_wins':sum(v>0 for v in scores),'score_losses':sum(v<0 for v in scores),
              'all_hit_net':sum(hits),'all_hit_wins':sum(v>0 for v in hits),'all_hit_losses':sum(v<0 for v in hits),
              'forced_ms_per_step_mean':mean(forced),'forced_ms_bootstrap95':boot(forced),
              'postprefill16_seconds_mean':mean(post),'postprefill16_bootstrap95':boot(post)}
        return {'n':len(sub),'methods':m,'comparisons':comparisons}
    report={'protocol':'100 same selected 32K prompts, one RTX3090, native method backends, 512 complete-history tokens plus visible current block, Full unbounded, 16 shared forced inputs, alternating 10-case method blocks',
            'warm_definition':'COBS index build <1 second, excluding graph captures after process start or recovery',
            'all':group(rows),'warm':group(warm),
            'niah_multivalue':group([r for r in rows if r['task']=='niah_multivalue']),
            'vt':group([r for r in rows if r['task']=='vt'])}
    suffix='' if args.limit==100 else f'_{args.limit}'
    save(root/f'paired_cases{suffix}.json',rows);save(root/f'summary{suffix}.json',report)
    if args.limit==100:
        save(root/'COMPLETE.json',{'status':'complete','cases':100,'warm_cases':len(warm),'summary':'summary.json'})
    print(json.dumps({'n':args.limit,'warm':len(warm),'all':report['all']['methods'],
                      'warm_postprefill':report['warm']['comparisons']['jssa_512_minus_cobs']['postprefill16_seconds_mean']},indent=2))
if __name__=='__main__':main()
