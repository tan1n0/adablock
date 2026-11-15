"""Verify all committed inputs/output rows, then write an evidence-based validation report."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import time

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/gpu_validation_v1')
sys.path.insert(0,str(root))
from rankloss_diagnostic.core import Means
from rankloss_diagnostic.data import read_json,sha256
from rankloss_diagnostic.run import csv_write,json_write
from rankloss_diagnostic.validation import LABEL,validate_committed_shard

out=root/'results/full_validation_gpu_v1'
waiting=time.perf_counter();previous=-1
while not (out/'summary.json').exists():
    progress=read_json(out/'progress.json') if (out/'progress.json').exists() else {}
    done=progress.get('completed_scenes',0)
    if done!=previous:
        print(json.dumps({'stage':'waiting_for_full_completion','completed_scenes':done,'total':414064}),flush=True)
        previous=done
    launch=read_json(root/'launch.json')
    status=Path('/proc')/str(launch['pid'])/'stat'
    if not status.exists() or status.read_text().split()[2]=='Z':
        raise RuntimeError('Full run exited without complete summary; inspect retained log/commits')
    if time.perf_counter()-waiting>1800:raise TimeoutError('Completion wait exceeded30 minutes; inspect progress')
    time.sleep(10)
summary=read_json(out/'summary.json')
if summary['status']!='complete' or summary['scenes']!=414064 or summary['shards']!=448:
    raise RuntimeError('Full validation is not complete')
manifest_path=Path(summary['config']['data'])/'manifest.json'
manifest=read_json(manifest_path)
assert sha256(manifest_path)==summary['manifest_sha256']
assert sha256(summary['config']['checkpoint'])==summary['checkpoint_sha256']
for name,digest in summary['diagnostic_code_sha256'].items():
    assert sha256(root/'rankloss_diagnostic'/name)==digest


def verify_shard(i):
    folder=out/'shards'/f'{i:04d}'
    saved=validate_committed_shard(folder,summary['signature'])
    entry=manifest['shards'][i]
    assert sha256(manifest_path.parent/entry['file'])==saved['source_sha256']
    start,end=saved['original_scene_range'];count=0;heads=Counter();candidates=Counter()
    raw_coverage={(m,k):[0.,0.,0.,0] for m in ('RankLoss','COBS','Quest','Oracle') for k in (4,8,16,32,64)}
    with gzip.open(folder/'scenes.jsonl.gz','rt') as stream:
        for line in stream:
            d=json.loads(line);assert d['analysis_split']==LABEL
            assert d['scene_index']==start+count
            assert d['document_index']==entry['document_index'] and d['layer_id']==entry['layer_id']
            assert d['kv_head_id']==count%4
            assert d['output_error_status']=='unavailable_no_attention_and_V'
            n=d['candidate_count'];fixed=d['fixed_mass'];heads[d['kv_head_id']]+=1;candidates[n]+=1
            for j,k in enumerate(d['budgets']):
                oracle=d['methods']['Oracle']['coverage'][j]
                for method in ('RankLoss','COBS','Quest','Oracle'):
                    m=d['methods'][method];cov=m['coverage'][j]
                    assert m['actual_k'][j]==min(n,k)
                    assert fixed-1e-7<=cov<=1.0002 and oracle+1e-7>=cov
                    assert abs((cov-fixed)-m['middle_mass'][j])<1e-10
                    assert abs(max(0.,oracle-cov)-m['oracle_gap'][j])<1e-10
                    totals=raw_coverage[(method,k)]
                    totals[0]+=cov;totals[1]+=min(n,k);totals[2]+=n<k;totals[3]+=1
            for m in d['methods'].values():
                for q in m['quality']:
                    required=q['required_k']
                    assert q['unreachable']==(required is None)
                    if required is not None:
                        assert isinstance(required,int) and 0<=required<=n
                        assert q['over_budget_cap']==(required>64)
                    assert q['within_cap_k']==(required if required is not None and required<=64 else None)
            count+=1
    assert count==saved['scenes']==entry['scenes']==end-start
    assert sum(heads.values())==count and all(v==count//4 for v in heads.values())
    return saved,candidates,raw_coverage


audit_start=time.perf_counter();rows={};candidate_counts=Counter();raw_by_doc={}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(verify_shard,i) for i in range(448)]
    for future in as_completed(futures):
        saved,candidates,raw=future.result();rows[saved['shard_id']]=saved;candidate_counts.update(candidates)
        doc=manifest['shards'][saved['shard_id']]['document_index']
        target=raw_by_doc.setdefault(doc,{key:[0.,0.,0.,0] for key in raw})
        for key,values in raw.items():
            for j,value in enumerate(values):target[key][j]+=value
        if len(rows)%56==0:print(json.dumps({'integrity_verified_shards':len(rows),'total':448}),flush=True)
assert sum(r['scenes'] for r in rows.values())==414064
assert len(summary['document_scene_counts'])==16 and len(summary['layer_scene_counts'])==28
assert summary['head_scene_counts']=={str(i):103516 for i in range(4)}
assert all(v==14788 for v in summary['layer_scene_counts'].values())
raw_aggregate_delta=0.
for row in summary['coverage']:
    if row['group']!='overall':continue
    values=[doc[(row['method'],row['budget'])] for doc in raw_by_doc.values()]
    for j,name in ((0,'coverage'),(1,'actual_k'),(2,'candidate_shortfall')):
        actual=(sum(v[j] for v in values)/sum(v[3] for v in values) if row['weighting']=='scene'
                else sum(v[j]/v[3] for v in values)/len(values))
        delta=abs(actual-row[name]);raw_aggregate_delta=max(raw_aggregate_delta,delta)
        assert delta<1e-9

# Recover every document's exact prefix statistics without recomputing inference.
quality=Means()
for i in range(448):quality.merge(read_json(out/'shards'/f'{i:04d}'/'aggregation.json')['quality'])
doc_quality=[]
for item in quality.export():
    group,method,kind,reference,target=item['key']
    if group!='overall':continue
    for doc,state in item['documents'].items():
        row={'analysis_split':LABEL,'document_index':int(doc),'method':method,'target_kind':kind,
             'reference_budget':reference,'absolute_target':target,'scenes':state['scenes']}
        for name,(total,count) in state['metrics'].items():row[name]=total/count;row[name+'_available_scenes']=count
        doc_quality.append(row)
csv_write(out/'quality_by_document.csv',doc_quality)
csv_write(out/'candidate_count_histogram.csv',[{'analysis_split':LABEL,'candidate_count':n,'scenes':count} for n,count in sorted(candidate_counts.items())])
integrity={'status':'passed','analysis_split':LABEL,'shards':448,'scenes':414064,'documents':16,
           'checks':'all source/checkpoint/manifest/code and committed output hashes; all gzip rows/order/head identities; same fixed/middle/oracle budgets; exact integer/null/cap schema',
           'verification_seconds':time.perf_counter()-audit_start,'metadata':manifest['metadata'],
           'raw_scene_recomputed_aggregate_max_delta':raw_aggregate_delta,
           'document_identities':manifest['documents'],'candidate_min':min(candidate_counts),'candidate_max':max(candidate_counts),
           'input_shard_bytes':sum((manifest_path.parent/s['file']).stat().st_size for s in manifest['shards']),
           'complete_output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
json_write(out/'integrity_audit.json',integrity)

methods=('RankLoss','COBS','Quest','Oracle');budgets=[4,8,16,32,64]
coverage={(r['weighting'],r['method'],r['budget']):r for r in summary['coverage'] if r['group']=='overall'}
quality={(r['weighting'],r['method'],r['target_kind'],r['reference_budget'],r['absolute_target']):r for r in summary['quality'] if r['group']=='overall'}
lines=['# Full selection-validation diagnostic','', '**validation_used_for_checkpoint_selection — not an independent test.**', '',
       f"Completed:414064 scenes /16 packed documents /28 layers /4 KV heads /448 shards. GPU run elapsed:{summary['wall_seconds']:.3f}s.",
       'Frozen all28 step50000 checkpoint SHA256:'+summary['checkpoint_sha256'],
       f"All committed input/output hashes and all414064 emitted row identities verified in{integrity['verification_seconds']:.3f}s.",
       '17 CPU unit/integration tests passed; no collection, training, base-model forward or downstream generation.',
       '', 'Coverage includes fixed sink/local/self once plus raw selected middle mass; budget counts only middle blocks.',
       'COBS and Quest reuse the actual repository baseline_scores (cobs_legacy_mean_logits / quest_aabb), including its legacy mean raw-score GQA convention. This is not a claim about a different normalized-head COBS implementation.']
for weighting in ('scene','document_equal'):
    lines += ['', '## '+weighting+' weighted coverage','', '| Middle budget | RankLoss | COBS | Quest | Oracle |', '|---:|---:|---:|---:|---:|']
    for k in budgets:lines.append('|'+str(k)+'|'+'|'.join(f'{coverage[(weighting,m,k)]["coverage"]*100:.6f}%' for m in methods)+'|')
    lines += ['', '## '+weighting+' weighted exact minimum prefix to match each scene\'s RankLoss coverage','',
              '| RankLoss reference budget | RankLoss | COBS | Quest | Oracle |','|---:|---:|---:|---:|---:|']
    for k in budgets:lines.append('|'+str(k)+'|'+'|'.join(f'{quality[(weighting,m,"RankLoss_coverage",k,"")]["required_k"]:.6f}' for m in methods)+'|')
    lines += ['', '## '+weighting+' weighted exact minimum prefix at absolute coverage','',
              '| Target | RankLoss | COBS | Quest | Oracle |','|---:|---:|---:|---:|---:|']
    for t in (.90,.95,.99):lines.append('|'+str(t)+'|'+'|'.join(f'{quality[(weighting,m,"absolute","",t)]["required_k"]:.6f}' for m in methods)+'|')
lines += ['', '## Candidate shortfall','', '| Requested middle budget | Scene fraction n<k | Document-equal fraction n<k |','|---:|---:|---:|']
for k in budgets:lines.append(f'|{k}|{coverage[("scene","RankLoss",k)]["candidate_shortfall"]:.6%}|{coverage[("document_equal","RankLoss",k)]["candidate_shortfall"]:.6%}|')
lines += ['', '## Layer variation','', '| Budget | Layers RankLoss below COBS | Layers RankLoss below Quest |','|---:|---|---|']
for k in budgets:
    losses=[]
    for baseline in ('COBS','Quest'):
        lower=[]
        for layer in range(28):
            data={r['method']:r for r in summary['coverage'] if r['group']==f'layer/{layer}' and r['weighting']=='scene' and r['budget']==k}
            if data['RankLoss']['coverage']<data[baseline]['coverage']:lower.append(layer)
        losses.append(str(lower))
    lines.append('|'+str(k)+'|'+'|'.join(losses)+'|')
lines += ['', '## Limits and numerical audit','',
          'All metrics are descriptive on selection validation, not an independent generalization result. Scenes within one packed document are not independent documents.',
          'Required K is an exact per-scene integer on the actual complete ranking; fractional summary values are means, not interpolated K. Cap64 is reported separately from actual K.',
          'Prefix hit tolerance is fixed at1e-7, matching the existing coverage-positive convention; the shortest integer prefix is exact at this stated threshold.',
          'All414064 scenes lack real per-head attention/V; AV output error remains unavailable.',
          'CPU/GPU FP32 can rank near-equal blocks differently. All256 pilot scenes were recomputed on CPU: one of2048 quality comparisons differed (CPU65 vs GPU64 at that scene\'s GPU RankLoss@64 target); max coverage difference6.95656e-5.',
          'The initial8 first-scene checks were equal in coverage/prefix; the expanded pilot numerical_difference_audit.json explicitly preserves the nonzero difference. Full results use actual GPU ranking without rounding scores or forcing prefix agreement.',
          '', '## Deliverables','',
          'summary.json; coverage.csv and quality.csv include overall/layer/head groups and both weights; coverage_by_document.csv; quality_by_document.csv; candidate_count_histogram.csv; coverage.svg;',
          'shards/0000..0447 each contain scenes.jsonl.gz, aggregation.json, summary.json; request.json, progress.json and integrity_audit.json preserve inputs, hashes and counts.',
          'Full launch/log and representative pilot with numeric audit are retained in the parent run directory.',
          '', '## Sample distribution','', 'Document scenes:'+json.dumps(summary['document_scene_counts']),
          'Every layer:14788 scenes. Every KV head:103516 scenes.',
          f"Output bytes:{integrity['complete_output_bytes']}; worker peak RSS:{summary['max_worker_rss_kib']}KiB; max GPU tensor allocation:{summary['max_cuda_peak_allocated_bytes']}bytes."]
with (out/'REPORT.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
print(json.dumps({'status':'complete_verified','wall_seconds':summary['wall_seconds'],'integrity':integrity,
                  'report':str(out/'REPORT.md'),
                  'overall_coverage':[r for r in summary['coverage'] if r['group']=='overall'],
                  'absolute95_quality':[r for r in summary['quality'] if r['group']=='overall' and r['target_kind']=='absolute' and r['absolute_target']==.95]},ensure_ascii=False),flush=True)
