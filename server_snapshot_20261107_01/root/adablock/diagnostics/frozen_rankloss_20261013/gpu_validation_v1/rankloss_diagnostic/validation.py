"""Explicit selection-validation analysis: bounded CPU/GPU workers, shard commits/resume."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Unit-test helpers remain importable on Windows.
    resource = None
import subprocess
import time
import uuid

import torch

from .core import METHODS, Means, coverage_curve, required_prefix, stable_order, validate_scene
from .data import read_json, scene_key, sha256
from .prepare import resolve_config
from .run import FEATURE_KEYS, csv_write, json_write, load_runtime, write_svg


LABEL = 'validation_used_for_checkpoint_selection'
STATE = {}


def check_validation_source(config, dataset, receipt):
    observed = sha256(Path(config['data'])/'manifest.json')
    if observed != receipt['manifest_hashes']['validation']:
        raise ValueError('Validation analysis requires the exact frozen selection-validation manifest')
    if dataset.metadata['layers'] != list(range(28)):
        raise ValueError('Validation analysis requires all 28 layers')
    return observed


def initialize(config, options, device_queue=None):
    # Only the frozen small ranker is placed on a GPU; never load Qwen.
    torch.set_num_threads(1)
    device = device_queue.get() if device_queue is not None else 'cpu'
    if device != 'cpu':
        torch.cuda.set_device(torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dataset, model, baseline, receipt, hashes, payload = load_runtime(config, False, False, device)
    check_validation_source(config,dataset,receipt)
    root = Path(config['repo_root'])
    for filename in ('utils/ranking_batch.py','utils/ranking_loss.py'):
        hashes[filename] = sha256(root/filename)
        if hashes[filename] != receipt['source_config']['source_sha256'][filename]:
            raise ValueError(f'Batched inference source differs from frozen experiment: {filename}')
    from utils.ranking_batch import pack_scenes, forward_packed
    from models.block_ranker import load_ranker
    cpu_model, _ = load_ranker(config['checkpoint'],'cpu')
    cpu_model.float().eval().requires_grad_(False)
    STATE.update(dataset=dataset,model=model,baseline=baseline,receipt=receipt,hashes=hashes,
                 cpu_model=cpu_model,device=device,payload=payload,options=options,
                 pack=pack_scenes,forward=forward_packed)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    json_write(temp,value); temp.replace(path)


def validate_committed_shard(directory, signature):
    directory=Path(directory)
    saved=read_json(directory/'summary.json')
    if saved['signature']!=signature:
        raise ValueError('Committed shard signature mismatch')
    for name,key in (('scenes.jsonl.gz','scene_output_sha256'),('aggregation.json','aggregation_sha256')):
        if not (directory/name).is_file() or sha256(directory/name)!=saved[key]:
            raise ValueError(f'Incomplete/corrupt committed shard: {name}')
    return saved


def process_shard(task):
    shard_id, output, signature, quota = task
    ds, model, opt = STATE['dataset'], STATE['model'], STATE['options']
    entry = ds.shards[shard_id]
    committed = Path(output)/'shards'/f'{shard_id:04d}'
    if committed.exists():
        return validate_committed_shard(committed,signature)
    start = time.perf_counter()
    temporary = Path(output)/'shards'/f'.{shard_id:04d}.partial-{uuid.uuid4().hex}'
    temporary.mkdir(parents=True)
    first = ds.ends[shard_id-1] if shard_id else 0
    end = ds.ends[shard_id]
    indices = list(range(first,end))
    if quota is not None and quota < len(indices):
        indices = torch.linspace(first,end-1,quota).long().unique().tolist()
    source = ds.directory/entry['file']
    hash_start = time.perf_counter(); source_hash = sha256(source)
    hash_seconds = time.perf_counter()-hash_start
    load_start = time.perf_counter(); ds[indices[0]]
    load_seconds = time.perf_counter()-load_start
    coverage, quality = Means(), Means()
    documents, layers, heads = Counter(), Counter(), Counter()
    skipped, completed, parity_delta, coverage_parity, prefix_parity = 0, 0, 0., 0., 0
    compute_start = time.perf_counter()
    with gzip.open(temporary/'scenes.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stream, torch.inference_mode():
        for begin in range(0,len(indices),opt['batch_size']):
            batch_indices = indices[begin:begin+opt['batch_size']]
            samples = [ds[i] for i in batch_indices]
            for sample in samples: validate_scene(sample,ds.metadata)
            packed = STATE['pack'](samples,STATE['device'])
            learned = STATE['forward'](model,packed).float().cpu()
            # First scene per shard verifies GPU/batched scores and actual coverage against CPU.
            if begin == 0:
                reference = STATE['cpu_model']({k:samples[0][k] for k in FEATURE_KEYS}).float()
                candidate = learned[:packed['offsets'][1]]
                parity_delta = float((reference-candidate).abs().max()) if len(reference) else 0.
                if not torch.allclose(reference,candidate,atol=2e-5,rtol=1e-5):
                    raise ValueError(f'Batched model parity failed in shard {shard_id}: {parity_delta}')
                if len(reference):
                    sample=samples[0]
                    cpu_curve=coverage_curve(sample['teacher_mass'],sample['fixed_mass'],stable_order(reference,sample['block_ids']))
                    device_curve=coverage_curve(sample['teacher_mass'],sample['fixed_mass'],stable_order(candidate,sample['block_ids']))
                    coverage_parity=max(abs(float(cpu_curve[min(k,len(reference))])-float(device_curve[min(k,len(reference))])) for k in opt['budgets'])
                    prefix_parity=sum(required_prefix(cpu_curve,t,opt['budget_cap'])['required_k'] != required_prefix(device_curve,t,opt['budget_cap'])['required_k'] for t in opt['targets'])
                    if coverage_parity>2e-4:
                        raise ValueError(f'CPU/device coverage parity exceeded teacher tolerance: {coverage_parity}')
            for j,(index,sample) in enumerate(zip(batch_indices,samples)):
                n = len(sample['block_ids'])
                if not n:
                    skipped += 1; continue
                doc, layer, head = (int(sample[k]) for k in ('document_index','layer_id','kv_head_id'))
                documents[doc] += 1; layers[layer] += 1; heads[head] += 1
                scores = STATE['baseline'](sample)
                scores.update(RankLoss=learned[packed['offsets'][j]:packed['offsets'][j+1]],Oracle=sample['teacher_mass'])
                orders = {m:stable_order(scores[m],sample['block_ids']) for m in METHODS}
                curves = {m:coverage_curve(sample['teacher_mass'],sample['fixed_mass'],orders[m]) for m in METHODS}
                targets = [('absolute','',float(t)) for t in opt['targets']]
                targets += [('RankLoss_coverage',k,float(curves['RankLoss'][min(k,n)])) for k in opt['budgets']]
                groups = ('overall',f'layer/{layer}',f'head/{head}')
                diagnostic = {'analysis_split':LABEL,'scene_index':index,'scene_key':scene_key(sample),
                              'document_index':doc,'layer_id':layer,'kv_head_id':head,
                              'token_position':int(sample['token_position']),'candidate_count':n,
                              'fixed_mass':float(sample['fixed_mass']),
                              'budgets':opt['budgets'],'quality_targets':[t[2] for t in targets],
                              'output_error_status':'unavailable_no_attention_and_V','methods':{}}
                for m in METHODS:
                    middle, covs, gaps, actual, requirements = [], [], [], [], []
                    for k in opt['budgets']:
                        take = min(k,n); cov = float(curves[m][take]); gap = max(0.,float(curves['Oracle'][take])-cov)
                        row = dict(coverage=cov,middle_mass=cov-float(sample['fixed_mass']),oracle_gap=gap,
                                   actual_k=take,candidate_count=n,candidate_shortfall=n<k,
                                   fixed_mass=float(sample['fixed_mass']))
                        for group in groups: coverage.add((group,m,k),doc,row)
                        middle.append(row['middle_mass']); covs.append(cov); gaps.append(gap); actual.append(take)
                    for kind,reference_budget,target in targets:
                        prefix = required_prefix(curves[m],target,opt['budget_cap'])
                        for group in groups:
                            quality.add((group,m,kind,reference_budget,target if kind=='absolute' else ''),doc,
                                        dict(target_mass=target,**prefix))
                        requirements.append({k:prefix[k] for k in ('required_k','unreachable','over_budget_cap','within_cap_k')})
                    diagnostic['methods'][m] = {'coverage':covs,'middle_mass':middle,'oracle_gap':gaps,
                                                'actual_k':actual,'quality':requirements}
                stream.write(json.dumps(diagnostic,separators=(',',':'),allow_nan=False)+'\n')
                completed += 1
            if begin % 128 == 0:
                atomic_json(Path(output)/'work'/f'{shard_id:04d}.json',
                            {'analysis_split':LABEL,'shard':shard_id,'done':completed,'total':len(indices),
                             'elapsed_seconds':time.perf_counter()-start})
    compute_seconds = time.perf_counter()-compute_start
    aggregation = {'analysis_split':LABEL,'coverage':coverage.export(),'quality':quality.export()}
    json_write(temporary/'aggregation.json',aggregation)
    summary = {'analysis_split':LABEL,'signature':signature,'shard_id':shard_id,'file':entry['file'],
               'scene_output_sha256':sha256(temporary/'scenes.jsonl.gz'),
               'aggregation_sha256':sha256(temporary/'aggregation.json'),
               'source_sha256':source_hash,'scenes':completed,'skipped_no_middle':skipped,
               'original_scene_range':[first,end],'pilot_indices':indices if quota is not None else None,
               'document_scene_counts':dict(documents),'layer_scene_counts':dict(layers),'head_scene_counts':dict(heads),
               'hash_seconds':hash_seconds,'load_seconds':load_seconds,'compute_write_seconds':compute_seconds,
               'elapsed_seconds':time.perf_counter()-start,'cpu_device_score_max_delta':parity_delta,
               'cpu_device_coverage_max_delta':coverage_parity,'cpu_device_prefix_disagreement_count':prefix_parity,
               'device':STATE['device'],
               'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
               'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if STATE['device']!='cpu' else 0,
               'output_bytes':sum(p.stat().st_size for p in temporary.iterdir() if p.is_file())}
    json_write(temporary/'summary.json',summary)
    temporary.rename(committed)  # Commit only after closed gzip, summary, and exact aggregation state.
    return summary


def summarize(output, request, shard_results, wall_seconds):
    output = Path(output)
    coverage, quality = Means(), Means()
    documents, layers, heads = Counter(), Counter(), Counter()
    for r in shard_results:
        state = read_json(output/'shards'/f'{r["shard_id"]:04d}'/'aggregation.json')
        coverage.merge(state['coverage']); quality.merge(state['quality'])
        for counter,key in ((documents,'document_scene_counts'),(layers,'layer_scene_counts'),(heads,'head_scene_counts')):
            counter.update({int(k):v for k,v in r[key].items()})
    coverage_rows = [dict(analysis_split=LABEL,**r) for r in coverage.rows(('group','method','budget'))]
    quality_rows = [dict(analysis_split=LABEL,**r) for r in quality.rows(('group','method','target_kind','reference_budget','absolute_target'))]
    csv_write(output/'coverage.csv',coverage_rows); csv_write(output/'quality.csv',quality_rows)
    # Individual document statistics use the same exact sufficient statistics.
    document_rows = []
    for item in coverage.export():
        group,m,k = item['key']
        if group != 'overall': continue
        for doc,state in item['documents'].items():
            document_rows.append(dict(analysis_split=LABEL,document_index=int(doc),method=m,budget=k,
                                     scenes=state['scenes'],**{name:total/count for name,(total,count) in state['metrics'].items()}))
    csv_write(output/'coverage_by_document.csv',document_rows)
    summary = dict(request,status='complete',wall_seconds=wall_seconds,scenes=sum(documents.values()),
                   shards=len(shard_results),skipped_no_middle=sum(r['skipped_no_middle'] for r in shard_results),
                   document_scene_counts=dict(documents),layer_scene_counts=dict(layers),head_scene_counts=dict(heads),
                   max_worker_rss_kib=max(r['max_rss_kib'] for r in shard_results),
                   output_bytes=sum(r['output_bytes'] for r in shard_results),
                   output_error_status='unavailable_no_attention_and_V',
                   cpu_device_score_max_delta=max(r['cpu_device_score_max_delta'] for r in shard_results),
                   cpu_device_coverage_max_delta=max(r['cpu_device_coverage_max_delta'] for r in shard_results),
                   cpu_device_prefix_disagreement_count=sum(r['cpu_device_prefix_disagreement_count'] for r in shard_results),
                   max_cuda_peak_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in shard_results),
                   coverage=coverage_rows,quality=quality_rows)
    json_write(output/'summary.json',summary)
    write_svg(output/'coverage.svg',[r for r in coverage_rows if r['group']=='overall'])
    svg=output/'coverage.svg'
    svg.write_text(svg.read_text().replace('</svg>',f'<title>{LABEL}</title><text x="450" y="14" text-anchor="middle" font-size="12">{LABEL}</text></svg>'),encoding='utf-8')
    return summary


def verify_available_gpus(indices):
    if len(set(indices)) != len(indices) or not indices or len(indices)>4:
        raise ValueError('Select1–4 distinct available GPU indices')
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=15)
    observed={int(parts[0]):(int(parts[1]),int(parts[2])) for parts in (line.split(',') for line in result.stdout.splitlines())}
    for index in indices:
        if index not in observed or observed[index][0]>1024 or observed[index][1]>10:
            raise ValueError(f'GPU{index} is unavailable/busy; will not compete with other jobs')
    return observed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--analysis-split',choices=['validation'],required=True)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--budgets',type=int,nargs='+',default=[4,8,16,32,64])
    p.add_argument('--targets',type=float,nargs='+',default=[.90,.95,.99]); p.add_argument('--budget-cap',type=int,default=64)
    p.add_argument('--pilot',action='store_true'); p.add_argument('--pilot-scenes',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if not 1 <= a.workers <= 8 or not 1 <= a.batch_size <= 16 or min(a.budgets)<0 or a.budget_cap<0:
        p.error('Limit workers to1–8, batch size to1–16; budgets/cap must be nonnegative')
    if any(not 0<t<=1 for t in a.targets) or a.pilot_scenes<1: p.error('Invalid target or pilot size')
    gpu_snapshot=verify_available_gpus(a.gpus) if a.device=='cuda' else None
    if a.device=='cuda' and a.workers>len(a.gpus): p.error('At most one worker per selected GPU')
    start = time.perf_counter(); config=resolve_config(a.config)
    options={'batch_size':a.batch_size,'budgets':sorted(set(a.budgets)),'targets':a.targets,'budget_cap':a.budget_cap,'device':a.device}
    initialize(config,options)
    ds=STATE['dataset']; output=Path(a.output)
    shard_ids=list(range(len(ds.shards)))
    if a.pilot:
        shard_ids=[i for i,s in enumerate(ds.shards) if s['document_index'] in (0,4,8,12) and s['layer_id'] in (0,14)]
    identity={'analysis_split':LABEL,'pilot':a.pilot,'pilot_scenes':a.pilot_scenes if a.pilot else None,
              'config':config,'options':options,'manifest_sha256':check_validation_source(config,ds,STATE['receipt']),
              'checkpoint_sha256':STATE['receipt']['checkpoint_sha256'],'source_sha256':STATE['hashes'],
              'shard_ids':shard_ids,'dataset_scenes':len(ds),'dataset_documents':len(ds.manifest['documents'])}
    identity['diagnostic_code_sha256']={name:sha256(Path(__file__).parent/name) for name in ('validation.py','core.py','run.py','data.py')}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    request=dict(identity,signature=signature,workers=a.workers,threads_per_worker=1,device=a.device,gpus=a.gpus if a.device=='cuda' else [],gpu_initial_snapshot=gpu_snapshot,
                 versions={'python':platform.python_version(),'torch':torch.__version__},
                 quality_target_order=['absolute/'+str(t) for t in a.targets]+['RankLoss@'+str(k) for k in options['budgets']],
                 semantics='selection-validation diagnostic; no independent test claim; raw mass + fixed sink/local/self once; min(k,n); exact full-ranking integer prefixes; scene and packed-document means',
                 unavailable_output_error='ranking v1 lacks full per-head attention/V',
                 scene_output='per-shard compact gzip JSONL; exact aggregation state and summary committed atomically')
    if output.exists():
        if not a.resume or read_json(output/'request.json')['signature'] != signature:
            raise ValueError('Use new directory or --resume with identical frozen inputs/options')
    else:
        (output/'shards').mkdir(parents=True); (output/'work').mkdir(); json_write(output/'request.json',request)
    completed={}
    for i in shard_ids:
        path=output/'shards'/f'{i:04d}'/'summary.json'
        if path.exists():
            r=validate_committed_shard(path.parent,signature)
            if sha256(ds.directory/r['file']) != r['source_sha256']: raise ValueError('Previously evaluated shard changed')
            completed[i]=r
    pending=[i for i in shard_ids if i not in completed]
    print(json.dumps({'analysis_split':LABEL,'stage':'start','total_shards':len(shard_ids),'already_completed':len(completed),
                      'total_dataset_scenes':len(ds),'workers':a.workers}),flush=True)
    if pending:
        context=multiprocessing.get_context('spawn')
        device_queue=context.Queue()
        for device in (['cuda:'+str(i) for i in a.gpus[:a.workers]] if a.device=='cuda' else ['cpu']*a.workers): device_queue.put(device)
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=context,
                                 initializer=initialize,initargs=(config,options,device_queue)) as pool:
            futures=[pool.submit(process_shard,(i,str(output),signature,a.pilot_scenes if a.pilot else None)) for i in pending]
            for future in as_completed(futures):
                r=future.result(); completed[r['shard_id']]=r
                progress={'analysis_split':LABEL,'status':'running','completed_shards':len(completed),'total_shards':len(shard_ids),
                          'completed_scenes':sum(v['scenes'] for v in completed.values()),'wall_seconds':time.perf_counter()-start,
                          'last_shard':r['shard_id'],'max_worker_rss_kib':max(v['max_rss_kib'] for v in completed.values())}
                atomic_json(output/'progress.json',progress); print(json.dumps(progress),flush=True)
    if len(completed)!=len(shard_ids): raise ValueError('Incomplete shard set')
    if sha256(config['checkpoint'])!=request['checkpoint_sha256']: raise ValueError('Checkpoint changed during run')
    if sha256(ds.directory/'manifest.json')!=request['manifest_sha256']: raise ValueError('Manifest changed during run')
    summary=summarize(output,request,[completed[i] for i in shard_ids],time.perf_counter()-start)
    if not a.pilot and summary['scenes']+summary['skipped_no_middle']!=len(ds): raise ValueError('Full scene count mismatch')
    atomic_json(output/'progress.json',{'analysis_split':LABEL,'status':'complete','scenes':summary['scenes'],
                                      'shards':summary['shards'],'wall_seconds':summary['wall_seconds']})
    print(json.dumps({'analysis_split':LABEL,'status':'complete','scenes':summary['scenes'],'wall_seconds':summary['wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
