import sys,pathlib,json,time,copy,collections,torch
ROOT=pathlib.Path('/root/adablock');sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,to_device
from utils.ranking_data import RankingDataset
from utils.ranking_loss import PairSamplingConfig,sample_pairs,weighted_rank_loss
from utils.ranking_batch import pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
OUT=pathlib.Path(__file__).parent; torch.set_num_threads(8);torch.manual_seed(42)
data=RankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003')
meta=data.metadata; cfg=PairSamplingConfig(); template=BlockRanker(BlockRankerConfig(meta['head_dim'],meta['num_layers'],meta['num_kv_heads'])).cuda()
counts=collections.Counter()
for e in data.shards:counts[data.manifest['documents'][e['document_index']]['source']]+=e['scenes']
results=[]; parity=[]
def now():torch.cuda.synchronize();return time.perf_counter()
for size in [4096,8192,16384,32768]:
 source=f'token_pack_{size}'; docs=[];chosen=[]
 for i,e in enumerate(data.shards):
  if data.manifest['documents'][e['document_index']]['source']!=source:continue
  d=e['document_index']
  if d not in docs and len(docs)<4:docs.append(d)
  if d in docs:chosen.append(i)
 indices=[]
 for i in chosen:
  start=data.ends[i-1] if i else 0;n=data.shards[i]['scenes']
  indices.extend(start+int(j*(n-1)/20) for j in range(21))
 # Real-data fixed-model score and sampling parity, independent of training drift.
 for start in range(0,len(indices),8):
  batch=[data[i] for i in indices[start:start+8]]
  with torch.no_grad():
   old=torch.cat([template(to_device(s,'cuda:0')) for s in batch])
   p=pack_scenes(batch,'cuda:0');new=forward_packed(template,p)
  g1=torch.Generator().manual_seed(43+start);g2=torch.Generator().manual_seed(43+start)
  a=sample_batch_pairs(old.cpu(),batch,p['offsets'],cfg,g1)
  b=sample_batch_pairs(new.cpu(),batch,p['offsets'],cfg,g2)
  mismatch=[j for j,(x,y) in enumerate(zip(a,b)) if not torch.equal(x,y)]
  parity.append({'source':source,'scenes':len(batch),'max_score_abs_difference':float((old-new).abs().max()),'pair_mismatch_scenes':len(mismatch),'rng_state_equal':torch.equal(g1.get_state(),g2.get_state())})
 for method in ['reference','batched']:
  model=copy.deepcopy(template);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
  gen=torch.Generator().manual_seed(43);parts=collections.defaultdict(float);valid=0;steps=0;paircount=0
  data.cache.clear(); torch.cuda.reset_peak_memory_stats(); overall=now()
  for start in range(0,len(indices),8):
   t=now();batch=[data[i] for i in indices[start:start+8]];parts['cpu_load_gather']+=now()-t
   opt.zero_grad(set_to_none=True)
   if method=='reference':
    losses=[]
    for s in batch:
     t=now();sample=to_device(s,'cuda:0');parts['h2d']+=now()-t
     t=now();score=model(sample);parts['forward']+=now()-t
     t=now();pairs=sample_pairs(score,sample['teacher_mass'],sample['fixed_mass'],cfg,gen);parts['cpu_sampling_and_score_transfer']+=now()-t
     if pairs.numel():
      t=now();loss=weighted_rank_loss(score,sample['teacher_mass'],pairs)
      assert torch.isfinite(loss);float(loss.detach());losses.append(loss);parts['loss_and_statistics']+=now()-t
      valid+=1;paircount+=len(pairs)
    loss=torch.stack(losses).mean() if losses else None
   else:
    t=now();p=pack_scenes(batch,'cuda:0');parts['h2d_and_pack']+=now()-t
    t=now();score=forward_packed(model,p);parts['forward']+=now()-t
    t=now();pairs=sample_batch_pairs(score.detach().cpu(),batch,p['offsets'],cfg,gen);parts['cpu_sampling_and_score_transfer']+=now()-t
    t=now();loss,per=loss_from_pairs(score,batch,p['offsets'],pairs)
    if loss is not None:assert torch.isfinite(loss);float(per.detach().sum())
    parts['loss_and_statistics']+=now()-t
    valid+=sum(bool(x.numel()) for x in pairs);paircount+=sum(len(x) for x in pairs)
   if loss is not None:
    t=now();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1;parts['backward_clip_step']+=now()-t
  elapsed=now()-overall
  result={'source':source,'method':method,'scenes':len(indices),'seconds':elapsed,'seconds_per_scene':elapsed/len(indices),'valid_scenes':valid,'steps':steps,'pairs':paircount,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'stages_seconds':dict(parts),'full_dataset_scenes':counts[source]}
  results.append(result);print(json.dumps(result),flush=True);del model,opt
report={'results':results,'real_data_initial_parity':parity,'note':'Synchronized stage instrumentation adds overhead. Sampling RNG preserved at fixed scores; near-tie mismatches, if any, explicitly counted. No final checkpoint saved.','estimated_10_epoch_training_hours':{method:sum(r['seconds_per_scene']*r['full_dataset_scenes'] for r in results if r['method']==method)*10/3600 for method in ['reference','batched']}}
(OUT/'benchmark_v1.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'estimates':report['estimated_10_epoch_training_hours'],'pair_mismatch_scenes':sum(x['pair_mismatch_scenes'] for x in parity),'max_score_abs_difference':max(x['max_score_abs_difference'] for x in parity),'all_rng_equal':all(x['rng_state_equal'] for x in parity)}),flush=True)
