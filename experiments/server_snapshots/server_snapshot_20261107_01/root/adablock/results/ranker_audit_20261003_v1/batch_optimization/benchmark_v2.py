import sys,pathlib,json,time,copy,collections,torch,statistics
ROOT=pathlib.Path('/root/adablock');sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,to_device
from utils.ranking_data import RankingDataset
from utils.ranking_loss import PairSamplingConfig,sample_pairs,weighted_rank_loss
from utils.ranking_batch import BatchRankingDataset,pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
OUT=pathlib.Path(__file__).parent
torch.set_num_threads(8);torch.manual_seed(42)
refdata=RankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003',cache_size=3)
newdata=BatchRankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003',cache_size=3)
meta=refdata.metadata;cfg=PairSamplingConfig()
template=BlockRanker(BlockRankerConfig(meta['head_dim'],meta['num_layers'],meta['num_kv_heads'])).cuda()
counts=collections.Counter();shardcounts=collections.Counter()
for e in refdata.shards:
 source=refdata.manifest['documents'][e['document_index']]['source']
 counts[source]+=e['scenes'];shardcounts[source]+=1
results=[];parity={'max_score_difference':0.,'pair_mismatch_scenes':0,'rng_mismatches':0,'feature_checks':0}
for size in [4096,8192,16384,32768]:
 source=f'token_pack_{size}'
 matching=[i for i,e in enumerate(refdata.shards) if refdata.manifest['documents'][e['document_index']]['source']==source][:3]
 indices=[]
 for i in matching:
  start=refdata.ends[i-1] if i else 0;n=refdata.shards[i]['scenes']
  indices.extend(start+int(j*(n-1)/83) for j in range(84))
 # Exact candidate/query/teacher feature parity for all benchmark scenes.
 for idx in indices:
  a,b=refdata[idx],newdata[idx]
  for key in b:
   if isinstance(b[key],torch.Tensor):assert torch.equal(a[key],b[key]),key
   else:assert a[key]==b[key],key
  parity['feature_checks']+=1
 for start in range(0,len(indices),8):
  a=[refdata[i] for i in indices[start:start+8]];b=[newdata[i] for i in indices[start:start+8]]
  with torch.no_grad():
   x=torch.cat([template(to_device(s,'cuda:0')) for s in a]);p=pack_scenes(b,'cuda:0');y=forward_packed(template,p)
  parity['max_score_difference']=max(parity['max_score_difference'],float((x-y).abs().max()))
  g1=torch.Generator().manual_seed(43+start);g2=torch.Generator().manual_seed(43+start)
  pa=[sample_pairs(x[p['offsets'][j]:p['offsets'][j+1]],s['teacher_mass'],s['fixed_mass'],cfg,g1).cpu() for j,s in enumerate(a)]
  pb=sample_batch_pairs(y.cpu(),b,p['offsets'],cfg,g2)
  parity['pair_mismatch_scenes']+=sum(not torch.equal(i,j) for i,j in zip(pa,pb))
  parity['rng_mismatches']+=int(not torch.equal(g1.get_state(),g2.get_state()))
 for method,threads in [('reference',8),('batched',8),('batched',1)]:
  torch.set_num_threads(threads);data=refdata if method=='reference' else newdata
  data.cache.clear();t=time.perf_counter()
  for i in matching:data[refdata.ends[i-1] if i else 0]
  cold=time.perf_counter()-t
  repetitions=[]
  for rep in range(2):
   model=copy.deepcopy(template);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);gen=torch.Generator().manual_seed(43)
   valid=0;steps=0;paircount=0
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start_time=time.perf_counter()
   for start in range(0,len(indices),8):
    batch=[data[i] for i in indices[start:start+8]];opt.zero_grad(set_to_none=True)
    if method=='reference':
     losses=[]
     for s in batch:
      sample=to_device(s,'cuda:0');score=model(sample)
      pairs=sample_pairs(score,sample['teacher_mass'],sample['fixed_mass'],cfg,gen)
      if pairs.numel():
       loss=weighted_rank_loss(score,sample['teacher_mass'],pairs);assert torch.isfinite(loss);float(loss.detach())
       losses.append(loss);valid+=1;paircount+=len(pairs)
     loss=torch.stack(losses).mean() if losses else None
    else:
     p=pack_scenes(batch,'cuda:0');scores=forward_packed(model,p)
     pairs=sample_batch_pairs(scores.detach().cpu(),batch,p['offsets'],cfg,gen)
     loss,per=loss_from_pairs(scores,batch,p['offsets'],pairs)
     if loss is not None:assert torch.isfinite(loss);float(per.detach().sum())
     valid+=sum(bool(x.numel()) for x in pairs);paircount+=sum(len(x) for x in pairs)
    if loss is not None:
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
   torch.cuda.synchronize();elapsed=time.perf_counter()-start_time
   repetitions.append({'seconds':elapsed,'valid_scenes':valid,'steps':steps,'pairs':paircount,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
   del model,opt
  r={'source':source,'method':method,'cpu_threads':threads,'scenes_per_repetition':len(indices),'repetitions':repetitions,'median_seconds_per_scene':statistics.median(x['seconds'] for x in repetitions)/len(indices),'cold_three_shard_load_seconds':cold,'full_dataset_scenes':counts[source],'full_dataset_shards':shardcounts[source]}
  results.append(r);print(json.dumps(r),flush=True)
est={}
for method,threads in [('reference',8),('batched',8),('batched',1)]:
 selected=[r for r in results if r['method']==method and r['cpu_threads']==threads]
 # Each file loaded once per epoch with the unchanged shard shuffle sampler.
 est[f'{method}_threads{threads}']=sum(r['median_seconds_per_scene']*r['full_dataset_scenes']+r['cold_three_shard_load_seconds']/3*r['full_dataset_shards'] for r in selected)*10/3600
report={'results':results,'parity':parity,'estimated_10_epoch_training_hours':est,'note':'Warm within-shard sampling plus measured one load per full shard per epoch. 252 scenes/length, three layers, one document per length, two repeats. Includes original batch8 updates. Excludes validation; estimates uncertain and CPU thread sensitivity measured.'}
(OUT/'benchmark_v2.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'estimates':est,'parity':parity}),flush=True)
