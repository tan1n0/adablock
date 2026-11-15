import sys,pathlib,json,time,collections,torch
ROOT=pathlib.Path('/root/adablock'); sys.path.insert(0,str(ROOT))
from utils.ranking_data import RankingDataset
from models.block_ranker import BlockRanker,BlockRankerConfig,to_device
from utils.ranking_loss import PairSamplingConfig,sample_pairs,weighted_rank_loss
OUT=pathlib.Path(__file__).parent
torch.set_num_threads(8); torch.manual_seed(42)
data=RankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003')
meta=data.metadata
model=BlockRanker(BlockRankerConfig(meta['head_dim'],meta['num_layers'],meta['num_kv_heads'])).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
sampling=PairSamplingConfig(.95,16,4,4,32,1e-6)
gen=torch.Generator().manual_seed(43)
counts=collections.Counter()
for e in data.shards: counts[data.manifest['documents'][e['document_index']]['source']]+=e['scenes']
results=[]
for size in [4096,8192,16384,32768]:
 source=f'token_pack_{size}'
 matching=[i for i,e in enumerate(data.shards) if data.manifest['documents'][e['document_index']]['source']==source]
 # Four distinct documents with all three layers, sampled uniformly within shards.
 chosen=[]
 docs=[]
 for i in matching:
  d=data.shards[i]['document_index']
  if d not in docs and len(docs)<4: docs.append(d)
  if d in docs: chosen.append(i)
 indices=[]
 for i in chosen:
  start=data.ends[i-1] if i else 0; n=data.shards[i]['scenes']
  indices.extend(start+int(j*(n-1)/20) for j in range(21))
 torch.cuda.synchronize(); start_time=time.perf_counter(); valid=0; candidates=0; steps=0
 for begin in range(0,len(indices),8):
  opt.zero_grad(set_to_none=True); losses=[]
  for idx in indices[begin:begin+8]:
   sample=to_device(data[idx],'cuda:0'); candidates+=len(sample['block_ids'])
   scores=model(sample)
   pairs=sample_pairs(scores,sample['teacher_mass'],sample['fixed_mass'],sampling,gen)
   if pairs.numel():
    losses.append(weighted_rank_loss(scores,sample['teacher_mass'],pairs)); valid+=1
  if losses:
   loss=torch.stack(losses).mean()
   assert torch.isfinite(loss)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step(); steps+=1
 torch.cuda.synchronize(); elapsed=time.perf_counter()-start_time
 r={'source':source,'scenes':len(indices),'seconds':elapsed,'seconds_per_scene':elapsed/len(indices),'valid_scenes':valid,'optimizer_steps':steps,'mean_candidate_blocks':candidates/len(indices),'full_dataset_scenes':counts[source]}
 results.append(r); print(json.dumps(r),flush=True)
estimate=sum(r['seconds_per_scene']*r['full_dataset_scenes'] for r in results)
report={'buckets':results,'estimated_training_hours_per_epoch':estimate/3600,'estimated_training_hours_10_epochs':10*estimate/3600,'caveats':'Includes cold shard loads, sequential one-scene model kernels, and sampled documents. Excludes full validation and checkpoint overhead; estimate not guaranteed. No training checkpoint saved.','device':torch.cuda.get_device_name(0)}
(OUT/'training_throughput_review.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
