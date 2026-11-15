import pathlib,sys,json,time,torch
ROOT=pathlib.Path('/root/adablock');sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig
from utils.ranking_batch import BatchRankingDataset,pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
from utils.ranking_loss import PairSamplingConfig
OUT=pathlib.Path(__file__).parent
while not (OUT/'ddp_trial_status.json').exists() or json.loads((OUT/'ddp_trial_status.json').read_text())['state']=='running':time.sleep(3)
state=json.loads((OUT/'ddp_trial_status.json').read_text());assert state['state']=='completed',state
torch.set_num_threads(1);torch.cuda.set_device(1);device='cuda:1'
saved=torch.load(OUT/'paused_disk_state.pt',map_location='cpu',weights_only=True)
model=BlockRanker(BlockRankerConfig(**saved['config'])).to(device);model.load_state_dict(saved['model_state_dict'])
data=BatchRankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003',cache_size=4);cfg=PairSamplingConfig()
rows=[]
for size in [4096,8192,16384,32768]:
 si=next(i for i,e in enumerate(data.shards) if data.manifest['documents'][e['document_index']]['source']==f'token_pack_{size}')
 end=data.ends[si]
 for bs in [8,32,64]:
  batch=[data[i] for i in range(end-bs,end)]
  model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(1);torch.cuda.synchronize(1);start=time.perf_counter()
  p=pack_scenes(batch,device);scores=forward_packed(model,p);cpu=scores.detach().cpu()
  st=time.perf_counter();pairs=sample_batch_pairs(cpu,batch,p['offsets'],cfg,torch.Generator().manual_seed(321));sampling=time.perf_counter()-st
  loss,per=loss_from_pairs(scores,batch,p['offsets'],pairs)
  if loss is not None:loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  torch.cuda.synchronize(1)
  r={'source_length_bucket':size,'batch':bs,'candidate_blocks_min':min(len(x['block_ids']) for x in batch),'candidate_blocks_max':max(len(x['block_ids']) for x in batch),'peak_allocated_mib':torch.cuda.max_memory_allocated(1)/2**20,'forward_backward_seconds':time.perf_counter()-start,'cpu_sampling_seconds':sampling}
  rows.append(r);print(json.dumps(r),flush=True)
  del p,scores,cpu,pairs,loss,per,batch
(OUT/'length_peak_memory.json').write_text(json.dumps({'rows':rows,'note':'Allocated tensor peak on otherwise idle GPU1, real last queries in each source bucket; includes backward, no optimizer updates.'},indent=2))
