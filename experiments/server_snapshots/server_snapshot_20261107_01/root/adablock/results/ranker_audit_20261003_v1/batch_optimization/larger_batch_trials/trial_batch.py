import sys,pathlib,json,time,copy,random,argparse,collections,torch
ROOT=pathlib.Path('/root/adablock');sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,save_ranker
from utils.ranking_data import RankingDataset
from utils.ranking_batch import BatchRankingDataset,pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
from utils.ranking_batch_eval import evaluate_batched
from utils.ranking_loss import PairSamplingConfig
p=argparse.ArgumentParser();p.add_argument('--batch',type=int);p.add_argument('--device');p.add_argument('--tag');args=p.parse_args()
OUT=pathlib.Path(__file__).parent;dest=OUT/args.tag;dest.mkdir(exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(42);device=torch.device(args.device);torch.cuda.set_device(device)
snapshot=torch.load(OUT/'paused_disk_state.pt',weights_only=True,map_location='cpu')
data=BatchRankingDataset(ROOT/'data/ranker_cobs_train_full_32k_20261003',cache_size=48)
val=RankingDataset(ROOT/'data/ranker_cobs_validation16_20261003',cache_size=48)
model=BlockRanker(BlockRankerConfig(**snapshot['config'])).to(device);model.load_state_dict(snapshot['model_state_dict'])
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);opt.load_state_dict(snapshot['optimizer_state_dict'])
cfg=PairSamplingConfig();gen=torch.Generator();gen.set_state(snapshot['pair_generator_state'])
m=json.loads((OUT/'trial_scenes.json').read_text());indices=m['indices']
# All trials use the same source scenes/order, starting model and optimizer.
start=time.perf_counter()
for i in m['shard_indices']:
 data[data.ends[i-1] if i else 0]
load_seconds=time.perf_counter()-start
before=evaluate_batched(model,val,device,max_scenes=512,batch_size=args.batch)
model.train();stats=collections.defaultdict(float);by_source=collections.defaultdict(lambda:collections.defaultdict(float))
torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
for begin in range(0,len(indices),args.batch):
 batch=[data[i] for i in indices[begin:begin+args.batch]];opt.zero_grad(set_to_none=True)
 t=time.perf_counter();packed=pack_scenes(batch,device);scores=forward_packed(model,packed);scores_cpu=scores.detach().cpu()
 stats['pack_forward_and_d2h_seconds']+=time.perf_counter()-t
 t=time.perf_counter();pairs=sample_batch_pairs(scores_cpu,batch,packed['offsets'],cfg,gen);stats['cpu_sampling_seconds']+=time.perf_counter()-t
 loss,per=loss_from_pairs(scores,batch,packed['offsets'],pairs)
 if loss is not None:
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  stats['steps']+=1;stats['loss_sum']+=float(per.detach().sum());stats['valid_scenes']+=len(per)
 stats['pairs']+=sum(len(x) for x in pairs)
torch.cuda.synchronize(device);elapsed=time.perf_counter()-start
peak=torch.cuda.max_memory_allocated(device)/2**20
after=evaluate_batched(model,val,device,max_scenes=512,batch_size=args.batch)
# Worst-length candidates stress on a separate model; does not alter trial model.
stress=copy.deepcopy(model);stress.train()
long_indices=m['long_candidate_indices'][-args.batch:]
batch=[data[i] for i in long_indices]
torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);t=time.perf_counter()
packed=pack_scenes(batch,device);scores=forward_packed(stress,packed)
stressgen=torch.Generator().manual_seed(123)
pairs=sample_batch_pairs(scores.detach().cpu(),batch,packed['offsets'],cfg,stressgen)
loss,per=loss_from_pairs(scores,batch,packed['offsets'],pairs)
if loss is not None:loss.backward();torch.nn.utils.clip_grad_norm_(stress.parameters(),1.,error_if_nonfinite=True)
torch.cuda.synchronize(device)
stress_result={'batch':len(batch),'candidate_blocks_min':min(len(x['block_ids']) for x in batch),'candidate_blocks_max':max(len(x['block_ids']) for x in batch),'seconds':time.perf_counter()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,'loss':float(loss) if loss is not None else None}
report={'tag':args.tag,'batch_size':args.batch,'device':args.device,'cpu_threads':1,'source_checkpoint_seen':snapshot['seen'],'scenes':len(indices),'source_bucket_counts':m['source_bucket_counts'],'training_seconds':elapsed,'preload_seconds':load_seconds,'scenes_per_second':len(indices)/elapsed,'steps_per_second':stats['steps']/elapsed,'peak_allocated_mib':peak,'stats':dict(stats),'mean_train_loss':stats['loss_sum']/max(1,stats['valid_scenes']),'validation_before':before,'validation_after':after,'long_candidate_stress':stress_result,'note':'Short batch trial only; same scene count/order/start optimizer/lr, but batch changes optimizer steps and trajectory. Parallel trials may contend for host CPU/memory; isolated repeat required.'}
(dest/'result.json').write_text(json.dumps(report,indent=2))
save_ranker(dest/'trial_final.pt',model,data.metadata,{'trial_only':True,'batch_size':args.batch,'scenes':len(indices),'lr':2e-4})
print(json.dumps({k:v for k,v in report.items() if k not in ['validation_before','validation_after']}),flush=True)
