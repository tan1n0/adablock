"""Short DDP correctness/throughput trial, never a full training launch."""
import sys,pathlib,json,time,copy,os,argparse,collections
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,save_ranker
from utils.ranking_data import RankingDataset
from utils.ranking_batch import BatchRankingDataset,pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
from utils.ranking_loss import PairSamplingConfig
from utils.ranking_batch_eval import evaluate_batched
from utils.ranking_ddp import DistributedBatchRanker,distributed_scene_loss,global_batch_for_rank
p=argparse.ArgumentParser();p.add_argument("--local-batch",type=int,required=True);p.add_argument("--tag",required=True);p.add_argument("--scene-manifest",default="trial_scenes.json");p.add_argument("--cache-size",type=int,default=48)
args=p.parse_args()
OUT=ROOT/"results/ranker_audit_20261003_v1/batch_optimization/larger_batch_trials"
rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device("cuda",local)
dist.init_process_group("nccl")
dest=OUT/args.tag
if rank==0:dest.mkdir(exist_ok=False)
dist.barrier(device_ids=[local])
snapshot=torch.load(OUT/"paused_disk_state.pt",weights_only=True,map_location="cpu")
data=BatchRankingDataset(ROOT/"data/ranker_cobs_train_full_32k_20261003",cache_size=args.cache_size)
manifest=json.loads((OUT/args.scene_manifest).read_text());indices=manifest["indices"]
assert len(set(indices))==len(indices)
cfg=PairSamplingConfig();config=BlockRankerConfig(**snapshot["config"])
def fresh():
    model=BlockRanker(config).to(device);model.load_state_dict(snapshot["model_state_dict"])
    return model
def check_normalization():
    # Same fixed pair sets vs a single global-batch reference.
    reference=fresh() if rank==0 else None
    model=DDP(DistributedBatchRanker(fresh()),device_ids=[local],broadcast_buffers=False)
    checks=[]
    for case in ["unequal_valid_counts","empty_tail_rank","all_no_pairs"]:
        global_ids=indices[:32 if case!="empty_tail_rank" else 3]
        batch_ids=global_batch_for_rank(global_ids,0,8,rank)
        scenes=[data[i] for i in batch_ids]
        if case=="all_no_pairs" or (case=="unequal_valid_counts" and rank==0):
            scenes=[dict(s,fixed_mass=torch.tensor(.99),teacher_mass=s["teacher_mass"]*.01/s["teacher_mass"].sum()) for s in scenes]
        model.zero_grad(set_to_none=True)
        scores,offsets=model(scenes)
        gen=torch.Generator().manual_seed(91+rank)
        pairs=sample_batch_pairs(scores.detach().cpu(),scenes,offsets,cfg,gen) if scenes else []
        scaled,mean,count,local_count=distributed_scene_loss(scores,scenes,offsets,pairs)
        scaled.backward()
        payload=[None]*world
        dist.all_gather_object(payload,{"ids":batch_ids,"pairs":[x.tolist() for x in pairs],"count":local_count})
        if rank==0:
            reference.zero_grad(set_to_none=True);all_scenes=[];all_pairs=[]
            for r,item in enumerate(payload):
                part=[data[i] for i in item["ids"]]
                if case=="all_no_pairs" or (case=="unequal_valid_counts" and r==0):
                    part=[dict(s,fixed_mass=torch.tensor(.99),teacher_mass=s["teacher_mass"]*.01/s["teacher_mass"].sum()) for s in part]
                all_scenes.extend(part)
                all_pairs.extend(torch.tensor(x,dtype=torch.long).reshape(-1,2) for x in item["pairs"])
            packed=pack_scenes(all_scenes,device);ref_scores=forward_packed(reference,packed)
            ref_loss,per=loss_from_pairs(ref_scores,all_scenes,packed["offsets"],all_pairs)
            if ref_loss is None:ref_loss=ref_scores.sum()*0.
            ref_loss.backward()
            delta=max(float((p.grad-q.grad).abs().max()) for p,q in zip(reference.parameters(),model.module.ranker.parameters()))
            assert abs(float(ref_loss)-mean)<2e-5,(case,float(ref_loss),mean)
            assert delta<3e-4,(case,delta)
            checks.append({"case":case,"local_effective_counts":[x["count"] for x in payload],"global_effective_count":count,"loss_difference":abs(float(ref_loss)-mean),"max_gradient_difference":delta})
        dist.barrier(device_ids=[local])
    del model,reference
    return checks
checks=check_normalization()
model=DDP(DistributedBatchRanker(fresh()),device_ids=[local],broadcast_buffers=False)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);opt.load_state_dict(snapshot["optimizer_state_dict"])
# Explicit independent deterministic streams: not the old serial RNG trajectory.
gen=torch.Generator().manual_seed(43+rank)
if rank==0:
    val=RankingDataset(ROOT/"data/ranker_cobs_validation16_20261003",cache_size=48)
    before=evaluate_batched(model.module.ranker,val,device,max_scenes=512,batch_size=32)
model.train()
dist.barrier(device_ids=[local])
t=time.perf_counter()
for i in manifest["shard_indices"]:data[data.ends[i-1] if i else 0]
preload=time.perf_counter()-t
global_batch=args.local_batch*world
stats=collections.defaultdict(float);processed=[]
dist.barrier(device_ids=[local]);torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
for begin in range(0,len(indices),global_batch):
    local_ids=global_batch_for_rank(indices,begin,args.local_batch,rank);processed.extend(local_ids)
    scenes=[data[i] for i in local_ids]
    opt.zero_grad(set_to_none=True)
    t=time.perf_counter();scores,offsets=model(scenes);cpu=scores.detach().cpu()
    stats["forward_transfer_seconds"]+=time.perf_counter()-t
    t=time.perf_counter()
    pairs=sample_batch_pairs(cpu,scenes,offsets,cfg,gen) if scenes else []
    stats["cpu_sampling_seconds"]+=time.perf_counter()-t
    scaled,mean,count,n=distributed_scene_loss(scores,scenes,offsets,pairs)
    scaled.backward()
    if count:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();stats["steps"]+=1
    stats["local_effective_scenes"]+=n;stats["local_pairs"]+=sum(len(x) for x in pairs)
    stats["global_loss_sum"]+=mean*count;stats["global_effective_count_accumulated"]+=count
torch.cuda.synchronize(device);dist.barrier(device_ids=[local]);elapsed=time.perf_counter()-start
train_peak=torch.cuda.max_memory_allocated(device)/2**20
all_ids=[None]*world;dist.all_gather_object(all_ids,processed)
if rank==0:
    flat=[i for block in all_ids for i in block]
    assert len(flat)==len(set(flat))==len(indices) and set(flat)==set(indices)
# Worst-case candidate batch on all ranks. No optimizer update.
model.zero_grad(set_to_none=True)
long_ids=manifest["long_candidate_indices"][-global_batch:]
local_ids=global_batch_for_rank(long_ids,0,args.local_batch,rank)
scenes=[data[i] for i in local_ids]
torch.cuda.reset_peak_memory_stats(device);dist.barrier(device_ids=[local]);t=time.perf_counter()
scores,offsets=model(scenes)
pairs=sample_batch_pairs(scores.detach().cpu(),scenes,offsets,cfg,torch.Generator().manual_seed(901+rank)) if scenes else []
scaled,_,count,_=distributed_scene_loss(scores,scenes,offsets,pairs);scaled.backward()
torch.cuda.synchronize(device)
stress={"local_scenes":len(scenes),"candidate_min":min(len(x["block_ids"]) for x in scenes) if scenes else 0,
        "candidate_max":max(len(x["block_ids"]) for x in scenes) if scenes else 0,
        "peak_allocated_mib":torch.cuda.max_memory_allocated(device)/2**20,"seconds":time.perf_counter()-t}
import hashlib
h=hashlib.sha256()
for param in model.module.ranker.parameters():h.update(param.detach().cpu().numpy().tobytes())
state_hashes=[None]*world;dist.all_gather_object(state_hashes,h.hexdigest())
assert len(set(state_hashes))==1,state_hashes
per_rank=[None]*world
dist.all_gather_object(per_rank,{"rank":rank,"elapsed_seconds":elapsed,"preload_seconds":preload,"peak_allocated_mib":train_peak,"stats":dict(stats),"long_candidate_stress":stress})
if rank==0:
    after=evaluate_batched(model.module.ranker,val,device,max_scenes=512,batch_size=32)
    report={"world_size":world,"local_batch":args.local_batch,"global_batch":global_batch,
        "scenes":len(indices),"steps":int(stats["steps"]),"seconds":max(x["elapsed_seconds"] for x in per_rank),
        "scenes_per_second":len(indices)/max(x["elapsed_seconds"] for x in per_rank),
        "mean_train_loss":stats["global_loss_sum"]/stats["global_effective_count_accumulated"],
        "per_rank":per_rank,"normalization_checks":checks,"unique_scene_coverage_verified":True,"rank_model_hashes_identical":True,"scene_manifest":args.scene_manifest,"source_bucket_counts":manifest["source_bucket_counts"],
        "rng":"CPU pair generator seed=43+rank; deterministic within configuration, NOT the original single-stream trajectory",
        "gradient_normalization":"DDP averages world gradients; local SUM(scene-normalized losses) * world / GLOBAL effective scene count",
        "validation_before":before,"validation_after":after,
        "note":"Short experiment only; same starting model/optimizer/lr/data. Global batch changes optimizer count. Old batch8 process remains stopped on GPU0 with about718MiB."}
    (dest/"result.json").write_text(json.dumps(report,indent=2))
    save_ranker(dest/"trial_final.pt",model.module.ranker,data.metadata,{"trial_only":True,"world_size":world,"global_batch":global_batch,"scenes":len(indices),"rng_seeds":[43+r for r in range(world)]})
    print(json.dumps({k:v for k,v in report.items() if k not in ["validation_before","validation_after","per_rank"]}),flush=True)
dist.barrier(device_ids=[local]);dist.destroy_process_group()
