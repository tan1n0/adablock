"""Full-data DDP ranker training with exact global-position checkpointing."""
from __future__ import annotations
import argparse,copy,datetime,hashlib,itertools,json,os,random,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,save_ranker
from utils.ranking_data import RankingDataset,ShardShuffleSampler,assert_disjoint,check_metadata
from utils.ranking_batch import BatchRankingDataset,sample_batch_pairs
from utils.ranking_ddp import DistributedBatchRanker,distributed_scene_loss,global_batch_for_rank
from utils.ranking_loss import PairSamplingConfig
from utils.ranking_batch_eval import evaluate_batched

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir",required=True);p.add_argument("--val-dir",required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--resume");p.add_argument("--trace-indices",action="store_true")
    p.add_argument("--epochs",type=int,default=10);p.add_argument("--local-batch",type=int,default=16)
    p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--coverage",type=float,default=.95);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--checkpoint-every-scenes",type=int,default=100000)
    p.add_argument("--log-every-batches",type=int,default=500)
    p.add_argument("--max-run-scenes",type=int,help="Test-only bounded run; saves resumable state then exits.")
    return p.parse_args()

def main(args):
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    if min(args.epochs,args.local_batch,args.lr,args.checkpoint_every_scenes,args.log_every_batches)<=0:raise ValueError("Invalid configuration")
    torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    torch.manual_seed(args.seed+rank);random.seed(args.seed+rank)
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=30))
    def barrier():dist.barrier(device_ids=[local])
    out=Path(args.output_dir)
    if rank==0:
        if out.exists() and any(out.iterdir()):raise FileExistsError(str(out))
        out.mkdir(parents=True,exist_ok=True)
    barrier()
    train=BatchRankingDataset(args.train_dir,cache_size=2);val=RankingDataset(args.val_dir)
    assert_disjoint(train,val)
    meta=train.metadata;global_batch=args.local_batch*world
    manifest_hashes={name:hashlib.sha256((Path(path)/"manifest.json").read_bytes()).hexdigest()
                     for name,path in [("train",args.train_dir),("validation",args.val_dir)]}
    config=BlockRankerConfig(meta["head_dim"],meta["num_layers"],meta["num_kv_heads"])
    saved=torch.load(args.resume,weights_only=True,map_location="cpu") if args.resume else None
    if saved:
        if saved["format"]!="adablock_ddp_resume_v1":raise ValueError("Only this DDP executor's resumable state is supported; legacy serial state is not silently imported.")
        assert saved["world_size"]==world and saved["local_batch"]==args.local_batch
        assert saved["manifest_hashes"]==manifest_hashes
        for key in ["seed","lr","weight_decay","coverage","epochs"]:
            assert saved["args"][key]==getattr(args,key),(key,saved["args"][key],getattr(args,key))
        check_metadata(saved["data_metadata"],meta)
        assert saved["config"]==config.__dict__
    base=BlockRanker(config).to(device)
    if saved:base.load_state_dict(saved["model_state_dict"])
    model=DDP(DistributedBatchRanker(base),device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    if saved:optimizer.load_state_dict(saved["optimizer_state_dict"])
    pair_generator=torch.Generator().manual_seed(args.seed+1+rank)
    order_generator=torch.Generator().manual_seed(args.seed)
    epoch=saved["epoch"] if saved else 1
    offset=saved["next_scene_offset"] if saved else 0
    epoch_start_state=saved["epoch_start_sampler_rng"] if saved else order_generator.get_state()
    history=saved["history"] if saved else []
    best=saved["best_metric"] if saved else -float("inf")
    best_state=saved["best_model_state_dict"] if saved else None
    best_epoch=saved["best_epoch"] if saved else None
    stats=saved["epoch_stats"] if saved else None
    trace=saved.get("trace_indices",[]) if saved else []
    if saved:
        rng=saved["rank_rng_states"][rank]
        torch.set_rng_state(rng["torch_cpu"]);torch.cuda.set_rng_state(rng["torch_cuda"],device)
        pair_generator.set_state(rng["pair"]);random.setstate(rng["python"])
    def state_hash(module):
        h=hashlib.sha256()
        for p in module.parameters():h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    provenance={"initialization":"DDP_checkpoint_resume" if saved else "random_initialization",
                "resume_path":args.resume,"initial_model_sha256":state_hash(base),
                "world_size":world,"local_batch":args.local_batch,"global_batch":global_batch,
                "pair_rng_seeds":[args.seed+1+r for r in range(world)],
                "loss_normalization":"SUM of scene-normalized losses / GLOBAL effective scene count",
                "data_order":"Original shard-shuffle generator, seed42 by default; global stream sliced without padding/repeats.",
                "manifest_hashes":manifest_hashes,"scheduler":None}
    training_meta={"executor":"ddp_v1",**provenance,"seed":args.seed,"lr":args.lr,
                   "train_scenes":len(train),"validation_scenes":len(val),"budgets":[4,8,16,32,64,128]}
    if rank==0:
        (out/"run_config.json").write_text(json.dumps({**vars(args),**provenance},indent=2))
        (out/"history.json").write_text(json.dumps(history,indent=2))
        if best_state is not None:
            temp=copy.deepcopy(base).cpu();temp.load_state_dict(best_state)
            save_ranker(out/"best.pt",temp,meta,dict(training_meta,epoch=best_epoch))
        print(json.dumps({"stage":"initialized",**provenance,"epoch":epoch,"next_scene_offset":offset}),flush=True)
    sampling=PairSamplingConfig(coverage=args.coverage)
    run_seen=0;run_start=time.perf_counter()
    while epoch<=args.epochs:
        model.train()
        if stats is None:
            stats={"loss_sum":0.,"valid_scenes":0,"pairs":0,"optimizer_steps":0,"scenes":0,"active_train_seconds":0.}
        assert stats["scenes"]==offset,(stats,offset)
        local_pairs=0;segment_start=time.perf_counter();last_checkpoint=offset
        order_generator.set_state(epoch_start_state)
        iterator=iter(ShardShuffleSampler(train,order_generator))
        # Skip INDEX generation only, never load or retrain consumed samples.
        for _ in itertools.islice(iterator,offset):pass
        def snapshot_stats():
            counter=torch.tensor(local_pairs,device=device,dtype=torch.long)
            dist.all_reduce(counter,op=dist.ReduceOp.SUM)
            return dict(stats,pairs=stats["pairs"]+int(counter.item()),
                        active_train_seconds=stats["active_train_seconds"]+time.perf_counter()-segment_start)
        def checkpoint(stage,checkpoint_epoch,checkpoint_offset,sampler_state,statistics):
            local_rng={"torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state(device),
                       "pair":pair_generator.get_state(),"python":random.getstate()}
            all_rng=[None]*world;dist.all_gather_object(all_rng,local_rng)
            if rank==0:
                payload={"format":"adablock_ddp_resume_v1","stage":stage,
                         "model_state_dict":base.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                         "scheduler_state_dict":None,"config":config.__dict__,"data_metadata":meta,
                         "args":vars(args),"world_size":world,"local_batch":args.local_batch,
                         "manifest_hashes":manifest_hashes,"epoch":checkpoint_epoch,
                         "next_scene_offset":checkpoint_offset,"epoch_start_sampler_rng":sampler_state,
                         "rank_rng_states":all_rng,"epoch_stats":statistics,"history":history,
                         "best_metric":best,"best_model_state_dict":best_state,"best_epoch":best_epoch,
                         "provenance":provenance,"trace_indices":trace if args.trace_indices else []}
                torch.save(payload,out/"resume_latest.tmp");(out/"resume_latest.tmp").replace(out/"resume_latest.pt")
            barrier()
        while offset<len(train):
            take=global_batch
            if args.max_run_scenes is not None:
                take=min(take,args.max_run_scenes-run_seen)
                if take<=0:break
            global_ids=list(itertools.islice(iterator,take))
            if not global_ids:raise RuntimeError("Sampler ended before manifest scene count")
            local_ids=global_batch_for_rank(global_ids,0,args.local_batch,rank)
            scenes=[train[i] for i in local_ids]
            optimizer.zero_grad(set_to_none=True)
            scores,offsets=model(scenes)
            pairs=sample_batch_pairs(scores.detach().cpu(),scenes,offsets,sampling,pair_generator) if scenes else []
            loss,mean,count,n=distributed_scene_loss(scores,scenes,offsets,pairs)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite ranking loss")
            loss.backward()  # Empty ranks/all-empty batches still join collectives.
            if count:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();stats["optimizer_steps"]+=1
            local_pairs+=sum(len(x) for x in pairs)
            stats["loss_sum"]+=mean*count;stats["valid_scenes"]+=count
            offset+=len(global_ids);stats["scenes"]+=len(global_ids);run_seen+=len(global_ids)
            if rank==0 and args.trace_indices:trace.extend([[epoch,i] for i in global_ids])
            event=(offset==len(train) or (offset//global_batch)%args.log_every_batches==0)
            if event:
                s=snapshot_stats()
                if rank==0:
                    row={"stage":"training","epoch":epoch,"seen":offset,"total_scenes":len(train),
                         "optimizer_steps":s["optimizer_steps"],"valid_scenes":s["valid_scenes"],
                         "pairs":s["pairs"],"train_loss":s["loss_sum"]/max(s["valid_scenes"],1),
                         "epoch_seconds":s["active_train_seconds"],
                         "scenes_per_second":offset/max(s["active_train_seconds"],1e-9),
                         "global_batch":global_batch,"output_bias":float(base.output.bias.detach())}
                    tmp=out/"progress.tmp";tmp.write_text(json.dumps(row,indent=2));tmp.replace(out/"progress.json")
                    print(json.dumps(row),flush=True)
            stop=args.max_run_scenes is not None and run_seen>=args.max_run_scenes
            if offset-last_checkpoint>=args.checkpoint_every_scenes or stop:
                s=snapshot_stats();checkpoint("training",epoch,offset,epoch_start_state,s);last_checkpoint=offset
            if stop:
                if rank==0 and args.trace_indices:(out/"trace_indices.json").write_text(json.dumps(trace))
                if rank==0:(out/"BOUNDED_RUN_COMPLETE").write_text(json.dumps({"epoch":epoch,"next_scene_offset":offset,"processed_this_run":run_seen}))
                dist.destroy_process_group();return
        if offset!=len(train):raise RuntimeError("Unexplained incomplete epoch")
        finished=snapshot_stats()
        checkpoint("validation_pending",epoch,offset,epoch_start_state,finished)
        if rank==0:
            print(json.dumps({"stage":"validation","epoch":epoch}),flush=True)
            validation=evaluate_batched(base,val,device,coverage=args.coverage,batch_size=32)
            learned=validation["groups"]["overall"]["methods"]["learned"]
            criterion=sum(learned[f"coverage@{k}"] for k in validation["budgets"])/len(validation["budgets"])
            row={"epoch":epoch,"scenes":offset,"train_loss":finished["loss_sum"]/max(1,finished["valid_scenes"]),
                 "valid_scenes":finished["valid_scenes"],"pairs":finished["pairs"],
                 "optimizer_steps":finished["optimizer_steps"],"training_seconds":finished["active_train_seconds"],
                 "mean_validation_coverage":criterion,"validation":validation}
            history.append(row)
            save_ranker(out/"last.pt",base,meta,dict(training_meta,epoch=epoch))
            if criterion>best:
                best=criterion;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
                save_ranker(out/"best.pt",base,meta,dict(training_meta,epoch=epoch))
            (out/"history.json").write_text(json.dumps(history,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!="validation"}),flush=True)
        barrier()
        # History/best model only need to be saved on rank0; scalar synchronizes logs.
        best_tensor=torch.tensor([best if rank==0 else 0.],device=device,dtype=torch.float64)
        dist.broadcast(best_tensor,src=0);best=float(best_tensor.item())
        epoch+=1;offset=0;stats=None
        # Iterator may not have been asked for StopIteration, but all randperm
        # draws for the completed epoch have already happened.
        epoch_start_state=order_generator.get_state()
        checkpoint("epoch_complete",epoch,0,epoch_start_state,
                   {"loss_sum":0.,"valid_scenes":0,"pairs":0,"optimizer_steps":0,"scenes":0,"active_train_seconds":0.})
    if rank==0:
        if args.trace_indices:(out/"trace_indices.json").write_text(json.dumps(trace))
        (out/"TRAINING_COMPLETE").write_text(json.dumps({"epochs":args.epochs,"seconds_this_run":time.perf_counter()-run_start}))
    barrier();dist.destroy_process_group()

if __name__=="__main__":main(parse_args())
