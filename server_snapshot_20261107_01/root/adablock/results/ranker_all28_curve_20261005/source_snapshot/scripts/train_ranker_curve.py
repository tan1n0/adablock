"""Balanced all-layer 5k/20k/50k curve; resumable, no epoch-based extension."""
import argparse,datetime,json,os,random,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,save_ranker
from utils.ranking_data import assert_disjoint,check_metadata
from utils.ranking_curve import MMapDataset,SelectedDataset,merge_evaluations
from utils.ranking_curve_eval import evaluate_batched
from utils.ranking_batch import sample_batch_pairs
from utils.ranking_ddp import DistributedBatchRanker,distributed_scene_loss
from utils.ranking_loss import PairSamplingConfig
def main():
 p=argparse.ArgumentParser();p.add_argument("--work",required=True);p.add_argument("--out",required=True);p.add_argument("--resume",action="store_true")
 p.add_argument("--targets",nargs="+",type=int,default=[5000,20000,50000]);p.add_argument("--stop-after",type=int);p.add_argument("--test",action="store_true")
 a=p.parse_args();rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"]);assert world==4
 os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";torch.use_deterministic_algorithms(True)
 torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device("cuda",local)
 torch.manual_seed(42+rank);random.seed(42+rank)
 dist.init_process_group("nccl",timeout=datetime.timedelta(hours=3))
 def barrier():dist.barrier(device_ids=[local])
 out=Path(a.out);work=Path(a.work)
 if rank==0:
  if not a.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
  out.mkdir(parents=True,exist_ok=True)
 barrier()
 audit=json.loads((work/"plan_audit.json").read_text())
 plan=np.load(work/"train_plan.npy",mmap_mode="r");vi=np.load(work/"validation_indices.npy")
 if a.test:vi=vi[::256] # 112 fixed scenes spanning every shard, tests only.
 train=MMapDataset(ROOT/"data/ranker_cobs_train_all28_20261004");val=MMapDataset(ROOT/"data/ranker_cobs_validation_all28_20261004",True)
 assert_disjoint(train,val);check_metadata(train.metadata,val.metadata)
 for key,ds in [("train",train),("validation",val)]:
  assert hashlib.sha256((ds.directory/"manifest.json").read_bytes()).hexdigest()==audit["manifest_hashes"][key]
 assert hashlib.sha256((work/"train_plan.npy").read_bytes()).hexdigest()==audit["plan_sha256"]
 config=BlockRankerConfig(train.metadata["head_dim"],28,4);base=BlockRanker(config).to(device)
 initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
 saved=torch.load(out/"resume_latest.pt",weights_only=True,map_location="cpu") if a.resume else None
 if saved:
  assert saved["plan_sha256"]==audit["plan_sha256"] and saved["targets"]==a.targets and saved["test"]==a.test
  assert saved["world_size"]==4 and saved["local_batch"]==16 and saved["lr"]==2e-4
  base.load_state_dict(saved["model"])
 model=DDP(DistributedBatchRanker(base),device_ids=[local],broadcast_buffers=False)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
 gen=torch.Generator().manual_seed(43+rank)
 step=saved["step"] if saved else 0;batch=saved["next_batch"] if saved else 0
 history=saved["history"] if saved else [];seconds=saved["train_seconds"] if saved else 0.
 if saved:
  opt.load_state_dict(saved["optimizer"]);rng=saved["rng"][rank]
  torch.set_rng_state(rng["cpu"]);torch.cuda.set_rng_state(rng["cuda"],device);gen.set_state(rng["pairs"]);random.setstate(rng["python"])
 def emit(row):
  if rank==0:
   print(json.dumps(row),flush=True);tmp=out/"progress.tmp";tmp.write_text(json.dumps(row,indent=2));tmp.replace(out/"progress.json")
 def checkpoint(stage,milestone=False):
  rng={"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state(device),"pairs":gen.get_state(),"python":random.getstate()}
  allrng=[None]*4;dist.all_gather_object(allrng,rng)
  if rank==0:
   payload={"format":"all28_curve_resume_v1","stage":stage,"model":base.state_dict(),"optimizer":opt.state_dict(),"rng":allrng,
            "step":step,"next_batch":batch,"train_seconds":seconds,"history":history,"plan_sha256":audit["plan_sha256"],
            "targets":a.targets,"test":a.test,"world_size":4,"local_batch":16,"lr":2e-4,"seed":42,"config":config.__dict__,
            "data_metadata":train.metadata,"manifest_hashes":audit["manifest_hashes"],"sampling_config":PairSamplingConfig().__dict__}
   tmp=out/"resume_latest.tmp"
   with tmp.open("wb") as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
   tmp.replace(out/"resume_latest.pt")
   if milestone:
    target=out/f"step_{step:06d}_resume.pt"
    if not target.exists():os.link(out/"resume_latest.pt",target)
    save_ranker(out/f"step_{step:06d}.pt",base,train.metadata,{"step":step,"seed":42,"lr":2e-4,"global_batch":64,"plan_sha256":audit["plan_sha256"]})
  barrier()
 def evaluate(indices,label):
  emit({"stage":"validation","step":step,"label":label,"scenes":len(indices)})
  local_indices=(np.array_split(indices,4)[rank] if a.test else indices[np.searchsorted(np.asarray(val.ends),indices,side="right")%4==rank])
  result=evaluate_batched(base,SelectedDataset(val,local_indices),device,batch_size=16)
  parts=[None]*4;dist.all_gather_object(parts,result);result=merge_evaluations(parts);result["dataset_scenes"]=len(val)
  if rank==0:(out/f"{label}.json").write_text(json.dumps(result,indent=2))
  barrier();return result
 def finish_point():
  nonlocal history
  if step in a.targets and not any(x["step"]==step for x in history):
   checkpoint("validation_pending",True)
   result=evaluate(vi,f"validation_step_{step:06d}")
   learned=result["groups"]["overall"]["methods"]["learned"];metric=sum(learned[f"coverage@{k}"] for k in result["budgets"])/len(result["budgets"])
   deltas=(base.layer_embedding.weight.detach().cpu()-initial["layer_embedding.weight"]).abs().sum(-1)
   moments=opt.state[base.layer_embedding.weight]["exp_avg"].detach().abs().sum(-1)
   assert bool((deltas>0).all()) and bool((moments>0).all())
   row={"step":step,"batches":batch,"train_seconds":seconds,"mean_coverage":metric,"validation_seconds":result["evaluation_seconds"],
        "methods":result["groups"]["overall"]["methods"],"all28_embedding_rows_changed":True,"all28_optimizer_moments_nonzero":True}
   history.append(row)
   if rank==0:
    (out/"history.json").write_text(json.dumps(history,indent=2))
    if not a.test:
     with (ROOT/"docs/ranker_full_training_20261003.md").open("a") as f:
      f.write("\n### All28 balanced curve: step "+str(step)+"\n\n"+json.dumps(row,ensure_ascii=False)+"\n")
   checkpoint("validated");emit({"stage":"point_complete",**row})
 finish_point();model.train();loss_sum=0.;count_sum=0
 while step<max(a.targets):
  if a.stop_after and step>=a.stop_after:break
  if batch>=len(plan):raise RuntimeError("Frozen sample plan exhausted")
  start=time.perf_counter();ids=plan[batch].reshape(4,4,4).transpose(1,0,2).reshape(64)[rank*16:(rank+1)*16];scenes=[train[int(i)] for i in ids]
  opt.zero_grad(set_to_none=True);scores,offsets=model(scenes)
  assert bool(torch.isfinite(scores).all())
  pairs=sample_batch_pairs(scores.detach().cpu(),scenes,offsets,PairSamplingConfig(coverage=.95),gen)
  loss,mean,count,_=distributed_scene_loss(scores,scenes,offsets,pairs)
  if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
  loss.backward()
  if count:
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
  batch+=1;torch.cuda.synchronize();seconds+=time.perf_counter()-start;loss_sum+=mean*count;count_sum+=count
  if batch%100==0:
   emit({"stage":"training","step":step,"batches":batch,"scenes":batch*64,"train_seconds":seconds,
         "scenes_per_second":batch*64/max(seconds,1e-9),"loss":loss_sum/max(count_sum,1),"remaining_train_seconds":(max(a.targets)-step)*seconds/max(step,1)})
  if batch%500==0:checkpoint("training")
  finish_point();model.train()
 checkpoint("training_finished",step in a.targets)
 if step<max(a.targets):
  if rank==0:(out/"BOUNDED_COMPLETE").write_text(str(step))
  dist.destroy_process_group();return
 best=max(history,key=lambda x:x["mean_coverage"])
 if rank==0:(out/"selection.json").write_text(json.dumps({"best_step":best["step"],"criterion":"fixed-subset macro mean coverage over six budgets","history":history},indent=2))
 if not a.test:
  bestckpt=torch.load(out/f"step_{best['step']:06d}.pt",weights_only=True,map_location="cpu");base.load_state_dict(bestckpt["state_dict"])
  full=evaluate(np.arange(len(val)),f"full_validation_best_{best['step']:06d}")
  if rank==0:
   save_ranker(out/"best.pt",base,train.metadata,{"step":best["step"],"selection":"independent fixed validation subset","full_validation":True})
   with (ROOT/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n### All28 curve complete\n\nBest step "+str(best["step"])+"; full validation: "+json.dumps(full["groups"]["overall"])+"\n")
 if rank==0:(out/"COMPLETE").write_text(json.dumps({"step":step,"best_step":best["step"],"train_seconds":seconds}))
 barrier();dist.destroy_process_group()
if __name__=="__main__":main()
