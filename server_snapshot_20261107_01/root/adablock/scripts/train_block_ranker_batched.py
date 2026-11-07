"""Optional batched executor; same model, batch8 defaults, samples and objective."""
from __future__ import annotations
import sys,json,random,time
from pathlib import Path
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.train_block_ranker import parse_args
from models.block_ranker import BlockRanker,BlockRankerConfig,save_ranker
from utils.ranking_data import RankingDataset,ShardShuffleSampler,assert_disjoint
from utils.ranking_batch import BatchRankingDataset,batch_loss
from utils.ranking_batch_eval import evaluate_batched
from utils.ranking_loss import PairSamplingConfig

def train(args):
    if min(args.epochs,args.batch_size,args.lr)<=0:raise ValueError("Invalid training sizes/lr")
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    train_data=BatchRankingDataset(args.train_dir)
    val_data=RankingDataset(args.val_dir)
    assert_disjoint(train_data,val_data)
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError(str(output))
    output.mkdir(parents=True,exist_ok=True)
    (output/"run_config.json").write_text(json.dumps(vars(args),indent=2))
    meta=train_data.metadata
    config=BlockRankerConfig(meta["head_dim"],meta["num_layers"],meta["num_kv_heads"],
                            projection_size=args.projection_size,hidden_size=args.hidden_size,
                            use_position=not args.no_position,identity_size=args.identity_size,
                            block_chunk_size=args.block_chunk_size)
    sampling=PairSamplingConfig(args.coverage,args.max_positives,args.hard_negatives,
                                args.random_negatives,args.positive_pairs,args.pair_delta)
    sampling.validate()
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model=BlockRanker(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    generator=torch.Generator().manual_seed(args.seed)
    pair_generator=torch.Generator().manual_seed(args.seed+1)
    loader=DataLoader(train_data,batch_size=args.batch_size,
                     sampler=ShardShuffleSampler(train_data,generator),collate_fn=list)
    history=[];best=-float("inf")
    training_meta={"sampling":asdict(sampling),"seed":args.seed,"lr":args.lr,
                   "budgets":args.eval_budgets,"train_scenes":len(train_data),
                   "validation_scenes":len(val_data),"parameters":sum(p.numel() for p in model.parameters()),
                   "executor":"ragged_batch_v1","batch_size":args.batch_size,
                   "cpu_threads":args.cpu_threads,"equivalence":"Numerical, not bitwise; pairwise score offsets are invariant."}
    run_start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();total_loss=0.;total_pairs=valid_scenes=seen=steps=0
        start=time.perf_counter();last_log=start;last_checkpoint=0
        for batch in loader:
            if args.max_train_scenes is not None:
                batch=batch[:max(0,args.max_train_scenes-seen)]
                if not batch:break
            seen+=len(batch);optimizer.zero_grad(set_to_none=True)
            loss,per_scene,pairs=batch_loss(model,batch,device,sampling,pair_generator)
            if loss is not None:
                if not bool(loss.isfinite()):raise FloatingPointError("Nonfinite rank loss")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();steps+=1
                total_loss+=float(per_scene.detach().sum());valid_scenes+=len(per_scene)
                total_pairs+=sum(len(p) for p in pairs)
            now=time.perf_counter()
            if now-last_log>=30 or seen==len(train_data):
                row={"stage":"training","epoch":epoch+1,"seen":seen,"total_scenes":len(train_data),
                     "valid_scenes":valid_scenes,"optimizer_steps":steps,"pairs":total_pairs,
                     "train_loss":total_loss/max(valid_scenes,1),"epoch_seconds":now-start,
                     "scenes_per_second":seen/(now-start),
                     "output_bias":float(model.output.bias.detach())}
                (output/"progress.json").write_text(json.dumps(row,indent=2))
                print(json.dumps(row),flush=True);last_log=now
            if seen-last_checkpoint>=100000:
                # Recoverable progress artifact, explicitly not an epoch-selected model.
                state={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                       "epoch":epoch+1,"seen":seen,"pair_generator_state":pair_generator.get_state(),
                       "sampler_generator_state_after_order":generator.get_state(),
                       "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),
                       "config":asdict(config),"data_metadata":meta,
                       "note":"Mid-epoch recovery state; automatic resume not implemented."}
                torch.save(state,output/"progress_state.tmp");(output/"progress_state.tmp").replace(output/"progress_state.pt")
                last_checkpoint=seen
            if args.max_train_scenes is not None and seen>=args.max_train_scenes:break
        if not valid_scenes:raise ValueError("No effective ranking pairs")
        training_seconds=time.perf_counter()-start
        validation=evaluate_batched(model,val_data,device,args.eval_budgets,args.coverage,args.max_val_scenes,args.batch_size)
        learned=validation["groups"]["overall"]["methods"]["learned"]
        criterion=sum(learned[f"coverage@{k}"] for k in validation["budgets"])/len(validation["budgets"])
        row={"epoch":epoch+1,"train_loss":total_loss/valid_scenes,"scenes":seen,
             "valid_scenes":valid_scenes,"pairs":total_pairs,"optimizer_steps":steps,
             "training_seconds":training_seconds,"mean_validation_coverage":criterion,"validation":validation}
        history.append(row)
        save_ranker(output/"last.pt",model,meta,dict(training_meta,epoch=epoch+1))
        if criterion>best:
            best=criterion;save_ranker(output/"best.pt",model,meta,dict(training_meta,epoch=epoch+1))
        (output/"history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!="validation"}),flush=True)
    (output/"TRAINING_COMPLETE").write_text(json.dumps({"epochs":args.epochs,"seconds":time.perf_counter()-run_start}))
    return history

if __name__=="__main__":train(parse_args())
