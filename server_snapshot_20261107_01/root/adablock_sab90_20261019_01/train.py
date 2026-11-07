"""Matched S/A/B schedules; QA answers never enter training or checkpoint selection."""
import json
import os
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
import core as c


def budget_phase(ranker,budget,optimizer,train_groups,val_groups,phase,out,initial_payload):
    start=time.perf_counter()
    ranker.eval();ranker.requires_grad_(False)
    data=c.supervision(ranker,train_groups,'cuda:0')
    val=c.supervision(ranker,val_groups,'cuda:0')
    refresh_seconds=time.perf_counter()-start
    torch.save(dict(train=data,validation=val),out/f'supervision_{phase}.pt')
    budget.train()
    x,y=data['features'].cuda(),data['labels'].cuda()
    vx,vy=val['features'].cuda(),val['labels'].cuda()
    torch.manual_seed(c.SEED+1000+phase);torch.cuda.manual_seed_all(c.SEED+1000+phase)
    history=[];best=float('inf');steps=0
    for epoch in range(c.CONFIG['budget_epochs_per_round']):
        budget.train();total=0.
        for ids in torch.randperm(len(y),device='cuda').split(c.CONFIG['budget_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(budget(x[ids]),y[ids]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(budget.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(ids);steps+=1
        budget.eval()
        with torch.no_grad():
            logits=torch.cat([budget(b) for b in vx.split(1024)])
            ce=float(F.cross_entropy(logits,vy))
        row=dict(round=phase,epoch=epoch+1,total_epoch=phase*20+epoch+1,
                 train_ce=total/len(y),validation_ce=ce)
        history.append(row)
        # Every deployed branch has its final ranker and a checkpoint from its final budget stage.
        if phase==1 and ce<best:
            best=ce
            c.save_pair(out/'best_pair.pt',ranker,budget,initial_payload,dict(row,branch=out.name))
        print(json.dumps(dict(branch=out.name,phase='budget',**row)),flush=True)
    ranker.requires_grad_(True)
    budget.eval()
    return dict(phase='budget',round=phase,steps=steps,epochs=20,refresh_seconds=refresh_seconds,
                total_seconds=time.perf_counter()-start,history=history,
                label_histogram=torch.bincount(y.cpu(),minlength=9).tolist())


@torch.no_grad()
def offline_metrics(ranker,budget,groups):
    data=c.supervision(ranker,groups,'cuda:0')
    p=c.bp.probabilities(budget,data);a=c.bp.quantile_actions(p,c.Q)
    ix=torch.arange(len(a));cov=data['coverage'][ix,a];k=data['actual_blocks'][ix,a].float()
    ids=data['identities'];full=ids[:,3].float()+1
    fixed=full-16*ids[:,4]
    tokens=fixed+k*16
    documents={}
    for doc in ids[:,0].unique().tolist():
        mask=ids[:,0]==doc
        read=(tokens[mask]/full[mask]).reshape(-1,8)
        padded=read.max(-1).values
        oracle=data['actual_blocks'][ix[mask],data['labels'][mask]].float()
        documents[str(doc)]=dict(mean_coverage=float(cov[mask].mean()),hit90=float((cov[mask]>=.9-1e-7).float().mean()),
                                  logical_fraction=float(read.mean()),padded_fraction=float(padded.mean()),
                                  predicted_blocks=float(k[mask].mean()),label_blocks=float(oracle.mean()),
                                  fixed32_coverage=float(data['coverage'][mask,4].mean()))
    return dict(documents=documents,document_mean={key:sum(d[key] for d in documents.values())/len(documents)
                                                  for key in next(iter(documents.values()))})


def main(name):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(c.SEED);torch.cuda.manual_seed_all(c.SEED)
    out=c.ROOT/'branches'/name;out.mkdir(exist_ok=False)
    ranker,budget,initial=c.initial_models('cuda:0')
    initial_rank={k:v.cpu().clone() for k,v in ranker.state_dict().items()}
    c.save_pair(out/'initial_pair.pt',ranker,budget,initial,dict(branch=name,phase='initial'))
    rank_optimizer=torch.optim.AdamW(ranker.parameters(),lr=c.CONFIG['rank_lr'],weight_decay=.01)
    budget_optimizer=torch.optim.AdamW(budget.parameters(),lr=c.CONFIG['budget_lr'],weight_decay=.01)
    rank_groups,selection=c.rank_groups()
    c.dump(out/'rank_selection.json',selection)
    train_groups=c.qa_groups('budget_train');val_groups=c.qa_groups('validation')
    c.dump(out/'inputs.json',dict(rank_groups=len(rank_groups),budget_groups=len(train_groups),
                                 validation_groups=len(val_groups),qa_selection_sha256=c.sha(c.ROOT/'qa_selected.json')))
    schedule=[('rank',0),('rank',1),('budget',0),('budget',1)] if name=='S' else [('rank',0),('budget',0),('rank',1),('budget',1)]
    history=[];started=time.perf_counter()
    for phase,round_id in schedule:
        if phase=='rank':
            ranker.requires_grad_(True)
            result=c.rank_phase(ranker,budget,rank_optimizer,rank_groups,name,round_id,'cuda:0')
            actions=result.pop('frozen_K')
            torch.save(actions,out/f'actions_{round_id}.pt')
            c.save_pair(out/f'rank_{round_id}_pair.pt',ranker,budget,initial,result)
            print(json.dumps(dict(branch=name,**result)),flush=True)
        else:
            result=budget_phase(ranker,budget,budget_optimizer,train_groups,val_groups,round_id,out,initial)
            c.save_pair(out/f'budget_{round_id}_pair.pt',ranker,budget,initial,dict(phase='budget',round=round_id))
        history.append(result);c.dump(out/'history.json',history)
    elapsed=time.perf_counter()-started
    c.save_pair(out/'last_pair.pt',ranker,budget,initial,dict(branch=name,phase='last'))
    ranker,budget,best=c.load_pair(out/'best_pair.pt','cuda:0')
    metrics=offline_metrics(ranker,budget,val_groups)
    metrics.update(seconds=elapsed,checkpoint=best['info'],
                   rank_steps=sum(r['steps'] for r in history if r['phase']=='rank'),
                   budget_steps=sum(r['steps'] for r in history if r['phase']=='budget'),
                   original_pairs=sum(r['original_pairs'] for r in history if r['phase']=='rank'),
                   boundary_pairs=sum(r['boundary_pairs'] for r in history if r['phase']=='rank'))
    c.dump(out/'validation.json',metrics)
    print('COMPLETE',name,flush=True)


if __name__=='__main__':main(sys.argv[1])
