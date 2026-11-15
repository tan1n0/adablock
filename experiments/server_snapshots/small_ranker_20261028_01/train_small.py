"""Matched original pretraining and S/B schedules; only architecture changes."""
import os,sys,json,time,gc,argparse,importlib.util
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/adablock_sab90_20261019_01')
sys.path.insert(0,str(OLD))
import core as c
import torch
spec=importlib.util.spec_from_file_location('matched_train',OLD/'train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
ARCHES={'H64':(64,64),'P32H64':(32,64)}

def status(arch,stage,**extra):
    c.dump(ROOT/arch/'status.json',dict(stage=stage,**extra))

def main(arch):
    outroot=ROOT/arch;outroot.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    from scripts.train_block_ranker_batched import train as pretrain
    args=json.loads((c.PILOT/'ranker/run_config.json').read_text())
    args.update(output_dir=str(outroot/'pretrain'),projection_size=ARCHES[arch][0],hidden_size=ARCHES[arch][1])
    status(arch,'pretraining')
    pretrain(argparse.Namespace(**args))
    rank_groups,selection=c.rank_groups()
    c.dump(outroot/'rank_selection.json',selection)
    train_groups=c.qa_groups('budget_train');val_groups=c.qa_groups('validation')
    (outroot/'branches').mkdir(exist_ok=True)
    for branch in ['S','B']:
        status(arch,'adapting',branch=branch)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(c.SEED);torch.cuda.manual_seed_all(c.SEED)
        ranker,initial=c.load_ranker(outroot/'pretrain/best.pt','cuda:0')
        bp=torch.load(c.PREVIOUS/'rho90/best.pt',map_location='cpu',weights_only=True)
        budget=c.bp.BudgetClassifier(bp['config']['feature_size']).cuda();budget.load_state_dict(bp['state_dict'])
        out=outroot/'branches'/branch;out.mkdir(exist_ok=False)
        c.save_pair(out/'initial_pair.pt',ranker,budget,initial,dict(branch=branch,architecture=arch,phase='initial'))
        ro=torch.optim.AdamW(ranker.parameters(),lr=c.CONFIG['rank_lr'],weight_decay=c.CONFIG['weight_decay'])
        bo=torch.optim.AdamW(budget.parameters(),lr=c.CONFIG['budget_lr'],weight_decay=c.CONFIG['weight_decay'])
        schedule=[('rank',0),('rank',1),('budget',0),('budget',1)] if branch=='S' else [('rank',0),('budget',0),('rank',1),('budget',1)]
        history=[];started=time.perf_counter()
        for phase,round_id in schedule:
            status(arch,'adapting',branch=branch,phase=phase,round=round_id)
            if phase=='rank':
                ranker.requires_grad_(True)
                result=c.rank_phase(ranker,budget,ro,rank_groups,branch,round_id,'cuda:0')
                torch.save(result.pop('frozen_K'),out/f'actions_{round_id}.pt')
                c.save_pair(out/f'rank_{round_id}_pair.pt',ranker,budget,initial,result)
            else:
                result=t.budget_phase(ranker,budget,bo,train_groups,val_groups,round_id,out,initial)
                c.save_pair(out/f'budget_{round_id}_pair.pt',ranker,budget,initial,dict(phase=phase,round=round_id))
            history.append(result);c.dump(out/'history.json',history)
        elapsed=time.perf_counter()-started
        ranker,budget,best=c.load_pair(out/'best_pair.pt','cuda:0')
        metrics=t.offline_metrics(ranker,budget,val_groups)
        metrics.update(seconds=elapsed,checkpoint=best['info'],parameters=sum(p.numel() for p in ranker.parameters()),
            rank_steps=sum(r['steps'] for r in history if r['phase']=='rank'),budget_steps=sum(r['steps'] for r in history if r['phase']=='budget'))
        c.dump(out/'validation.json',metrics)
        print(json.dumps(dict(architecture=arch,branch=branch,stage='complete',seconds=elapsed,parameters=metrics['parameters'])),flush=True)
        del ranker,budget,ro,bo;gc.collect();torch.cuda.empty_cache()
    status(arch,'complete')

if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception:
        import traceback
        status(sys.argv[1],'failed',error=traceback.format_exc());raise
