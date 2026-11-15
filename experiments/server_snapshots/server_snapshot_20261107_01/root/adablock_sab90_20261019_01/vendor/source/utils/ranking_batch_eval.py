"""Offline evaluation with batched learned scores and identical baseline scenes."""
from collections import defaultdict
import time
import torch
from utils.ranking_batch import pack_scenes, forward_packed
from utils.ranking_eval import baseline_scores

@torch.no_grad()
def evaluate_batched(model, dataset, device="cuda:0", budgets=(4,8,16,32,64,128),
                     coverage=.95, max_scenes=None, batch_size=8):
    budgets=sorted(set(map(int,budgets)))
    if not budgets or min(budgets)<0 or not 0<coverage<=1:
        raise ValueError("Invalid budgets/coverage")
    model.eval(); start=time.perf_counter()
    indices=range(len(dataset))
    if max_scenes is not None and max_scenes<len(dataset):
        if max_scenes<1: raise ValueError("max_scenes must be positive")
        indices=torch.linspace(0,len(dataset)-1,max_scenes).long().unique().tolist()
    accumulators={};counts=defaultdict(int)
    for begin in range(0,len(indices),batch_size):
        samples=[dataset[indices[i]] for i in range(begin,min(begin+batch_size,len(indices)))]
        p=pack_scenes(samples,device)
        learned=forward_packed(model,p).float().cpu()
        for i,s in enumerate(samples):
            mass=s["teacher_mass"].float();fixed=float(s["fixed_mass"])
            if not len(mass): continue
            scores=baseline_scores(s)
            scores["learned"]=learned[p["offsets"][i]:p["offsets"][i+1]]
            t=s["token_position"]
            length_bin=next((name for end,name in [(4096,"0-4K"),(8192,"4-8K"),(16384,"8-16K"),(32768,"16-32K")] if t<end),"32K+")
            groups=["overall",length_bin]
            if fixed<coverage-1e-7:
                groups+=["fixed_below_rho","fixed_below_rho/"+length_bin]
            oracle=mass.sort(descending=True).values.cumsum(0)
            per_method={}
            for name,score in scores.items():
                if not torch.isfinite(score).all(): raise FloatingPointError(name)
                curve=mass[score.argsort(descending=True,stable=True)].cumsum(0)
                total=fixed+curve
                unreachable=float(total[-1])<coverage-1e-7
                required=0 if fixed>=coverage-1e-7 else min(int((total<coverage-1e-7).sum())+1,len(mass))
                row={"required_k":required,"unreachable_rate":float(unreachable),
                     "over_max_budget_rate":float(required>max(budgets))}
                for k in budgets:
                    take=min(k,len(mass))
                    selected=float(curve[take-1]) if take else 0.
                    best=float(oracle[take-1]) if take else 0.
                    row.update({f"coverage@{k}":fixed+selected,f"middle_mass@{k}":selected,
                                f"mass_gap@{k}":max(0.,best-selected),f"actual_k@{k}":take,
                                f"coverage_target_hit@{k}":float(fixed+selected>=coverage-1e-7)})
                per_method[name]=row
            for group in groups:
                counts[group]+=1
                if group not in accumulators:accumulators[group]={name:defaultdict(float) for name in scores}
                for name,row in per_method.items():
                    for k,v in row.items():accumulators[group][name][k]+=v
    if not counts["overall"]: raise ValueError("No scenes")
    return {"scenes":counts["overall"],"dataset_scenes":len(dataset),"budgets":budgets,
            "coverage_target":coverage,"evaluation_seconds":time.perf_counter()-start,
            "gqa_baseline":"mean raw per-query scores (legacy repository)",
            "required_k_semantics":"Capped at candidate count if unreachable; unreachable_rate reported separately. Zero if fixed part already reaches rho.",
            "groups":{g:{"scenes":counts[g],"methods":{name:{k:v/counts[g] for k,v in row.items()} for name,row in methods.items()}}
                      for g,methods in accumulators.items()}}
