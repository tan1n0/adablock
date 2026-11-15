"""Shared frozen data, original networks, CE budget updates and bounded rank pairs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parent
PILOT=Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
QA=Path('/root/adablock/results/llama_qa_budget_20261015_01')
PREVIOUS=Path('/root/adablock_budget_rho_20261019_01')
sys.path.insert(0,str(ROOT/'vendor'))
import run
import runtime as rt
import budget_pilot as bp
from models.block_ranker import load_ranker, BlockRanker, BlockRankerConfig
from utils.ranking_batch import pack_scenes, forward_packed, loss_from_pairs, sample_batch_pairs
from utils.ranking_loss import PairSamplingConfig

SEED=20261019
RHO=.90
Q=.95
CONFIG=dict(seed=SEED,deterministic_training=True,rho=RHO,quantile=Q,rounds=2,rank_groups_per_step=8,
            rank_positions_per_layer=4,rank_lr=.0002,budget_lr=.001,
            budget_batch=512,budget_epochs_per_round=20,weight_decay=.01,
            boundary_weight=1.,boundary_pairs=32,total_pair_cap=192,
            normalization='frozen shared initial budget-train statistics',
            objective='unweighted budget cross entropy, identical for S/A/B',
            checkpoint='minimum calibration CE in final 20 budget epochs, final ranker fixed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def initial_models(device):
    ranker,payload=load_ranker(PILOT/'ranker/best.pt',device)
    b=torch.load(PREVIOUS/'rho90/best.pt',map_location='cpu',weights_only=True)
    budget=bp.BudgetClassifier(b['config']['feature_size']).to(device)
    budget.load_state_dict(b['state_dict'])
    return ranker,budget,payload


def load_pair(path,device):
    p=torch.load(path,map_location='cpu',weights_only=True)
    ranker=BlockRanker(BlockRankerConfig(**p['ranker_config'])).to(device)
    ranker.load_state_dict(p['ranker_state']);ranker.eval()
    budget=bp.BudgetClassifier(588).to(device)
    budget.load_state_dict(p['budget_state']);budget.eval()
    return ranker,budget,p


def save_pair(path,ranker,budget,initial_payload,info):
    torch.save(dict(ranker_config=initial_payload['config'],
                    ranker_state=ranker.state_dict(),budget_state=budget.state_dict(),
                    ranker_state_sha256=state_digest(ranker),budget_state_sha256=state_digest(budget),
                    info=info,config=CONFIG),path)


def rank_groups():
    folder=PILOT/'data/rank_train'
    manifest=json.loads((folder/'manifest.json').read_text())
    result=[];selection=[]
    for entry in manifest['shards']:
        shard=torch.load(folder/entry['file'],map_location='cpu',weights_only=True)
        count=len(shard['records'])
        indices=torch.linspace(0,count-1,min(4,count)).round().long().unique().tolist()
        d=shard['digests']
        for ri in indices:
            r=shard['records'][ri];ids=r['block_ids']
            assert ((ids+1)*16<=r['token_position']).all()
            select=slice(int(ids[0]),int(ids[-1])+1)
            assert torch.equal(ids,torch.arange(int(ids[0]),int(ids[-1])+1))
            group=[]
            for h in range(8):
                group.append(dict(query=r['query'][h],mean=d['mean'][h,select],
                                  basis=d['covariance_basis'][h,select],
                                  eigenvalues=d['covariance_eigenvalues'][h,select],
                                  counts=d['counts'][select],block_ids=ids,block_size=16,
                                  teacher_mass=r['teacher_mass'][h],fixed_mass=r['fixed_mass'][h],
                                  token_position=r['token_position'],layer_id=entry['layer_id'],
                                  kv_head_id=h,document_index=entry['document_index']))
            result.append(group)
            selection.append([entry['file'],ri,r['token_position']])
    assert len(result)==1536,len(result)
    return result,selection


def qa_groups(split):
    groups=[]
    for path in sorted((ROOT/'raw_qa').glob(split+'_*.pt')):
        p=torch.load(path,map_location='cpu',weights_only=True)
        for trajectory in ['full','old_adaptive']:
            digest=p['digests'][trajectory]
            for r in p['rows'][trajectory]:
                ids=r['block_ids'];d=digest[r['layer_id']]
                assert ((ids+1)*16<=r['token_position']).all()
                select=slice(int(ids[0]),int(ids[-1])+1)
                assert torch.equal(ids,torch.arange(int(ids[0]),int(ids[-1])+1))
                group=[]
                for h in range(8):
                    group.append(dict(query=r['query'][h],mean=d['mean'][h,select],
                                      basis=d['basis'][h,select],eigenvalues=d['eigenvalues'][h,select],
                                      counts=d['counts'][select],block_ids=ids,block_size=16,
                                      teacher_mass=r['teacher_mass'][h],fixed_mass=r['fixed_mass'][h],
                                      token_position=r['token_position'],layer_id=r['layer_id'],
                                      kv_head_id=h,document_index=p['document']['document'],trajectory=trajectory))
                groups.append(group)
    assert groups
    return groups


@torch.no_grad()
def supervision(ranker,groups,device):
    ranker.eval()
    features=[];labels=[];coverage=[];actual=[];identities=[]
    for start in range(0,len(groups),8):
        scenes=[s for g in groups[start:start+8] for s in g]
        packed=pack_scenes(scenes,device)
        scores=forward_packed(ranker,packed).float().cpu()
        for j,s in enumerate(scenes):
            score=scores[packed['offsets'][j]:packed['offsets'][j+1]]
            label,cov,cost,_=bp.labels_for_order(s['teacher_mass'],s['fixed_mass'],score,rho=RHO)
            features.append(bp.online_features(s,score));labels.append(label)
            coverage.append(cov);actual.append(cost)
            identities.append([s['document_index'],s['layer_id'],s['kv_head_id'],s['token_position'],len(score),
                               int(s.get('trajectory')=='old_adaptive')])
    return dict(features=torch.stack(features),labels=torch.tensor(labels),coverage=torch.stack(coverage),
                actual_blocks=torch.stack(actual),identities=torch.tensor(identities),rho=RHO,
                ranker_state_sha256=state_digest(ranker))


def boundary_pairs(scores,mass,k,limit=32):
    n=len(scores)
    if k<=0 or k>=n:return torch.empty((0,2),dtype=torch.long)
    order=scores.detach().cpu().argsort(descending=True,stable=True)
    mass=mass.detach().float().cpu()
    inside=order[:k];outside=order[k:]
    low=inside[mass[inside].argsort(stable=True)[:8]]
    high=outside[mass[outside].argsort(descending=True,stable=True)[:8]]
    pairs=[(i,j) for i in high.tolist() for j in low.tolist() if mass[i]-mass[j]>1e-6]
    pairs.sort(key=lambda pair:(-float(mass[pair[0]]-mass[pair[1]]),pair))
    return torch.tensor(pairs[:limit],dtype=torch.long).reshape(-1,2)


@torch.no_grad()
def frozen_actions(ranker,budget,groups,device):
    data=supervision(ranker,groups,device)
    budget.eval()
    p=bp.probabilities(budget,data)
    actions=bp.quantile_actions(p,Q)
    return data['actual_blocks'][torch.arange(len(actions)),actions].reshape(-1,8),data


def rank_phase(ranker,budget,optimizer,groups,branch,phase,device):
    import time
    start=time.perf_counter()
    takes,_=frozen_actions(ranker,budget,groups,device)
    ranker.train()
    # Independent RNG streams keep S/A sampling identical despite intervening budget training.
    generator=torch.Generator().manual_seed(SEED+phase)
    order=torch.randperm(len(groups),generator=torch.Generator().manual_seed(SEED+100+phase))
    pair_config=PairSamplingConfig(coverage=RHO)
    counts=dict(original_pairs=0,boundary_pairs=0,steps=0,empty_batches=0)
    losses=[]
    for indices in order.split(CONFIG['rank_groups_per_step']):
        scenes=[s for idx in indices.tolist() for s in groups[idx]]
        packed=pack_scenes(scenes,device);scores=forward_packed(ranker,packed)
        detached=scores.detach().float().cpu()
        pairs=sample_batch_pairs(detached,scenes,packed['offsets'],pair_config,generator)
        base_loss,_=loss_from_pairs(scores,scenes,packed['offsets'],pairs)
        counts['original_pairs']+=sum(len(p) for p in pairs)
        extra=[]
        if branch=='B':
            for j,s in enumerate(scenes):
                gi=int(indices[j//8]);head=j%8
                p=boundary_pairs(detached[packed['offsets'][j]:packed['offsets'][j+1]],
                                 s['teacher_mass'],int(takes[gi,head]))
                assert len(p)+len(pairs[j])<=CONFIG['total_pair_cap']
                extra.append(p)
            boundary,_=loss_from_pairs(scores,scenes,packed['offsets'],extra)
        else:boundary=None
        counts['boundary_pairs']+=sum(len(p) for p in extra)
        terms=[v for v in [base_loss,None if boundary is None else CONFIG['boundary_weight']*boundary] if v is not None]
        optimizer.zero_grad(set_to_none=True)
        if terms:
            loss=sum(terms)
        else:
            loss=scores.sum()*0
            counts['empty_batches']+=1
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(ranker.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        counts['steps']+=1
    counts.update(phase='rank',round=phase,seconds=time.perf_counter()-start,
                  mean_loss=sum(losses)/max(1,len(losses)),frozen_K=takes)
    ranker.eval()
    return counts
