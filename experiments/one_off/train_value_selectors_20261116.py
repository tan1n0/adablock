"""Train cached dual encoders from real decode mass and context-tagged swaps.

Three controlled arms: K mass; K mass+value swaps; K+V mass+value swaps.
Each arm uses the same scenes and optimizer-step budget. Validation has
independent source documents and stores Full value contributions for output
error measurement. The frozen Llama model is never loaded during training.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.cacheable_selectors import DualEncoder,DualEncoderConfig,group_probabilities

DEFAULT=ROOT/'results/value_aware_selector_20261116'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)

def load_scenes(directory,device):
    scenes=[];components=set();files=sorted(Path(directory).glob('doc_*.pt'))
    if not files: raise FileNotFoundError(f'No completed document shards in {directory}')
    for path in files:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        components.update(c for p in payload['source_parts'] for c in p['component_doc_ids'])
        summary=payload['summaries']
        for record in payload['records']:
            layer,head=record['layer'],record['kv_head']
            scene={'summary':summary[layer][head].to(device).float(),
                   'query':record['query_scaled'].to(device).float(),
                   'mass_g':record['teacher_mass_g'].to(device).float(),
                   'fixed_g':record['fixed_mass_g'].to(device).float(),
                   'position':record['token_position'],
                   'swaps':record['swaps'],'initial':record['mass_selected'],
                   'full_contribution':record.get('contribution_g')}
            scenes.append(scene)
    return scenes,components,{str(p):sha(p) for p in files}

def logits_for(model,scene,use_value):
    summary=scene['summary'] if use_value else scene['summary'][:,:256]
    ids=torch.arange(len(summary),device=summary.device)
    counts=torch.full((len(summary),),16,device=summary.device)
    return model(scene['query'],summary,ids,counts,scene['position'])

def losses(model,scene,use_value,value_weight):
    logits=logits_for(model,scene,use_value)
    probabilities=logits.float().softmax(-1)
    target=scene['mass_g']/scene['mass_g'].sum(-1,keepdim=True).clamp_min(1e-8)
    mass_loss=-(target*probabilities.clamp_min(1e-9).log()).sum(-1).mean()
    swaps=scene['swaps']
    if not swaps or value_weight==0: return mass_loss,mass_loss,logits.new_zeros(())
    rank=group_probabilities(logits).clamp_min(1e-9).log()
    outgoing=torch.tensor([int(x[0]) for x in swaps],device=rank.device)
    incoming=torch.tensor([int(x[1]) for x in swaps],device=rank.device)
    gains=torch.tensor([float(x[2]) for x in swaps],device=rank.device)
    # All labels come from accepted swaps; their set context is in initial and
    # ordered swaps, while this light independent scorer cannot see the set.
    value_loss=(gains*F.softplus(rank[outgoing]-rank[incoming])).sum()/gains.sum().clamp_min(1e-12)
    return mass_loss+value_weight*value_loss,mass_loss,value_loss

@torch.no_grad()
def evaluate(model,scenes,use_value):
    model.eval();rows=[]
    for scene in scenes:
        logits=logits_for(model,scene,use_value)
        probabilities=logits.float().softmax(-1)
        group=probabilities.mean(0)
        mass=scene['mass_g'].mean(0)
        fixed=float(scene['fixed_g'].mean())
        ids=group.argsort(dim=-1,descending=True,stable=True)[:32]
        oracle=mass.argsort(dim=-1,descending=True,stable=True)[:32]
        target=scene['mass_g']/scene['mass_g'].sum(-1,keepdim=True).clamp_min(1e-8)
        pair_hits=[float(group[int(x[1])]>group[int(x[0])]) for x in scene['swaps']]
        row={'coverage':float(mass[ids].sum())+fixed,
             'ideal_coverage':float(mass[oracle].sum())+fixed,
             'distribution_kl':float((target*(target.clamp_min(1e-9).log()-probabilities.clamp_min(1e-9).log())).sum(-1).mean()),
             'pair_hits':sum(pair_hits),'pairs':len(pair_hits)}
        c=scene['full_contribution']
        if c is not None:
            c=c.to(logits.device).float()
            chosen=torch.cat((ids,ids.new_tensor([len(mass)])))
            retained_mass=scene['mass_g'][:,ids].sum(-1)+scene['fixed_g']
            output=c[:,chosen].sum(-2)/retained_mass.clamp_min(1e-8)[:,None]
            full=c.sum(-2)
            row['output_mse']=float((output-full).square().sum(-1).mean())
        rows.append(row)
    cover=sorted(x['coverage'] for x in rows)
    result={'scenes':len(rows),'mean_coverage_at_32':sum(cover)/len(cover),
            'p10_coverage_at_32':cover[max(0,int(.1*len(cover))-1)],
            'ideal_coverage_at_32':sum(x['ideal_coverage'] for x in rows)/len(rows),
            'mean_distribution_kl':sum(x['distribution_kl'] for x in rows)/len(rows),
            'swap_pair_fit':sum(x['pair_hits'] for x in rows)/max(1,sum(x['pairs'] for x in rows)),
            'swap_pairs':sum(x['pairs'] for x in rows)}
    if all('output_mse' in x for x in rows):
        result['mean_output_mse']=sum(x['output_mse'] for x in rows)/len(rows)
    return result

def train(args):
    torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    train_scenes,train_components,train_hashes=load_scenes(args.train_labels,device)
    valid_scenes,valid_components,valid_hashes=load_scenes(args.validation_labels,device)
    if train_components & valid_components: raise ValueError('Train/validation components overlap')
    if any(s['full_contribution'] is None for s in valid_scenes):
        raise ValueError('Validation contributions required for output-error selection')
    protocol={'train_files':train_hashes,'validation_files':valid_hashes,
              'train_scenes':len(train_scenes),'validation_scenes':len(valid_scenes),
              'train_validation_component_overlap':0,'steps':args.steps,'eval_every':args.eval_every,
              'seed':args.seed,'learning_rate':args.lr,'value_weight':args.value_weight,
              'embedding_dim':args.embedding_dim,'selection':'best independent validation output MSE',
              'model':'frozen Llama; only dual selector is optimized',
              'label_context':'initial mass-selected set and ordered accepted swaps retained per scene'}
    args.output.mkdir(parents=True,exist_ok=True)
    pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol: raise ValueError('Protocol mismatch')
    save_json(pp,protocol)
    arms=[('k_mass',False,0.),('k_value',False,args.value_weight),
          ('kv_value',True,args.value_weight)]
    for name,use_value,weight in arms:
        result_path=args.output/f'{name}_result.json'
        if result_path.exists(): print(name,'skip complete',flush=True);continue
        torch.manual_seed(args.seed)
        model=DualEncoder(DualEncoderConfig(head_dim=128,embedding_dim=args.embedding_dim,
                                             use_value=use_value)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
        initial=evaluate(model,valid_scenes,use_value)
        best=None;history=[]
        order=list(range(len(train_scenes)));random.Random(args.seed).shuffle(order)
        for step in range(1,args.steps+1):
            scene=train_scenes[order[(step-1)%len(order)]]
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,mass_loss,value_loss=losses(model,scene,use_value,weight)
            loss.backward();optimizer.step()
            if step%args.eval_every==0 or step==args.steps:
                metrics=evaluate(model,valid_scenes,use_value)
                history.append({'step':step,'train_loss':float(loss),
                                'train_mass_loss':float(mass_loss),
                                'train_value_loss':float(value_loss),**metrics})
                if best is None or metrics['mean_output_mse']<best['mean_output_mse']:
                    best={'step':step,**metrics}
                    checkpoint={'name':name,'config':asdict(model.config),'state_dict':model.state_dict(),
                                'input_summary':'exact K/V population mean and diagonal variance',
                                'protocol':protocol}
                    dst=args.output/f'{name}_best.pt';tmp=dst.with_suffix('.tmp')
                    torch.save(checkpoint,tmp);tmp.replace(dst)
        save_json(result_path,{'initial':initial,'best':best,'history':history,
                               'checkpoint_sha256':sha(args.output/f'{name}_best.pt')})
        print(name,best,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-labels',type=Path,default=DEFAULT/'value_labels_pilot_train')
    parser.add_argument('--validation-labels',type=Path,default=DEFAULT/'value_labels_pilot_validation')
    parser.add_argument('--output',type=Path,default=DEFAULT/'value_train_pilot')
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--eval-every',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--value-weight',type=float,default=0.5)
    parser.add_argument('--embedding-dim',type=int,default=32)
    parser.add_argument('--device',default='cuda')
    train(parser.parse_args())
