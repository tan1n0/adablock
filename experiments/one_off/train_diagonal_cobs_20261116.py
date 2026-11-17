"""Fit exact-diagonal COBS on the same real 16K/32K decode labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.cacheable_selectors import DiagonalCOBS
from experiments.one_off.train_value_selectors_20261116 import load_scenes,evaluate,losses,save_json,sha

BASE=ROOT/'results/value_aware_selector_20261116'

def run(args):
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    train,train_components,train_hashes=load_scenes(args.train_labels,device)
    valid,valid_components,valid_hashes=load_scenes(args.validation_labels,device)
    if train_components&valid_components:raise ValueError('Document component overlap')
    protocol={'train_files':train_hashes,'validation_files':valid_hashes,
              'train_scenes':len(train),'validation_scenes':len(valid),
              'seed':args.seed,'steps':args.steps,'eval_every':args.eval_every,
              'lr':args.lr,'model':'exact diagonal COBS, four learned coefficients',
              'label':'per-query-head conditional historical mass',
              'checkpoint_rule':'best independent validation output MSE'}
    args.output.mkdir(parents=True,exist_ok=True)
    pp=args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol mismatch')
    save_json(pp,protocol)
    result_path=args.output/'diag_cobs_result.json'
    if result_path.exists():print('skip complete');return
    model=DiagonalCOBS().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    initial=evaluate(model,valid,False)
    best=None;history=[]
    order=list(range(len(train)));random.Random(args.seed).shuffle(order)
    for step in range(1,args.steps+1):
        scene=train[order[(step-1)%len(order)]]
        model.train();optimizer.zero_grad(set_to_none=True)
        loss,_,_=losses(model,scene,False,0.)
        loss.backward();optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(model,valid,False)
            history.append({'step':step,'loss':float(loss),**metrics})
            if best is None or metrics['mean_output_mse']<best['mean_output_mse']:
                best={'step':step,**metrics,'coefficients':model.coefficients.detach().float().cpu().tolist()}
                dst=args.output/'diag_cobs_best.pt';tmp=dst.with_suffix('.tmp')
                torch.save({'model':'diag_cobs','state_dict':model.state_dict(),
                            'input_summary':'exact K population mean and diagonal variance',
                            'protocol':protocol},tmp);tmp.replace(dst)
    save_json(result_path,{'initial':initial,'best':best,'history':history,
                           'checkpoint_sha256':sha(args.output/'diag_cobs_best.pt')})
    print(json.dumps(best,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-labels',type=Path,default=BASE/'value_labels_full/train')
    parser.add_argument('--validation-labels',type=Path,default=BASE/'value_labels_full/validation')
    parser.add_argument('--output',type=Path,default=BASE/'diag_cobs_training')
    parser.add_argument('--steps',type=int,default=4096)
    parser.add_argument('--eval-every',type=int,default=512)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
