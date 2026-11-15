"""Frozen-ranker coverage-target comparison; all outputs belong to this run."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
QA = Path('/root/adablock/results/llama_qa_budget_20261015_01')
PILOT = Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
sys.path.insert(0, str(ROOT/'vendor'))
import run
import runtime as rt
import budget_pilot as bp

SEED = 20261019
Q = .95
EPOCHS = 40
N_QA = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def branch(rho):
    return ROOT/f'rho{round(100*rho)}'


def labels(coverage, rho):
    hit = coverage >= rho-1e-7
    assert hit.any(-1).all()
    return hit.long().argmax(-1)


def prepare():
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    sys.path.insert(0, '/root/adablock')
    from scripts.build_strict_training_corpus import sampled_ngram_hashes, LongBenchDecontaminator
    assert not (ROOT/'protocol.json').exists()
    sources = [QA/'budget_data'/f'{s}.pt' for s in ['budget_train', 'validation']]
    rankhash = sha(PILOT/'ranker/best.pt')
    source_data = [torch.load(p, map_location='cpu', weights_only=True) for p in sources]
    for d in source_data:
        assert d['ranker_sha256'] == rankhash
        assert (d['coverage'].diff(dim=-1) >= -1e-7).all()
    for rho in [.90, .95]:
        out = branch(rho)
        out.mkdir()
        (out/'budget_data').mkdir()
        for split, data in zip(['budget_train', 'validation'], source_data):
            data = dict(data)
            data['labels'] = labels(data['coverage'], rho)
            data['rho'] = rho
            # Historical exact-count fields were collected at .95, not valid .90 supervision.
            data.pop('required_exact', None)
            data.pop('oracle_sort_required_exact', None)
            torch.save(data, out/'budget_data'/f'{split}.pt')
    assert torch.all(labels(source_data[0]['coverage'], .90) <= labels(source_data[0]['coverage'], .95))
    assert torch.equal(labels(source_data[0]['coverage'], .95), source_data[0]['labels'])

    previous = json.loads((QA/'selected.json').read_text())
    blocked_fp = set()
    questions = set()
    blocked_ids = set()
    for r in previous:
        blocked_fp.update(sampled_ngram_hashes(r['context'], 13, 5))
        questions.add(r['question'].strip().lower())
    for split in ['rank_train', 'budget_train', 'validation', 'test']:
        for line in (PILOT/f'{split}.jsonl').read_text().splitlines():
            r = json.loads(line)
            blocked_fp.update(sampled_ngram_hashes(r['text'], 13, 5))
    for p in Path('/root/adablock/data/longbench').glob('*.jsonl'):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            questions.add(r.get('input', '').strip().lower())
            if r.get('_id'):
                blocked_ids.add(r['_id'])
    guard = LongBenchDecontaminator(Path('/root/adablock/data/longbench'), 13, 5, 3)
    tok = AutoTokenizer.from_pretrained(run.MODEL, local_files_only=True)
    prompt = json.loads((QA/'protocol.json').read_text())['prompt_template']
    raw_files = sorted(Path('/root/adablock/data/hotpot_official').glob('*.parquet'))
    assert len(raw_files) == 2
    metadata = []
    # Seeded random order of source rows; no answer or model-output based selection.
    for p in raw_files:
        rows = pq.read_table(p).to_pylist()
        metadata.extend((str(p), i, r) for i, r in enumerate(rows))
    random.Random(SEED).shuffle(metadata)
    selected, excluded, selected_fp = [], Counter(), set()
    for filename, index, r in metadata:
        if r['id'] in blocked_ids or r['question'].strip().lower() in questions:
            excluded['previous_question'] += 1
            continue
        context = '\n\n'.join(title+'\n'+''.join(sentences)
                              for title, sentences in zip(r['context']['title'], r['context']['sentences']))
        fp = set(sampled_ngram_hashes(context, 13, 5))
        if len(fp & blocked_fp) >= 3 or guard.contaminated(context):
            excluded['prior_context_overlap'] += 1
            continue
        if len(fp & selected_fp) >= 3:
            excluded['within_qa_context_overlap'] += 1
            continue
        ids = tok.apply_chat_template([dict(role='user', content=prompt.format(context=context, question=r['question']))],
                                      tokenize=True, add_generation_prompt=True)
        if not 2048 <= len(ids) <= 8160:
            excluded['outside_2k_8k'] += 1
            continue
        selected.append(dict(source_index=len(selected), source_id=r['id'], source_file=filename,
                             source_row=index, question=r['question'], answers=[r['answer']],
                             input_ids=ids, prompt_tokens=len(ids), truncated=False,
                             input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                             context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                             type=r['type'], level=r['level']))
        selected_fp.update(fp)
        if len(selected) == N_QA:
            break
    assert len(selected) == N_QA, len(selected)
    dump(ROOT/'qa_selected.json', selected)
    protocol = dict(seed=SEED, rho=[.90,.95], frozen_ranker=str(PILOT/'ranker/best.pt'),
                    ranker_sha256=rankhash, budget_data={str(p):sha(p) for p in sources},
                    initial_budget='identical seeded random state; same original BudgetClassifier',
                    objective='original unweighted CE; no new cost penalties', epochs=EPOCHS,
                    batch_size=512, lr=.001, weight_decay=.01, gradient_clip=1.,
                    model_selection='minimum existing held-out WikiHop calibration CE; no QA answer selection',
                    normalization='shared budget-train features mean/std; std floor .001; input clip [-8,8]',
                    deployment_quantile=Q, calibration='fixed equal q=.95 for both branches; no quantile scan',
                    qa=dict(dataset='official HotpotQA TRAIN distractor', n=N_QA,
                            selection='seeded random source order; 2048..8160 tokens; no truncation; frozen before training',
                            exclusions=dict(excluded), old_questions_excluded=True,
                            context_overlap_filter='>=3 sampled 13-word matches to budget/pilot corpus, LongBench, or chosen QA',
                            files={str(p):sha(p) for p in raw_files},
                            prompt_tokens=[r['prompt_tokens'] for r in selected],
                            types=dict(Counter(r['type'] for r in selected)),
                            role='new development QA validation; never used to select checkpoint or thresholds'),
                    model=run.MODEL, dtype='BF16', max_new_tokens=32, decoding='greedy; original EOS set',
                    shared_prefill=True, attention='full SDPA prefill; original sparse selected attention from second output token',
                    block_size=16, sink_tokens=32, local_blocks=32,
                    practical_quality_tolerance_f1=1.0,
                    decision='report paired F1 delta/CI and both costs; no threshold changes after viewing QA',
                    limitations=['32-question pilot; no noninferiority claim without sufficient paired precision',
                                 'token slots are proxies, not measured GPU memory traffic'])
    dump(ROOT/'protocol.json', protocol)
    dump(ROOT/'status.json', dict(stage='prepared'))
    print(json.dumps(dict(stage='prepared', n=len(selected), lengths=protocol['qa']['prompt_tokens'])), flush=True)


def train():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    states = []
    for rho in [.90, .95]:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        out = branch(rho)
        assert not (out/'best.pt').exists()
        data, val = [torch.load(out/'budget_data'/f'{s}.pt', map_location='cpu', weights_only=True)
                     for s in ['budget_train', 'validation']]
        model = bp.BudgetClassifier(data['features'].shape[1]).cuda()
        model.mean.copy_(data['features'].mean(0))
        model.std.copy_(data['features'].std(0, unbiased=False).clamp_min(.001))
        initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if states:
            assert all(torch.equal(v, states[0][k]) for k,v in initial.items())
        states.append(initial)
        torch.save(initial, out/'initial_state.pt')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        x,y = data['features'].cuda(),data['labels'].cuda()
        vx,vy = val['features'].cuda(),val['labels'].cuda()
        history,best,steps = [],float('inf'),0
        start = time.perf_counter()
        for epoch in range(EPOCHS):
            model.train()
            total = 0.
            for indices in torch.randperm(len(y), device='cuda').split(512):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[indices]), y[indices])
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(indices)
                steps += 1
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(b) for b in vx.split(1024)])
                ce = float(F.cross_entropy(logits, vy))
            row = dict(rho=rho,epoch=epoch+1,train_ce=total/len(y),validation_ce=ce,steps=steps)
            history.append(row)
            if ce < best:
                best=ce
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch+1, rho=rho,
                                config=dict(feature_size=x.shape[1])),out/'best.pt')
            print(json.dumps(row),flush=True)
        torch.cuda.synchronize()
        dump(out/'training.json', dict(history=history,seconds=time.perf_counter()-start,
                                     steps=steps,selected_epoch=torch.load(out/'best.pt',weights_only=True)['epoch']))
        del model, optimizer, x,y,vx,vy
        torch.cuda.empty_cache()
    assert sha(PILOT/'ranker/best.pt') == json.loads((ROOT/'protocol.json').read_text())['ranker_sha256']
    dump(ROOT/'status.json',dict(stage='trained',initial_states_equal=True,ranker_unchanged=True))


class CostController(rt.AdaptiveLlama):
    def install(self):
        self.costs=[]
        super().install()

    def finish_step(self):
        for row in self.step_rows:
            lengths=row['selected_tokens']
            self.costs.append(dict(read=sum(lengths),padded=len(lengths)*max(lengths),
                                   full=len(lengths)*(self.position+1)))
        super().finish_step()


def profile(model,pairs,logits,ranker,budget,digests,eos):
    timings=defaultdict(list)
    originals={}
    for name in ['forward_packed','online_features','selected_attention']:
        original=getattr(rt,name)
        originals[name]=original
        def wrapped(*args,_name=name,_fn=original,**kwargs):
            run.sync();start=time.perf_counter()
            value=_fn(*args,**kwargs)
            run.sync();timings[_name].append(time.perf_counter()-start)
            return value
        setattr(rt,name,wrapped)
    def before(module,args):
        run.sync();module._profile_start=time.perf_counter()
    def after(module,args,output):
        run.sync();timings['budget_forward'].append(time.perf_counter()-module._profile_start)
    hooks=[budget.register_forward_pre_hook(before),budget.register_forward_hook(after)]
    try:
        result=run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,
                          quantile=Q,max_new=8,ignore_eos=True)
    finally:
        for name,original in originals.items():setattr(rt,name,original)
        for hook in hooks:hook.remove()
    return dict(decode_steps=result['decode_steps'],
                components={k:dict(seconds=sum(v),calls=len(v),seconds_per_step=sum(v)/result['decode_steps'])
                            for k,v in timings.items()},
                note='Separate seven-step instrumented diagnostic; synchronization overhead; not main decode latency')


@torch.inference_mode()
def evaluate(shard):
    torch.set_num_threads(4)
    run.AdaptiveLlama=CostController
    model,tok,ranker,oldbudget,eos,_=run.load()
    del oldbudget
    ranker.requires_grad_(False)
    budgets={}
    for rho in [.90,.95]:
        payload=torch.load(branch(rho)/'best.pt',map_location='cpu',weights_only=True)
        b=bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval()
        b.load_state_dict(payload['state_dict']);b.requires_grad_(False)
        budgets[f'rho{round(rho*100)}']=b
    rows=json.loads((ROOT/'qa_selected.json').read_text())
    run.prefill(model,rows[0]['input_ids'][:128])
    # Warm both exact execution paths and predictors; warm-up is excluded from QA metrics.
    pairs,logits,_=run.prefill(model,rows[0]['input_ids'][:1024])
    digests=rt.build_prefix_digests(pairs)
    run.decode(model,pairs,logits,'full',ranker,None,None,eos,max_new=4,ignore_eos=True)
    for b in budgets.values():
        run.decode(model,pairs,logits,'adaptive',ranker,b,digests,eos,quantile=Q,max_new=4,ignore_eos=True)
    # The all-token custom sparse path must agree with native full attention.
    full=run.decode(model,pairs,logits,'full',ranker,None,None,eos,max_new=6,ignore_eos=True)
    all_read=run.decode(model,pairs,logits,'all',ranker,None,None,eos,max_new=6,ignore_eos=True)
    assert full['ids']==all_read['ids']
    dump(ROOT/f'preflight_{shard}.json',dict(all_token_parity=True,ragged_error=rt.ragged_test(),
                                          gpu=torch.cuda.get_device_name()))
    del pairs,logits,digests
    with (ROOT/f'qa_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            pairs,logits,prefill_seconds=run.prefill(model,row['input_ids'])
            run.sync();start=time.perf_counter()
            digests=rt.build_prefix_digests(pairs)
            run.sync();index_seconds=time.perf_counter()-start
            names=['full','rho90','rho95']
            # Rotate execution order to balance systematic warm/cache effects.
            names=names[i%3:]+names[:i%3]
            record=dict(source_index=i,source_id=row['source_id'],prompt_tokens=row['prompt_tokens'],
                        input_ids_sha256=row['input_ids_sha256'],prefill_seconds=prefill_seconds,
                        index_seconds=index_seconds,order=names,methods={})
            for name in names:
                result=run.decode(model,pairs,logits,'full' if name=='full' else 'adaptive',ranker,
                                  budgets.get(name),None if name=='full' else digests,eos,quantile=Q,capture=True)
                controller=result.pop('_controller')
                if controller:
                    costs=controller.costs
                    read=sum(c['read'] for c in costs);padded=sum(c['padded'] for c in costs)
                    denom=sum(c['full'] for c in costs)
                    result.update(logical_slots=read,padded_slots=padded,full_slots=denom,
                                  logical_fraction=read/max(1,denom),padded_fraction=padded/max(1,denom))
                    assert all(v==result['decode_steps'] for v in result['audit']['ranker_calls'])
                    assert all(v==result['decode_steps'] for v in result['audit']['budget_calls'])
                else:
                    denom=sum((len(row['input_ids'])+s+1)*32*8 for s in range(result['decode_steps']))
                    result.update(logical_slots=denom,padded_slots=denom,full_slots=denom,
                                  logical_fraction=1. if denom else 0.,padded_fraction=1. if denom else 0.)
                result['prediction']=tok.decode(result['ids'],skip_special_tokens=True).strip()
                result['f1']=100*max(run.token_f1(result['prediction'],a) for a in row['answers'])
                result['e2e_seconds']=prefill_seconds+result['decode_seconds']+(index_seconds if name!='full' else 0)
                result.pop('steps');record['methods'][name]=result
                del controller
            journal.write(json.dumps(record,ensure_ascii=False)+'\n');journal.flush()
            print(json.dumps(dict(source_index=i,scores={k:v['f1'] for k,v in record['methods'].items()})),flush=True)
            if i==shard:
                diagnostic={name:profile(model,pairs,logits,ranker,b,digests,eos) for name,b in budgets.items()}
                dump(ROOT/f'profile_{shard}.json',diagnostic)
            del pairs,logits,digests,record
            gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


def summarize():
    rows=[json.loads(line) for p in sorted(ROOT.glob('qa_*.jsonl')) for line in p.read_text().splitlines()]
    assert len(rows)==N_QA and len({r['source_id'] for r in rows})==N_QA
    rows.sort(key=lambda r:r['source_index'])
    metrics=['f1','logical_fraction','padded_fraction','logical_slots','padded_slots',
             'decode_seconds','decode_steps','seconds_per_decode_step','e2e_seconds','mean_middle_blocks']
    summary={name:{m:float(np.mean([r['methods'][name][m] for r in rows])) for m in metrics}
             for name in ['full','rho90','rho95']}
    rng=np.random.default_rng(SEED)
    delta=np.array([r['methods']['rho90']['f1']-r['methods']['rho95']['f1'] for r in rows])
    boot=delta[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(1)
    against_full={}
    for name in ['rho90','rho95']:
        diff=np.array([r['methods'][name]['f1']-r['methods']['full']['f1'] for r in rows])
        samples=diff[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(1)
        against_full[name]=dict(f1_delta=float(diff.mean()),bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
                                wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum()))
    profiles=[json.loads(p.read_text()) for p in sorted(ROOT.glob('profile_*.json'))]
    report=dict(n=len(rows),summary=summary,
                paired_90_minus_95=dict(f1_delta=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                                       wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),
                                       relative_read_fraction_reduction=1-summary['rho90']['logical_fraction']/summary['rho95']['logical_fraction'],
                                       relative_padded_fraction_reduction=1-summary['rho90']['padded_fraction']/summary['rho95']['padded_fraction']),
                paired_against_full=against_full,
                profile_diagnostics=profiles,protocol_sha256=sha(ROOT/'protocol.json'),
                frozen_ranker_unchanged=sha(PILOT/'ranker/best.pt')==json.loads((ROOT/'protocol.json').read_text())['ranker_sha256'],
                limitations=['new QA validation, not final benchmark; n=32',
                             'same model and settings; different policies have different generated states and lengths',
                             'no post-QA threshold/checkpoint selection; slot reductions are not measured speedups'])
    assert report['frozen_ranker_unchanged']
    dump(ROOT/'comparison.json',report)
    dump(ROOT/'status.json',dict(stage='complete',questions=len(rows)))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def self_test():
    coverage=torch.tensor([[.8,.88,.92,.96,1.],[.96,.97,.98,.99,1.]])
    assert labels(coverage,.90).tolist()==[2,0]
    assert labels(coverage,.95).tolist()==[3,0]
    assert bp.quantile_actions(torch.tensor([[.94,0.,.06]]),.95).item()==2
    assert rt.ragged_test()<2e-7
    print('PASS: coverage labels, fixed-only labels, deployment quantile, masked selected attention')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','evaluate','summarize','self_test'])
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.stage=='evaluate':evaluate(args.shard)
    else:globals()[args.stage]()
