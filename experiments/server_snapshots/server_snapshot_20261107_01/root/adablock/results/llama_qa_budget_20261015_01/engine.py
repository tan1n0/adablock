"""Frozen ranker, real generated-state budget training; no HotpotQA training labels."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parent
PILOT=Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
BASE=Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
sys.path.insert(0,str(ROOT/'source'))
import run
import runtime as rt
import budget_pilot as bp
ORIGINAL_ATTENTION=rt.selected_attention
STEPS={0,1,3,7,15,23,30}

def dump(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def unused_legacy_prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(run.MODEL,local_files_only=True)
    rows=[]; used=set(); audits={}
    for split in ['rank_train','budget_train','validation','test']:
        source=[json.loads(s) for s in (PILOT/f'{split}.jsonl').read_text().splitlines()]
        components={c for r in source for c in r['component_doc_ids']}
        assert not used & components
        used.update(components)
        assert not any('hotpot' in c.lower() or 'longbench' in c.lower() for c in components)
        audits[split]=dict(documents=len(source),components=len(components),source_sha256=sha(PILOT/f'{split}.jsonl'))
        if split=='rank_train':continue
        for i,r in enumerate(source):
            # Use the same source documents and prefixes; reserve room for instructions and generation.
            material=tok.decode(r['pilot_input_ids'][1:r['pilot_max_tokens']-160],skip_special_tokens=True)
            prompt='Read the following material and summarize its main facts in two or three sentences.\n\nMaterial:\n'+material+'\n\nSummary:'
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
            assert 544<len(ids)<=8160
            rows.append(dict(split=split,document=i,doc_id=r['doc_id'],component_doc_ids=r['component_doc_ids'],
                source=r['source'],input_ids=ids,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                prompt_tokens=len(ids),source_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest()))
    dump('selected.json',rows)
    dump('protocol.json',dict(seed=20261015,documents_per_split=6,source_audit=audits,ranker_sha256=sha(PILOT/'ranker/best.pt'),
        old_budget_sha256=sha(PILOT/'budget/best.pt'),old_quantile=.95,trajectories=['full','old_adaptive'],sample_decode_steps=sorted(STEPS),
        max_new_tokens=32,prompt='Summarize main facts in two or three sentences; non-LongBench source material.',
        teacher_precision='FP32 QK and softmax; mean over GQA query heads',training='Same CE classifier, normalization, seed, optimizer, 40 epochs, val-CE checkpoint selection, validation-only quantile grid as original pilot.',
        changes='Generated states and chat summary prompts replace sampled teacher-forced raw-text positions; pooled full and old-adaptive trajectories.',
        scope='Small six-document training pilot; Hotpot50 is a previously inspected regression test, never a training or calibration set.',
        preserved=['original models','original results','ranker','feature definition','budget classes','coverage target']))


class Collector(rt.AdaptiveLlama):
    def install(self):
        self.records=[]; self.initial_position=self.position
        super().install();rt.selected_attention=self.attention

    def uninstall(self):
        rt.selected_attention=ORIGINAL_ATTENTION;super().uninstall()

    def choose(self,layer,query,scale):
        selected=super().choose(layer,query,scale)
        self.probe=None
        if self.position-self.initial_position not in STEPS:return selected
        history=(self.position+15)//16
        _,_,middle=rt.fixed_and_middle_blocks(history,2,32,history-1)
        ids=torch.tensor(sorted(middle),device=query.device,dtype=torch.long)
        h=self.model.config.num_key_value_heads
        grouped=query[0,:,0].float().view(h,-1,query.shape[-1])*scale
        dg=self.digests[layer]
        scenes=[dict(query=grouped[i],mean=dg['mean'][i,ids],basis=dg['basis'][i,ids],eigenvalues=dg['eigenvalues'][i,ids],
            counts=dg['counts'][ids],block_ids=ids,block_size=16,token_position=self.position,layer_id=layer,kv_head_id=i) for i in range(h)]
        scores=rt.forward_packed(self.ranker,rt.pack_scenes(scenes,query.device)).view(h,-1).float().cpu()
        features=torch.stack([rt.online_features(dict(s,query=s['query'].cpu()),scores[i]) for i,s in enumerate(scenes)])
        self.probe=(layer,ids,scores,features)
        return selected

    def attention(self,query,keys,values,indices,valid,scale):
        if self.probe is not None:
            layer,ids,scores,features=self.probe
            h=keys.shape[1]
            q=query[0,:,0].float().reshape(h,-1,query.shape[-1])
            mass=torch.einsum('hgd,htd->hgt',q,keys[0].float()).mul(scale).softmax(-1).mean(1)
            assert torch.allclose(mass.sum(-1),torch.ones(h,device=mass.device),atol=2e-6)
            blocks=torch.nn.functional.pad(mass,(0,(-mass.shape[-1])%16)).reshape(h,-1,16).sum(-1)
            candidate=blocks[:,ids].cpu();fixed=1-candidate.sum(-1)
            for head in range(h):
                label,coverage,actual,needed=bp.labels_for_order(candidate[head],fixed[head],scores[head])
                _,_,_,oracle=bp.labels_for_order(candidate[head],fixed[head],candidate[head])
                curve=torch.cat([fixed[head,None],fixed[head]+candidate[head,scores[head].argsort(descending=True,stable=True)].cumsum(0)])
                self.records.append(dict(features=features[head],label=label,coverage=coverage,actual=actual,required=needed,oracle=oracle,
                    identity=[0,0,layer,head,self.position,len(ids),float(fixed[head])],
                    fixed112_coverage=float(curve[min(112,len(ids))]),fixed112_blocks=min(112,len(ids))))
        return ORIGINAL_ATTENTION(query,keys,values,indices,valid,scale)


@torch.inference_mode()
def collect(shard):
    model,tok,ranker,budget,eos,config=run.load()
    rows=json.loads((ROOT/'selected.json').read_text())
    (ROOT/'collected').mkdir(exist_ok=True)
    run.prefill(model,rows[0]['input_ids'][:128])
    for number,row in enumerate(rows):
        if number%4!=shard:continue
        target=ROOT/'collected'/f"{row['split']}_{row['document']}.pt"
        assert not target.exists()
        pairs,logits,_=run.prefill(model,row['input_ids'])
        digests=rt.build_prefix_digests(pairs)
        records=[];generations={};parity={}
        for trajectory,mode in [('full','all'),('old_adaptive','adaptive')]:
            run.AdaptiveLlama=Collector
            try:
                result=run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,quantile=.95,capture=True)
            finally:run.AdaptiveLlama=rt.AdaptiveLlama
            controller=result.pop('_controller')
            assert controller.records
            for rec in controller.records:rec['trajectory']=trajectory;rec['identity'][1]=row['document']
            records.extend(controller.records)
            generations[trajectory]=dict(ids=result['ids'],text=tok.decode(result['ids'],skip_special_tokens=True),steps=result['decode_steps'],
                sampled_steps=len(controller.records)//256,mean_middle_blocks=result['mean_middle_blocks'])
            # Native full path parity on every document; sparse parity on first document per GPU.
            if trajectory=='full' or number<4:
                baseline=run.decode(model,pairs,logits,'full' if trajectory=='full' else 'adaptive',ranker,budget,digests,eos,quantile=.95)
                assert baseline['ids']==result['ids'],(row['doc_id'],trajectory)
                parity[trajectory]=True
            del controller,result
        torch.save(dict(records=records,document=row,generations=generations,parity=parity),target)
        print(json.dumps(dict(split=row['split'],document=row['document'],scenes=len(records),generations=generations,parity=parity)),flush=True)
        del pairs,logits,digests,records;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


def merge():
    (ROOT/'budget_data').mkdir()
    rankhash=sha(PILOT/'ranker/best.pt')
    for split in ['budget_train','validation','test']:
        expected=sum(r['split']==split for r in json.loads((ROOT/'selected.json').read_text()))
        files=sorted((ROOT/'collected').glob(split+'_*.pt'));assert len(files)==expected
        payloads=[torch.load(f,weights_only=True) for f in files]
        recs=[r for p in payloads for r in p['records']]
        for i,r in enumerate(recs):r['identity'][0]=i
        data=dict(features=torch.stack([r['features'] for r in recs]),labels=torch.tensor([r['label'] for r in recs]),
            coverage=torch.stack([r['coverage'] for r in recs]),actual_blocks=torch.stack([r['actual'] for r in recs]),
            identities=torch.tensor([r['identity'] for r in recs],dtype=torch.float64),required_exact=torch.tensor([r['required'] for r in recs]),
            oracle_sort_required_exact=torch.tensor([r['oracle'] for r in recs]),trajectory=torch.tensor([r['trajectory']=='old_adaptive' for r in recs]),
            fixed112_coverage=torch.tensor([r['fixed112_coverage'] for r in recs]),fixed112_blocks=torch.tensor([r['fixed112_blocks'] for r in recs]),
            documents=[p['document'] for p in payloads],ranker_sha256=rankhash,budgets=bp.BUDGETS,rho=.95,split=split,
            identity_columns=['scene','document','layer','kv_head','position','candidates','fixed_mass'])
        assert torch.isfinite(data['features']).all()
        assert torch.all(data['coverage'].diff(dim=1)>=-1e-7)
        assert torch.allclose(data['coverage'][:,-1],torch.ones(len(recs)),atol=2e-5)
        torch.save(data,ROOT/'budget_data'/f'{split}.pt')
        dump(f'{split}_data_summary.json',dict(scenes=len(recs),documents=expected,label_histogram=torch.bincount(data['labels'],minlength=9).tolist(),
            features=data['features'].shape[1],full_scenes=int((~data['trajectory']).sum()),old_adaptive_scenes=int(data['trajectory'].sum())))
    (ROOT/'ranker').mkdir();(ROOT/'ranker'/'best.pt').symlink_to(PILOT/'ranker/best.pt')


def train():
    bp.ROOT=ROOT
    bp.self_test();bp.train()


def evaluate():
    bp.ROOT=ROOT;torch.set_num_threads(4)
    new_payload=torch.load(ROOT/'budget/best.pt',weights_only=True)
    old_payload=torch.load(PILOT/'budget/best.pt',weights_only=True)
    models=[]
    for payload in [old_payload,new_payload]:
        model=bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();model.load_state_dict(payload['state_dict']);models.append(model)
    val=bp.load_data('validation')
    old_val=bp.probabilities(models[0],val)
    grid=[dict(quantile=q,**bp.metrics(val,bp.quantile_actions(old_val,q))) for q in bp.QUANTILES]
    old_recal=min([r for r in grid if r['hit_95']>=.95],key=lambda r:(r['mean_middle_blocks'],r['quantile']))['quantile']
    newq=json.loads((ROOT/'budget/calibration.json').read_text())['selected_quantile']
    report=dict(new_epoch=new_payload['epoch'],new_quantile=newq,old_recalibrated_quantile=old_recal,old_recalibration_grid=grid,splits={})
    for split in ['validation','test']:
        data=bp.load_data(split);oldp,newp=[bp.probabilities(m,data) for m in models]
        actions=dict(old_original=bp.quantile_actions(oldp,.95),old_recalibrated=bp.quantile_actions(oldp,old_recal),
            new_argmax=newp.argmax(-1),new_calibrated=bp.quantile_actions(newp,newq),oracle_budget=data['labels'],
            fixed32=torch.full_like(data['labels'],4),fixed64=torch.full_like(data['labels'],5),fixed128=torch.full_like(data['labels'],6))
        groups=dict(overall=torch.ones(len(oldp),dtype=torch.bool),full=~data['trajectory'],old_adaptive=data['trajectory'])
        for doc in range(len(data['documents'])):groups[f'document_{doc}']=data['identities'][:,1]==doc
        scores={}
        for group,mask in groups.items():
            scores[group]={k:bp.metrics(data,a,mask) for k,a in actions.items()}
            cov=data['fixed112_coverage'][mask];cost=data['fixed112_blocks'][mask].float()
            scores[group]['fixed112']=dict(scenes=int(mask.sum()),mean_coverage=float(cov.mean()),hit_95=float((cov>=.95-1e-7).float().mean()),mean_middle_blocks=float(cost.mean()))
        report['splits'][split]=scores
        torch.save(dict(old_probabilities=oldp,new_probabilities=newp,actions=actions),ROOT/f'{split}_predictions.pt')
    dump('budget_results.json',report)
    print(json.dumps(report),flush=True)


@torch.inference_mode()
def hotpot(shard):
    model,tok,ranker,oldbudget,eos,config=run.load()
    payload=torch.load(ROOT/'budget/best.pt',weights_only=True)
    budget=bp.BudgetClassifier(payload['config']['feature_size']).cuda().eval();budget.load_state_dict(payload['state_dict'])
    q=json.loads((ROOT/'budget/calibration.json').read_text())['selected_quantile']
    rows=json.loads((BASE/'selected.json').read_text())
    run.prefill(model,rows[0]['input_ids'][:128])
    with (ROOT/f'hotpot_{shard}.jsonl').open('x') as journal:
        for i,row in enumerate(rows):
            if i%4!=shard:continue
            pairs,logits,prefill_seconds=run.prefill(model,row['input_ids'])
            run.sync();start=time.perf_counter();digests=rt.build_prefix_digests(pairs);run.sync();index_seconds=time.perf_counter()-start
            result=run.decode(model,pairs,logits,'adaptive',ranker,budget,digests,eos,quantile=q)
            result['prediction']=tok.decode(result['ids'],skip_special_tokens=True).strip()
            result['f1']=100*max(run.token_f1(result['prediction'],a) for a in row['answers'])
            record=dict(source_index=row['source_index'],input_ids_sha256=row['input_ids_sha256'],truncated=row['truncated'],
                quantile=q,prefill_seconds=prefill_seconds,index_seconds=index_seconds,new_adaptive=result)
            journal.write(json.dumps(record)+'\n');journal.flush()
            print(json.dumps(dict(source_index=row['source_index'],f1=result['f1'],blocks=result['mean_middle_blocks'])),flush=True)
            del pairs,logits,digests,result;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ['collect','hotpot']:globals()[stage](int(sys.argv[2]))
    else:globals()[stage]()
