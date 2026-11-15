import json,sys,hashlib
from pathlib import Path
import pyarrow.parquet as pq
import torch
QA=Path('/root/adablock/results/llama_qa_budget_20261015_01')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(QA));import engine as e

def prepare():
    rows=json.loads((QA/'selected.json').read_text())
    train=sorted([r for r in rows if r['split']=='budget_train'],key=lambda r:r['prompt_tokens'])
    chosen=train[::4]+[r for r in rows if r['split']=='validation']
    wanted={r['raw_id'] for r in chosen}
    rawpath=json.loads((QA/'protocol.json').read_text())['source']
    raw={r['id']:r for b in pq.ParquetFile(rawpath).iter_batches() for r in b.to_pylist() if r['id'] in wanted}
    tok=e.run.AutoTokenizer.from_pretrained(e.run.MODEL,local_files_only=True)
    from prepare_qa import PROMPT
    for r in chosen:
        candidates=raw[r['raw_id']]['candidates']
        assert r['answers'][0] in candidates
        text=PROMPT.format(context=r['context'],question=r['question'])
        assert tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)==r['input_ids']
        # Preserve context/question verbatim; add the supplied candidates and explicit selection instruction.
        text=text.removesuffix('Answer:')+'Candidates:\n'+'\n'.join(candidates)+'\nSelect exactly one candidate and output its text only.\nAnswer:'
        r['candidate_ids']=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
        r['candidates']=candidates
        assert len(r['candidate_ids'])<=8160
    assert len(chosen)==24 and all(r['split']!='test' for r in chosen)
    with (ROOT/'selected.json').open('x') as f:json.dump(chosen,f)
    (ROOT/'protocol.json').write_text(json.dumps(dict(selection='12 training rows at sorted-length indices 0,4,...44 plus all 12 validation rows; no outcome selection',variants=['open','candidate_aware'],max_new_tokens=32,attention='full',candidate_order='original dataset order',note='Candidate-aware generative diagnostic, not reproduction of original candidate-scoring architecture',selected_sha256=hashlib.sha256((ROOT/'selected.json').read_bytes()).hexdigest()),indent=2))

@torch.inference_mode()
def worker(shard):
    model,tok,ranker,budget,eos,_=e.run.load()
    rows=json.loads((ROOT/'selected.json').read_text())
    with (ROOT/f'worker_{shard}.jsonl').open('x') as f:
        for i,r in enumerate(rows):
            if i%4!=shard:continue
            result=dict(raw_id=r['raw_id'],split=r['split'],answers=r['answers'],methods={})
            for method,key in [('open','input_ids'),('candidate_aware','candidate_ids')]:
                pairs,logits,_=e.run.prefill(model,r[key])
                out=e.run.decode(model,pairs,logits,'full',ranker,budget,None,eos)
                pred=tok.decode(out['ids'],skip_special_tokens=True).strip()
                result['methods'][method]=dict(prediction=pred,f1=100*max(e.run.token_f1(pred,a) for a in r['answers']),exact_match=pred.lower().strip()==r['answers'][0].lower().strip(),hit_generation_limit=out['hit_generation_limit'],ids=out['ids'])
                del pairs,logits
            f.write(json.dumps(result)+'\n');f.flush()
            print(json.dumps(result),flush=True)

if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else worker(int(sys.argv[1]))
