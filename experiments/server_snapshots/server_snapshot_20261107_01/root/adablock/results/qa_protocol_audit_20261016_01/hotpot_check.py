import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/adablock/results/llama_qa_budget_20261015_01')
import engine as e

@torch.inference_mode()
def main(shard):
    rows=[r for r in json.loads((ROOT/'hotpot_selected.json').read_text()) if r['split']!='test']
    model,tok,ranker,budget,eos,_=e.run.load()
    with (ROOT/f'hotpot_check_{shard}.jsonl').open('x') as f:
        for i,r in enumerate(rows):
            if i%4!=shard:continue
            pairs,logits,_=e.run.prefill(model,r['input_ids'])
            out=e.run.decode(model,pairs,logits,'full',ranker,budget,None,eos)
            pred=tok.decode(out['ids'],skip_special_tokens=True).strip()
            record=dict(id=r['id'],split=r['split'],bucket=r['bucket'],prompt_tokens=r['prompt_tokens'],prediction=pred,answer=r['answer'],f1=100*e.run.token_f1(pred,r['answer']),ids=out['ids'],hit_generation_limit=out['hit_generation_limit'])
            f.write(json.dumps(record)+'\n');f.flush()
            print(json.dumps(record),flush=True)
            del pairs,logits

if __name__=='__main__':main(int(sys.argv[1]))
