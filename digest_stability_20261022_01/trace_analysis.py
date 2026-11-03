import diagnose as d
from transformers import AutoTokenizer
import torch,json
tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb',local_files_only=True)
a=torch.load(d.ROOT/'trace_svd.pt',weights_only=True,map_location='cpu')
b=torch.load(d.ROOT/'trace_gram.pt',weights_only=True,map_location='cpu')
report=d.compare(a,b,tok)
row=next(r for r in json.loads((d.PREVIOUS/'selected.json').read_text()) if r['task']=='cwe' and r['length']==8192)
event=report['first_selection_difference']
event['decode_step']=event['position']-len(row['input_ids'])+1
ra=next(r for r in a['trace'] if r['position']==event['position'] and r['layer']==event['layer'])
rb=next(r for r in b['trace'] if r['position']==event['position'] and r['layer']==event['layer'])
event['candidate_scores']=[]
for head in event['heads']:
    h=head['head']
    for block in head['lost']+head['gained']:
        i=int((ra['block_ids']==block).nonzero()[0])
        event['candidate_scores'].append(dict(head=h,block=block,reference=float(ra['scores'][h,i]),gram=float(rb['scores'][h,i])))
event['block_texts']={str(block):tok.decode(row['input_ids'][block*16:(block+1)*16])
    for head in event['heads'] for kind in ['lost','gained'] for block in head[kind]}
d.e.dump('trace_analysis.json',report)
print(json.dumps(report),flush=True)
