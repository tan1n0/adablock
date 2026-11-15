from pathlib import Path
import json,hashlib,re,string,collections,datetime
from transformers import AutoTokenizer
ROOT=Path('/root/adablock')
OUT=ROOT/'results/longbench_ranker_hotpotqa_cap32_postprocess_20261004'
MODEL=Path('/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
SOURCES={'full':ROOT/'results/longbench_ranker_hotpotqa_full_20261004/full/full.jsonl','all28':ROOT/'results/longbench_ranker_all28_hotpotqa_20261004/learned_all28.jsonl'}
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def f1(p,refs):
 def one(a):
  x=norm(p).split();y=norm(a).split();n=sum((collections.Counter(x)&collections.Counter(y)).values())
  return 200*n/(len(x)+len(y)) if n else 0.
 return max(one(a) for a in refs)
def em(p,refs):return int(any(norm(p)==norm(a) for a in refs))
data={};changed={};audit={'source_sha256':{m:sha(p) for m,p in SOURCES.items()},'tokenizer_path':str(MODEL),'tokenizer_sha256':{f:sha(MODEL/f) for f in ['tokenizer.json','tokenizer_config.json']},'cap_includes_eos':True,'decode_options':{'skip_special_tokens':True,'strip_after_decode':True},'no_model_run':True}
for m,p in SOURCES.items():
 rows=[json.loads(l) for l in p.read_text().splitlines()]
 assert len(rows)==200 and {r['index'] for r in rows}==set(range(200))
 results=[]
 for r in sorted(rows,key=lambda x:x['index']):
  tr=r['trace'];ids=[t['token_id'] for t in tr]
  assert [t['step'] for t in tr]==list(range(len(tr)))
  assert len(ids)==r['stats']['generated_tokens']
  assert ids[-1]==tok.eos_token_id and tok.eos_token_id not in ids[:-1]
  pred128=tok.decode(ids,skip_special_tokens=True).strip()
  assert pred128==r['prediction'],(m,r['index'],'decode mismatch')
  assert abs(f1(pred128,r['answers'])-r['score'])<1e-8
  pred32=tok.decode(ids[:32],skip_special_tokens=True).strip()
  results.append(dict(index=r['index'],input_ids_sha256=r['input_ids_sha256'],answers=r['answers'],original_generated_token_ids=ids,retained_token_ids=ids[:32],original_generated_tokens=len(ids),cap32_ended_with_eos=ids[:32][-1]==tok.eos_token_id,prediction128=pred128,prediction32=pred32,f1_128=f1(pred128,r['answers']),f1_32=f1(pred32,r['answers']),em128=em(pred128,r['answers']),em32=em(pred32,r['answers'])))
 data[m]=results;changed[m]=[r for r in results if r['prediction32']!=r['prediction128']]
for x,y in zip(data['full'],data['all28']):
 assert x['index']==y['index'] and x['input_ids_sha256']==y['input_ids_sha256'] and x['answers']==y['answers']
summary={'samples':200,'methods':{},'ranker_vs_full':{}}
for m,rs in data.items():
 summary['methods'][m]={}
 for cap in [128,32]:
  summary['methods'][m][str(cap)]={'f1':sum(r[f'f1_{cap}'] for r in rs)/200,'em_percent':sum(r[f'em{cap}'] for r in rs)/2,'em_count':sum(r[f'em{cap}'] for r in rs)}
 summary['methods'][m]['delta_32_minus_128']={k:summary['methods'][m]['32'][k]-summary['methods'][m]['128'][k] for k in ['f1','em_percent']}
 summary['methods'][m]['changed_prediction_indices']=[r['index'] for r in changed[m]]
 summary['methods'][m]['changed_score_indices']=[r['index'] for r in rs if abs(r['f1_32']-r['f1_128'])>1e-8]
for cap in [128,32]:
 ds=[r[f'f1_{cap}']-f[f'f1_{cap}'] for r,f in zip(data['all28'],data['full'])]
 summary['ranker_vs_full'][str(cap)]={'f1_delta':sum(ds)/200,'wins':sum(d>1e-8 for d in ds),'ties':sum(abs(d)<=1e-8 for d in ds),'losses':sum(d< -1e-8 for d in ds)}
rows=json.loads((ROOT/'results/longbench_ranker_hotpotqa_full_20261004/selected_rows.json').read_text())
changes=[]
for m,rs in changed.items():
 for r in rs:
  changes.append({'method':m,'index':r['index'],'question':rows[r['index']]['input'],'answers':r['answers'],'tokens128':r['original_generated_tokens'],'prediction128':r['prediction128'],'prediction32':r['prediction32'],'f1_128':r['f1_128'],'f1_32':r['f1_32'],'delta':r['f1_32']-r['f1_128']})
audit.update(all400_trace_lengths_match=True,all400_full_trace_decodes_match_original=True,all400_original_scores_match=True,all400_eos_last_only=True,all200_input_hashes_and_answers_match=True,source_hashes_unchanged=all(sha(p)==audit['source_sha256'][m] for m,p in SOURCES.items()))
for m,rs in data.items():
 with (OUT/(m+'_cap32.jsonl')).open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for name,obj in [('summary.json',summary),('audit.json',audit),('changed_cases.json',changes)]:
 with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
doc=ROOT/'docs/ranker_full_training_20261003.md'
with doc.open('a') as f:
 f.write('\n\n### HotpotQA 128→32 token离线精确截断复评分（2026-10-04）\n\n')
 f.write('仅使用原full/all28各200条真实生成token trace，取前32个新增token（EOS占预算），原tokenizer decode(skip_special_tokens=True).strip()。400条完整trace回解码与原预测、原F1全部一致；同200输入哈希及答案对齐。未运行模型，未修改原结果。\n\n')
 f.write(json.dumps(summary,ensure_ascii=False,indent=2))
 f.write('\n\n产物：'+str(OUT)+'。脚本rescore.py、audit.json、changed_cases.json及两方法cap32 JSONL。\n')
 f.write('等价范围：既有手写greedy无长度相关logit处理或提前计划，128和32仅改变循环上限；本批首token无EOS，因此截前32精确对应同实现同数值执行的32上限输出。此为离线反事实，不是新推理实测，不复现HF generate重复惩罚或LouisKV未知协议。\n')
print(json.dumps({'summary':summary,'audit':audit,'largest_changes':sorted(changes,key=lambda r:-abs(r['delta']))[:6]},ensure_ascii=False,indent=2))
