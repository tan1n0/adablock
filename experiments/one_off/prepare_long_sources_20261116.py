"""Prepare disjoint 16K/32K Llama contexts from non-RULER packed documents.

The source packs used Qwen tokenization. We retokenize text with the local
Llama tokenizer and add more disjoint source packs when needed. These are
inputs only; no oracle labels or decoding queries are fabricated here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[2]
MODEL=Path('/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb')
OUTPUT=ROOT/'results/value_aware_selector_20261116/long_sources'
SEED=20261116
SEP='\n\n--- DOCUMENT BOUNDARY ---\n\n'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)

def load_candidates(path):
    candidates={16384:[],32768:[],4096:[]}
    with path.open() as f:
        for line_number,line in enumerate(f):
            row=json.loads(line)
            target=row['target_token_count']
            if target in candidates:
                candidates[target].append((line_number,row))
    return candidates

def prepare(args):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.bos_token_id is None:
        raise ValueError('Llama BOS token required')
    rng=random.Random(args.seed)
    planned={'model':str(args.model.resolve()),'tokenizer_class':type(tokenizer).__name__,
             'seed':args.seed,'source_sha256':{},'samples':{},
             'source': 'disjoint strict packed documents, no RULER evaluation prompts',
             'query_plan': 'on GPU collect 64 sampled prefill positions plus first 16 real greedy-decode queries per context, per layer and query head',
             'label_plan': 'post-RoPE full QK softmax; m[g,b], c[g,b], fixed current block and exact K/V diagonal summaries',
             'status': 'inputs prepared; Q/K/V and labels not collected'}
    all_components={}
    for split,n in [('train',args.train_per_length),('validation',args.valid_per_length)]:
        source=ROOT/f'data/{split}_strict_packed_tokens.jsonl'
        planned['source_sha256'][split]=sha(source)
        candidates=load_candidates(source)
        for rows in candidates.values(): rng.shuffle(rows)
        used=set()
        produced=[]
        for length in (32768,16384):
            # First take packs near the target length, then extend with 4K packs.
            primary=list(candidates[length])
            supplement=list(candidates[4096])+list(candidates[16384])+list(candidates[32768])
            rng.shuffle(supplement)
            primary_cursor=0
            for sample_index in range(n):
                ids=[tokenizer.bos_token_id]
                components=[]; parts=[]
                pool=primary
                while len(ids)<length:
                    chosen=None
                    for line_number,row in pool[primary_cursor:] if pool is primary else pool:
                        c=set(row['component_doc_ids'])
                        if c and not c.intersection(used) and not c.intersection(components):
                            chosen=(line_number,row,c)
                            break
                    if chosen is None:
                        if pool is primary:
                            pool=supplement
                            continue
                        raise RuntimeError(f'Insufficient disjoint {split} text for {length} tokens')
                    line_number,row,c=chosen
                    if pool is primary:
                        primary_cursor=primary.index((line_number,row))+1
                    else:
                        supplement.remove((line_number,row))
                    if parts:
                        ids.extend(tokenizer.encode(SEP,add_special_tokens=False))
                    ids.extend(tokenizer.encode(row['text'],add_special_tokens=False))
                    components.extend(sorted(c))
                    parts.append({'source_line':line_number,'doc_id':row['doc_id'],
                                  'source':row['source'],'component_doc_ids':sorted(c),
                                  'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest()})
                    used.update(c)
                ids=ids[:length]
                item={'split':split,'target_length':length,'sample_index':sample_index,
                      'tokenizer':str(args.model.resolve()),'source_parts':parts,
                      'component_doc_ids':components,'input_ids':ids,
                      'input_sha256':hashlib.sha256(bytes().join(int(i).to_bytes(4,'little') for i in ids)).hexdigest()}
                produced.append(item)
                all_components[(split,length,sample_index)]=set(components)
        path=args.output/f'{split}.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('w') as f:
            for item in produced: f.write(json.dumps(item,ensure_ascii=False)+'\n')
        tmp.replace(path)
        planned['samples'][split]={'count':len(produced),'lengths':{str(length):sum(x['target_length']==length for x in produced) for length in (16384,32768)},
                                   'file':str(path.resolve()),'sha256':sha(path)}
    train_components=set().union(*(v for k,v in all_components.items() if k[0]=='train'))
    valid_components=set().union(*(v for k,v in all_components.items() if k[0]=='validation'))
    if train_components & valid_components:
        raise ValueError('Train/validation source component overlap')
    planned['train_validation_component_overlap']=0
    target=args.output/'protocol.json'
    if target.exists() and json.loads(target.read_text()) != planned:
        raise ValueError('Existing protocol differs; use another output directory')
    save(target,planned)
    print(json.dumps(planned['samples'],indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--train-per-length',type=int,default=4)
    parser.add_argument('--valid-per-length',type=int,default=2)
    prepare(parser.parse_args())
