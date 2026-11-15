import json,hashlib,sys,random,re
from pathlib import Path
from collections import Counter,defaultdict
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
DATA=Path('/root/adablock/data')
sys.path.insert(0,'/root/adablock')
from scripts.build_strict_training_corpus import LongBenchDecontaminator,sampled_ngram_hashes
sys.path.insert(0,'/root/adablock/results/llama_qa_budget_20261015_01')
from prepare_qa import PROMPT,MODEL

def main():
    tree=json.loads((ROOT/'hf_tree.json').read_text());sources=[]
    for t in tree:
        if not t['path'].startswith('distractor/train-'):continue
        p=DATA/'hotpot_official'/Path(t['path']).name
        part=p.with_suffix(p.suffix+'.part')
        source=p if p.exists() else part
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==t['lfs']['oid'],(str(source),digest)
        if source==part:part.rename(p)
        sources.append(dict(path=str(p),sha256=digest))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    guard=LongBenchDecontaminator(DATA/'longbench',13,5,3)
    norm=lambda x:' '.join(re.findall(r'[a-z0-9]+',x.lower()))
    lb=[json.loads(l) for p in (DATA/'longbench').glob('*.jsonl') for l in p.read_text().splitlines()]
    lbq={norm(r.get('input','')) for r in lb};lbids={str(r.get('_id',r.get('id',''))) for r in lb}
    previous=json.loads(Path('/root/adablock/results/llama_qa_budget_20261015_01/selected.json').read_text())
    prevfp=set().union(*(set(sampled_ngram_hashes(r['context'],13,5)) for r in previous))
    counts=Counter();lengths=[];eligible=defaultdict(list)
    for src in sources:
        for batch in pq.ParquetFile(src['path']).iter_batches(batch_size=128):
            rows=batch.to_pylist();texts=[];contexts=[]
            for r in rows:
                c=r['context'];context='\n\n'.join(title+'\n'+''.join(sentences) for title,sentences in zip(c['title'],c['sentences']))
                contexts.append(context);texts.append(tok.apply_chat_template([dict(role='user',content=PROMPT.format(context=context,question=r['question']))],tokenize=False,add_generation_prompt=True))
            encoded=tok(texts,add_special_tokens=False)['input_ids']
            for r,context,ids in zip(rows,contexts,encoded):
                n=len(ids);lengths.append(n);bucket='under2k' if n<2048 else '2k4k' if n<4096 else '4k6k' if n<6144 else '6k8k' if n<=8160 else 'over8k';counts['raw_'+bucket]+=1
                if bucket in ['under2k','over8k']:continue
                if norm(r['question']) in lbq or r['id'] in lbids:counts['excluded_question_or_id']+=1;continue
                if guard.contaminated(context):counts['excluded_longbench_context']+=1;continue
                fp=set(sampled_ngram_hashes(context,13,5))
                if len(fp&prevfp)>=3:counts['excluded_previous_qa_context']+=1;continue
                eligible[bucket].append(dict(id=r['id'],question=r['question'],answer=r['answer'],context=context,input_ids=ids,prompt_tokens=n,bucket=bucket))
            if len(lengths)%4096==0:print('processed',len(lengths),flush=True)
    # Freeze held-out IDs before any inference. Shared material is filtered before accepting each split.
    rng=random.Random(20261016);usedfp=set();selected=[];shared_exclusions=Counter()
    for bucket in ['6k8k','4k6k','2k4k']:
        pool=sorted(eligible[bucket],key=lambda r:r['id']);rng.shuffle(pool)
        for split,target in [('test',4),('validation',4),('budget_train',16)]:
            n=0;splitfp=set();remaining=[]
            for r in pool:
                fp=set(sampled_ngram_hashes(r['context'],13,5))
                if n<target and len(fp&usedfp)<3:
                    selected.append(dict(r,split=split));splitfp.update(fp);n+=1
                else:remaining.append(r)
            pool=remaining;usedfp.update(splitfp)
    with (ROOT/'hotpot_selected.json').open('x') as f:json.dump(selected,f)
    stats=dict(total=len(lengths),min_tokens=min(lengths),max_tokens=max(lengths),mean_tokens=sum(lengths)/len(lengths),counts=dict(counts),eligible={k:len(v) for k,v in eligible.items()},selected=dict(Counter(r['split']+'_'+r['bucket'] for r in selected)),sources=sources,revision=json.loads((ROOT/'dataset_revision.json').read_text())['sha'],source_note='Hugging Face hotpotqa/hotpot_qa train parquet via hf-mirror; verified against LFS SHA256; original CMU download unavailable',protocol='All original context paragraphs in original order; no truncation, added distractors, answer field or supporting-fact labels in prompt. Full prompt tokenized once. Selection seeded before inference.',decontamination='LongBench ID/question and >=3 sampled 13-gram context matches, plus all previous QA contexts; held-out splits assigned before training; >=3 cross-split matches rejected',selected_sha256=hashlib.sha256((ROOT/'hotpot_selected.json').read_bytes()).hexdigest())
    (ROOT/'hotpot_audit.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
