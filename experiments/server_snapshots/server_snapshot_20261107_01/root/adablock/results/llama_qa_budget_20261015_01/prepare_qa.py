"""Restore full WikiHop supports, remove answer suffixes, and freeze QA splits."""
import hashlib
import json
import random
import sys
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
PILOT=Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
DATA=Path('/root/adablock/data')
sys.path.insert(0,'/root/adablock')
from scripts.build_strict_training_corpus import clean_text,LongBenchDecontaminator,sampled_ngram_hashes
MODEL='/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb'
RAW=Path('/root/adablock/.cache/adablock_training_sources/datasets--MoE-UNC--wikihop/snapshots/2a59420990da1085b42b474cc38d2520c27a6325/data/train-00000-of-00001.parquet')
SEED=20261015
PROMPT=('Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n'
        'The following are given passages.\n{context}\n\n'
        'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n'
        'Question: {question}\nAnswer:')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))

def main():
    assert not (ROOT/'selected.json').exists()
    used={c for split in ['rank_train','budget_train','validation','test'] for line in (PILOT/f'{split}.jsonl').read_text().splitlines() for c in json.loads(line)['component_doc_ids']}
    original=defaultdict(list)
    for split in ['train','validation']:
        for line in (DATA/'strict_sources'/f'wikihop_{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            prefix,answer=r['text'].rsplit('\n\nAnswer:\n',1)
            context,question=prefix.rsplit('\n\nQuestion:\n',1)
            original[(clean_text(question),clean_text(answer))].append(dict(r,source_pool=split,question=question,answer=answer,
                original_context=context.removeprefix('Supporting documents:\n')))
    guard=LongBenchDecontaminator(DATA/'longbench',13,5,3)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    counts=Counter();pools=defaultdict(list);seen=set()
    for batch in pq.ParquetFile(RAW).iter_batches(batch_size=128):
        for r in batch.to_pylist():
            key=(clean_text(r['query']).replace('_',' '),clean_text(r['answer']))
            if key not in original:continue
            context=clean_text('\n\n'.join(f'Document {i+1}: {clean_text(s)}' for i,s in enumerate(r['supports'])))
            matches=[]
            for src in original[key]:
                chunks=src['original_context'].split('\n...[middle omitted]...\n')
                if all(chunk.strip() in context for chunk in chunks):matches.append(src)
            if len(matches)!=1:counts['ambiguous_or_unmatched']+=1;continue
            src=matches[0]
            if 'wikihop::'+src['doc_id'] in used:counts['previous_pilot_component']+=1;continue
            if src['doc_id'] in seen:counts['duplicate_source']+=1;continue
            seen.add(src['doc_id'])
            if guard.contaminated(context):counts['restored_longbench_overlap']+=1;continue
            text=PROMPT.format(context=context,question=src['question'])
            ids=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
            if not 2048<=len(ids)<=8160:counts['outside_2k_8k']+=1;continue
            pools[src['source_pool']].append(dict(doc_id=src['doc_id'],raw_id=r['id'],source='wikihop_full_supports',
                component_doc_ids=['wikihop::'+src['doc_id']],question=src['question'],answers=[src['answer']],
                context=context,input_ids=ids,prompt_tokens=len(ids),input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),source_pool=src['source_pool']))
    print(json.dumps(dict(eligible={k:len(v) for k,v in pools.items()},excluded=dict(counts))),flush=True)
    rng=random.Random(SEED)
    # Fix all selection before collecting any model output. Rank-length thirds give broad natural lengths.
    train=sorted(pools['train'],key=lambda r:(r['prompt_tokens'],r['doc_id']))
    held=sorted(pools['validation'],key=lambda r:(r['prompt_tokens'],r['doc_id']))
    assert len(train)>=48 and len(held)>=24,(len(train),len(held))
    def fingerprints(r):return set(sampled_ngram_hashes(r['context'],13,5))
    def sample_lengths(pool,n):
        ordered=sorted(pool,key=lambda r:(r['prompt_tokens'],r['doc_id']))
        thirds=[ordered[len(ordered)*i//3:len(ordered)*(i+1)//3] for i in range(3)]
        for group in thirds:rng.shuffle(group)
        assert all(len(group)>=n//3 for group in thirds),(len(pool),n)
        return [r for group in thirds for r in group[:n//3]]
    validation=sample_lengths(held,12)
    validation_ids={r['doc_id'] for r in validation}
    validation_fp=set().union(*(fingerprints(r) for r in validation))
    test_pool=[r for r in held if r['doc_id'] not in validation_ids and len(fingerprints(r)&validation_fp)<3]
    test=sample_lengths(test_pool,12)
    held_fp=validation_fp|set().union(*(fingerprints(r) for r in test))
    train_pool=[r for r in train if len(fingerprints(r)&held_fp)<3]
    training=sample_lengths(train_pool,48)
    counts['cross_split_support_filter_train']=len(train)-len(train_pool)
    counts['cross_split_support_filter_test']=len(held)-12-len(test_pool)
    selected=[dict(r,split=s) for s,group in [('budget_train',training),('validation',validation),('test',test)] for r in group]
    final=[]
    for split in ['budget_train','validation','test']:
        group=sorted([r for r in selected if r['split']==split],key=lambda r:(r['prompt_tokens'],r['doc_id']))
        for i,r in enumerate(group):final.append(dict(r,document=i))
    assert len({r['doc_id'] for r in final})==72 and len({r['raw_id'] for r in final})==72
    assert len({r['context_sha256'] for r in final})==72
    # WikiHop questions can share supporting articles even with different IDs: measure and disclose it.
    supports={split:set() for split in ['budget_train','validation','test']}
    for r in final:supports[r['split']].update(sampled_ngram_hashes(r['context'],13,5))
    overlap={a+'__'+b:len(supports[a]&supports[b]) for a,b in [('budget_train','validation'),('budget_train','test'),('validation','test')]}
    dump('selected.json',final)
    dump('protocol.json',dict(seed=SEED,source=str(RAW),source_sha256=sha(RAW),source_metadata_sha256=sha(DATA/'strict_sources/wikihop_metadata.json'),
        split_counts=dict(Counter(r['split'] for r in final)),eligible_counts={k:len(v) for k,v in pools.items()},excluded=dict(counts),
        length_summary={s:dict(min=min(r['prompt_tokens'] for r in final if r['split']==s),max=max(r['prompt_tokens'] for r in final if r['split']==s),
            mean=sum(r['prompt_tokens'] for r in final if r['split']==s)/sum(r['split']==s for r in final)) for s in supports},
        selection='Seeded sampling from length thirds, before generation; full original supports, no truncation or added distractors.',
        split_source='Budget train from strict WikiHop train; validation/test from strict WikiHop validation, excluding every prior pilot component.',
        shared_13gram_fingerprints=overlap,cross_split_filter='Reject training question with >=3 sampled 13-gram matches to validation/test union; reject test question with >=3 matches to validation union.',decontamination=dict(ngram_words=13,stride=5,min_matches=3,files=guard.files),
        answer_leakage_guard='Gold answer suffix removed structurally; answers naturally occurring in supporting documents are retained.',
        prompt_template=PROMPT,max_new_tokens=32,sample_decode_steps=[0,1,3,7,15,23,30],trajectories=['full','old_adaptive'],
        ranker_sha256=sha(PILOT/'ranker/best.pt'),old_budget_sha256=sha(PILOT/'budget/best.pt'),
        architecture='Unchanged 588 -> 128 -> 64 -> 9 CE budget classifier; frozen ranker.',
        training='Same seed, optimizer, 40 epochs, validation-CE checkpoint selection and validation-only quantile grid.',
        scope='QA data and document count change together; not a one-factor attribution study. Hotpot50 remains regression-only.'))
    print(json.dumps(json.loads((ROOT/'protocol.json').read_text())['length_summary']),flush=True)

if __name__=='__main__':main()
