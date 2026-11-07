import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
PILOT = Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
MODEL = '/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb'
DATA = Path('/root/adablock/data/longbench/hotpotqa.jsonl')
PROMPT = ('Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n'
          'The following are given passages.\n{context}\n\n'
          'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n'
          'Question: {input}\nAnswer:')
SEED = 20261015
MAX_NEW = 32
MAX_INPUT = 8192 - MAX_NEW


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    encoded = []
    for index, row in enumerate(rows):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT.format(**row)}],
                                             tokenize=True, add_generation_prompt=True)
        encoded.append(dict(source_index=index, input_ids=ids, answers=row['answers'],
                            question=row['input'], prompt_tokens=len(ids),
                            source_id=row.get('_id'), input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    eligible = [r for r in encoded if 544 < r['prompt_tokens'] <= MAX_INPUT]
    candidates = encoded[:]
    random.Random(SEED).shuffle(candidates)
    selected = sorted(sorted(candidates, key=lambda r: r['prompt_tokens'])[:50], key=lambda r: r['source_index'])
    for row in selected:
        row['original_prompt_tokens'] = row['prompt_tokens']
        row['truncated'] = row['prompt_tokens'] > MAX_INPUT
        if row['truncated']:
            row['input_ids'] = row['input_ids'][:MAX_INPUT//2] + row['input_ids'][-(MAX_INPUT-MAX_INPUT//2):]
        row['prompt_tokens'] = len(row['input_ids'])
        row['input_ids_sha256'] = hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()
    if (ROOT / 'selected.json').exists():
        raise FileExistsError('selection already frozen')
    (ROOT / 'selected.json').write_text(json.dumps(selected, ensure_ascii=False))
    calibration = json.loads((PILOT / 'budget/calibration.json').read_text())
    config = dict(model=MODEL, methods=['full', 'fixed32', 'fixed64', 'adaptive'], seed=SEED,
        task='hotpotqa', n=50, source_samples=len(rows), eligible_samples=len(eligible),
        selection='50 shortest complete prompts by token length; seeded tie order; includes all 29 untruncated eligible examples and 21 nearest over-limit examples; no answer-based selection',
        selected_indices=[r['source_index'] for r in selected], input_tokens=[r['prompt_tokens'] for r in selected],
        max_input_tokens=MAX_INPUT, max_new_tokens=MAX_NEW, max_total_tokens=8192,
        truncation='if over8160, retain first4080 and last4080 tokens; separate untruncated/truncated results', prompt=PROMPT, chat_template='tokenize=True, add_generation_prompt=True; no extra BOS',
        dtype='bfloat16', prefill='full SDPA; identical prefill cache and first-token logits shared across methods',
        decoding='greedy, all generation_config EOS ids, sparse from the second output token onward',
        budget_quantile=calibration['selected_quantile'], budget_calibration='frozen from previous non-LongBench validation',
        block_size=16, sink_blocks=2, local_blocks=32, cobs_rank=4, reuse=False,
        attention='exact selected K/V SDPA, valid-mask padding for variable per-KV-head budgets; full KV retained on GPU',
        metric='normalized answer token F1, maximum across references, sample macro mean times 100',
        data_sha256=sha(DATA), ranker_sha256=sha(PILOT/'ranker/best.pt'), budget_sha256=sha(PILOT/'budget/best.pt'),
        comparison_scope='8K-capped 50-question length-selected subset; not official all-200 LongBench scores',
        timing='warm model; per-question shared prefill and index creation measured once, charged to each applicable method; independent autoregressive decode timing includes scoring, budget, gather, attention and cache update',
        sparse_index='same immutable prefix summaries shared by sparse methods; newly completed blocks appended separately',
        decision_rule='no retuning on HotpotQA results; no automatic data expansion based only on this small subset')
    (ROOT/'protocol.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(json.dumps({'eligible':len(eligible),'selected':50,'min_tokens':min(config['input_tokens']),
                      'max_tokens':max(config['input_tokens']),'quantile':config['budget_quantile']}))


if __name__ == '__main__':
    prepare()
