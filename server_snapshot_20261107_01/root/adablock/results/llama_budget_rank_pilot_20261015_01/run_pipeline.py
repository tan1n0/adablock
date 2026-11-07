"""Run the isolated pilot, retain logs and fail closed on any stage error."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def status(stage, **kwargs):
    record = dict(stage=stage, time=time.time(), **kwargs)
    (ROOT / 'status.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def run(command, log, gpu='0'):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    with (ROOT / log).open('w') as handle:
        subprocess.run([PYTHON, '-u', *command], stdout=handle, stderr=subprocess.STDOUT,
                       env=env, cwd=ROOT, check=True)


if __name__ == '__main__':
    try:
        (ROOT / 'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / 'source').rglob('*.py')}, indent=2))
        status('extracting')
        workers = []
        handles = []
        for gpu, split in enumerate(('rank_train', 'budget_train', 'validation', 'test')):
            handle = (ROOT / f'collect_{split}.log').open('w')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
            workers.append(subprocess.Popen([PYTHON, '-u', str(ROOT / 'prepare_collect.py'), 'collect', '--split', split],
                    stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
        codes = [p.wait() for p in workers]
        for handle in handles:
            handle.close()
        if any(codes):
            raise RuntimeError(f'Extraction failed: {codes}')
        status('ranker_training')
        run([str(ROOT / 'source/scripts/train_block_ranker_batched.py'),
             '--train-dir', str(ROOT / 'data/rank_train'), '--val-dir', str(ROOT / 'data/validation'),
             '--output-dir', str(ROOT / 'ranker'), '--epochs', '3', '--batch-size', '64',
             '--lr', '.0002', '--seed', '20261015', '--cpu-threads', '4', '--max-val-scenes', '4096'], 'ranker.log')
        status('ranker_conditioned_budget_labels')
        for split in ('budget_train', 'validation', 'test'):
            run([str(ROOT / 'budget_pilot.py'), 'labels', '--split', split], f'labels_{split}.log')
        status('budget_training')
        run([str(ROOT / 'budget_pilot.py'), 'train'], 'budget.log')
        status('heldout_evaluation')
        run([str(ROOT / 'budget_pilot.py'), 'evaluate'], 'evaluation.log')
        status('complete')
    except Exception as exc:
        status('failed', error=repr(exc))
        raise
