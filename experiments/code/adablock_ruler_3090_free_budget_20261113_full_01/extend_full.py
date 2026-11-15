"""Run the remaining cases in both existing RULER 32K task pools."""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import compare

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.with_name('adablock_ruler_3090_free_budget_20261113_01')
TARGET = 100


def prepare():
    source = compare.AUTO / 'selected.json'
    protocol = json.loads((PILOT / 'protocol.json').read_text())
    assert compare.sha(source) == protocol['source_sha256']
    assert compare.sha(PILOT / 'selected.json') == protocol['selected_sha256']
    assert compare.sha(ROOT / 'compare.py') == compare.sha(PILOT / 'compare.py')
    source_rows = [r for r in json.loads(source.read_bytes())
                   if r['length'] == 32768 and r['task'] in compare.TASKS]
    assert Counter(r['task'] for r in source_rows) == {t: 50 for t in compare.TASKS}
    rows_by_hash = {r['input_sha256']: r for r in source_rows}
    assert len(rows_by_hash) == TARGET
    prior = json.loads((PILOT / 'selected.json').read_text())
    seen = set()
    rows = []
    for row in [*prior, *source_rows]:
        ident = row['input_sha256']
        assert rows_by_hash[ident] == row
        if ident not in seen:
            seen.add(ident)
            rows.append(row)
    assert len(rows) == TARGET and rows[:20] == prior[:20]
    for row in rows:
        assert len(row['input_ids']) == row['prompt_tokens']
        assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest() == row['input_sha256']
    selected = ROOT / 'selected.json'
    if selected.exists():
        assert json.loads(selected.read_text()) == rows
    else:
        compare.save('selected.json', rows)
    for i in range(20):
        file = f'quality_{i:03d}.json'
        if not (ROOT / file).exists():
            shutil.copyfile(PILOT / file, ROOT / file)
        assert compare.sha(ROOT / file) == compare.sha(PILOT / file)
    validate_results(rows)
    protocol.update(target=TARGET, reused_pilot_cases=20,
                    evaluation_metric='official RULER per-question target recall, mean over questions',
                    selected_sha256=compare.sha(selected), compare_sha256=compare.sha(ROOT / 'compare.py'),
                    coverage='all 50 existing 32K cases per task, including prior small-sample cases')
    compare.save('protocol.json', protocol)
    return rows


def validate_results(rows):
    methods = {'full'} | {f'{name}_{k}' for k in compare.BUDGETS for name in ('B', 'jssa')}
    for path in ROOT.glob('quality_*.json'):
        i = int(path.stem.split('_')[-1])
        case = json.loads(path.read_text())
        assert case['input_sha256'] == rows[i]['input_sha256'], path
        assert set(case['methods']) == methods, path
        assert all(0 <= r['ruler_score'] <= 100 for r in case['methods'].values()), path


def launch():
    rows = prepare()
    env = dict(os.environ, PYTHONPATH=str(compare.OLD / 'deps'),
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    jobs = []
    started = time.time()
    try:
        for gpu in range(4):
            with (ROOT / f'quality_{TARGET}_{gpu}.log').open('a') as log:
                jobs.append(subprocess.Popen([sys.executable, '-u', str(ROOT / 'compare.py'),
                                              'quality', str(gpu), str(TARGET)],
                                             env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                             stdout=log, stderr=subprocess.STDOUT))
        while True:
            done = len(list(ROOT.glob('quality_*.json')))
            codes = [p.poll() for p in jobs]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Workers exited with codes {codes}')
            if all(code == 0 for code in codes):
                break
            compare.save('status.json', dict(stage='quality', completed=done, target=TARGET,
                                           reused=20, started=started, pids=[p.pid for p in jobs]))
            time.sleep(5)
        validate_results(rows)
        compare.summarize(TARGET)
        compare.save('status.json', dict(stage='complete', completed=TARGET, target=TARGET,
                                       reused=20, seconds=time.time()-started))
    finally:
        for process in jobs:
            if process.poll() is None:
                process.terminate()
        for process in jobs:
            process.wait()


if __name__ == '__main__':
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            launch()
        except Exception:
            compare.save('status.json', dict(stage='failed', completed=len(list(ROOT.glob('quality_*.json'))),
                                           target=TARGET, traceback=traceback.format_exc()))
            raise
