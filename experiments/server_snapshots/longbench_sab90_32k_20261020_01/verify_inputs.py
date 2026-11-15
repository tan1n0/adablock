"""Verify the length-only change against the preserved 8K run."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
old=Path('/root/adablock_longbench_sab90_20261020_01')
previous=json.loads((old/'selected.json').read_text())
current=json.loads((root/'selected.json').read_text())
manifest=json.loads((root/'manifest.json').read_text())
source_hashes=json.loads(Path('/root/adablock_sab90_20261019_01/result_hashes.json').read_text())
assert len(current)==len(previous)==300
for a,b in zip(previous,current):
    for key in ['task','source_index','source_id','question','answers','original_tokens']:
        assert a[key]==b[key],key
    assert b['prompt_tokens']==min(b['original_tokens'],32768)
    assert hashlib.sha256(json.dumps(b['input_ids']).encode()).hexdigest()==b['input_hash']
    if not a['truncated']:
        assert a['input_ids']==b['input_ids']
    else:
        assert a['input_ids']==b['input_ids'][:4096]+b['input_ids'][-4096:]
for n,h in manifest['checkpoints'].items():
    assert h==source_hashes[f'branches/{n}/best_pair.pt']
report=dict(same_300_questions=True,checkpoints_unchanged=True,tasks={})
for task in ['qasper','multifieldqa_en','musique']:
    rows=[r for r in current if r['task']==task]
    report['tasks'][task]=dict(n=len(rows),truncated=sum(r['truncated'] for r in rows),
                              max_input_tokens=max(r['prompt_tokens'] for r in rows))
(root/'input_verification.json').write_text(json.dumps(report,indent=2))
# Preserve the old run's status and annotate why it ended.
original=old/'status_before_32k_switch.json'
if not original.exists():
    original.write_bytes((old/'status.json').read_bytes())
(old/'status.json').write_text(json.dumps(dict(stage='stopped_for_32k_replacement',
    replacement=str(root),partial_results_preserved=True)))
print(json.dumps(report,indent=2))
