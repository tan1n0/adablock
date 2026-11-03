"""Completed-run integrity and matched-control verification; package reviewable results."""
import json
from pathlib import Path
import tarfile
import torch
import core as c


def equal_state(a,b):
    return all(torch.equal(a[k],b[k]) for k in a)


def main():
    torch.set_num_threads(4)
    root=c.ROOT
    protected=json.loads((root/'protected_hashes.json').read_text())
    verification=dict(protected_files_unchanged={path:c.sha(path)==value for path,value in protected.items()})
    initials={name:torch.load(root/'branches'/name/'initial_pair.pt',map_location='cpu',weights_only=True) for name in ['S','A','B']}
    verification['initial_rankers_identical']=all(equal_state(initials['S']['ranker_state'],p['ranker_state']) for p in initials.values())
    verification['initial_budgets_identical']=all(equal_state(initials['S']['budget_state'],p['budget_state']) for p in initials.values())
    verification['configs_identical']=all(p['config']==c.CONFIG for p in initials.values())
    selections=[json.loads((root/'branches'/name/'rank_selection.json').read_text()) for name in ['S','A','B']]
    verification['rank_selection_identical']=selections[0]==selections[1]==selections[2]
    inputs=[json.loads((root/'branches'/name/'inputs.json').read_text()) for name in ['S','A','B']]
    verification['inputs_identical']=inputs[0]==inputs[1]==inputs[2]
    reports={name:json.loads((root/'branches'/name/'validation.json').read_text()) for name in ['S','A','B']}
    verification['updates']={name:dict(rank_steps=r['rank_steps'],budget_steps=r['budget_steps'],
                                       original_pairs=r['original_pairs'],boundary_pairs=r['boundary_pairs'],seconds=r['seconds'])
                             for name,r in reports.items()}
    verification['rank_versions']={};verification['supervision_bound_to_ranker']={}
    for name in ['S','A','B']:
        phase_hashes=[]
        for phase in [0,1]:
            rank_phase=1 if name=='S' else phase
            pair=torch.load(root/'branches'/name/f'rank_{rank_phase}_pair.pt',map_location='cpu',weights_only=True)
            data=torch.load(root/'branches'/name/f'supervision_{phase}.pt',map_location='cpu',weights_only=True)
            verification['supervision_bound_to_ranker'][f'{name}_{phase}']=all(data[s]['ranker_state_sha256']==pair['ranker_state_sha256'] for s in ['train','validation'])
            phase_hashes.append(data['train']['ranker_state_sha256'])
        verification['rank_versions'][name]=phase_hashes
    s=torch.load(root/'branches/S/rank_1_pair.pt',map_location='cpu',weights_only=True)
    a=torch.load(root/'branches/A/rank_1_pair.pt',map_location='cpu',weights_only=True)
    verification['S_A_final_rankers_identical']=equal_state(s['ranker_state'],a['ranker_state'])
    verification['S_A_rank_max_abs_difference']=max(float((s['ranker_state'][k]-a['ranker_state'][k]).abs().max()) for k in s['ranker_state'])
    actions=[torch.load(root/'branches/B'/f'actions_{phase}.pt',map_location='cpu',weights_only=True) for phase in [0,1]]
    verification['B_changed_action_fraction']=float((actions[0]!=actions[1]).float().mean())
    rows=[json.loads(line) for f in root.glob('qa_[0-3].jsonl') for line in f.read_text().splitlines()]
    selected={r['source_index']:r for r in json.loads((root/'qa_selected.json').read_text())}
    verification['qa_complete']=len(rows)==32 and len({r['source_index'] for r in rows})==32
    verification['same_qa_as_previous']=c.sha(root/'qa_selected.json')==c.sha(c.PREVIOUS/'qa_selected.json')
    verification['prompt_hashes_match']=all(r['input_ids_sha256']==selected[r['source_index']]['input_ids_sha256'] for r in rows)
    verification['first_tokens_shared']=all(len({m['ids'][0] for m in r['methods'].values()})==1 for r in rows)
    verification['preflight']={f.name:json.loads(f.read_text()) for f in root.glob('qa_preflight_*.json')}
    collection_logs=list(root.glob('collect_*.log'))
    verification['shared_collection_wall_seconds']=max(f.stat().st_mtime for f in collection_logs)-(root/'collection_pids.json').stat().st_mtime
    assert all(verification['protected_files_unchanged'].values())
    for key in ['initial_rankers_identical','initial_budgets_identical','configs_identical','rank_selection_identical',
                'inputs_identical','qa_complete','same_qa_as_previous','prompt_hashes_match','first_tokens_shared',
                'S_A_final_rankers_identical']:
        assert verification[key],(key,verification[key])
    assert all(verification['supervision_bound_to_ranker'].values())
    assert len({(r['rank_steps'],r['budget_steps']) for r in reports.values()})==1
    assert reports['B']['boundary_pairs']>0
    c.dump(root/'verification.json',verification)
    files=[f for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts
           and 'raw_qa' not in f.parts and not f.name.startswith('supervision_')]
    c.dump(root/'result_hashes.json',{str(f.relative_to(root)):c.sha(f) for f in files})
    archive=root.parent/(root.name+'_results.tar.gz')
    with tarfile.open(archive,'w:gz') as package:
        for f in files+[root/'result_hashes.json']:
            package.add(f,arcname=str(Path(root.name)/f.relative_to(root)))
    print(json.dumps(dict(verification=verification,archive=str(archive),sha256=c.sha(archive)),indent=2),flush=True)


if __name__=='__main__':main()
