"""Generate unchanged official harder tasks in an isolated directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/adablock_ruler_budget_20261021_01')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    repo=SOURCE/'RULER'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit==PROTOCOL['official_commit'],(commit,PROTOCOL['official_commit'])
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','scripts'],text=True).splitlines()
    verified={}
    for name in tracked:
        if not (name.endswith(('.py','.yaml','.yml','.sh')) or name.endswith('/english_words.json')):continue
        expected=subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{name}'])
        assert (repo/name).read_bytes()==expected,f'Modified upstream source: {name}'
        target=ROOT/'RULER'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if expected.startswith(b'version https://git-lfs.github.com/spec/v1'):
            fields=expected.decode().splitlines()
            digest=next(line.split('sha256:')[1] for line in fields if line.startswith('oid '))
            size=int(next(line.split()[1] for line in fields if line.startswith('size ')))
            if not target.exists() or sha(target)!=digest:
                url=f'https://media.githubusercontent.com/media/NVIDIA/RULER/{commit}/{name}'
                print('FETCH_OFFICIAL_LFS',name,flush=True)
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                payload=opener.open(url,timeout=30).read()
                assert len(payload)==size and hashlib.sha256(payload).hexdigest()==digest
                target.write_bytes(payload)
            verified[name]=digest
            continue
        if target.exists():assert target.read_bytes()==expected
        else:target.write_bytes(expected)
        verified[name]=hashlib.sha256(expected).hexdigest()
    for name in ['deps','nltk_data']:
        if not (ROOT/name).exists():(ROOT/name).symlink_to(SOURCE/name,target_is_directory=True)
    sys.path.insert(0,str(ROOT/'deps'))
    from official import limit_for,metric_for,normalize_prediction
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(PROTOCOL['model'],local_files_only=True)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'extra_deps'),str(ROOT/'deps')]),NLTK_DATA=str(ROOT/'nltk_data'),
             PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    rows=[];audits=[]
    for length in PROTOCOL['lengths']:
        for task in PROTOCOL['tasks']:
            limit=limit_for(task)
            assert limit==PROTOCOL['official_caps'][task]
            subprocess.run([sys.executable,str(ROOT/'RULER/scripts/data/prepare.py'),
                '--save_dir',str(ROOT/'data'/str(length)),'--benchmark','synthetic','--task',task,
                '--tokenizer_path',PROTOCOL['model'],'--tokenizer_type','hf','--max_seq_length',str(length),
                '--num_samples',str(PROTOCOL['samples_per_cell']),'--random_seed',str(PROTOCOL['generation_seed']),
                '--model_template_type','meta-llama3'],env=env,check=True)
            source=ROOT/f'data/{length}/{task}/validation.jsonl'
            records=[json.loads(s) for s in source.read_text().splitlines()]
            assert len(records)==PROTOCOL['samples_per_cell']
            for line,row in enumerate(records):
                # Do not assume the upstream sample index is unique.
                assert row.get('answer_prefix'),(task,line,'missing answer prefix')
                prompt=row['input']+row['answer_prefix']
                # Official CWE prepends one complete worked example, including
                # its own chat template. Preserve it rather than stripping BOS.
                expected_bos=2 if task=='cwe' else 1
                assert prompt.count('<|begin_of_text|>')==expected_bos
                assert prompt.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'+row['answer_prefix'])
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                assert tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==prompt
                assert len(ids)+limit<=length,(task,length,len(ids),limit)
                assert row['outputs'] and all(isinstance(a,str) and a for a in row['outputs'])
                rows.append(dict(task=task,length=length,source_line=line,sample_index=row['index'],
                    input_ids=ids,answers=row['outputs'],max_new=limit,prompt_tokens=len(ids),
                    answer_prefix=row['answer_prefix'],source_sha256=sha(source),
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
            # Controlled metric checks use the actual official task scorer and cleanup.
            assert metric_for(task)([normalize_prediction('Alpha\x00BETA',task)],[['alpha','beta']])==100
            assert metric_for(task)(['nothing'],[['alpha','beta']])==0
            audits.append(dict(task=task,length=length,n=len(records),generation_cap=limit,source_sha256=sha(source)))
    rows.sort(key=lambda r:(r['length'],r['source_line'],PROTOCOL['tasks'].index(r['task'])))
    assert len(rows)==450
    (ROOT/'selected.json').write_text(json.dumps(rows))
    checkpoints=json.loads((SOURCE/'manifest.json').read_text())['checkpoints']
    manifest=dict(n=450,selected_sha256=sha(ROOT/'selected.json'),checkpoints=checkpoints,
        official_commit=commit,verified_official_sources=verified,cells=audits,protocol_sha256=sha(ROOT/'protocol.json'))
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PREPARED_AND_AUDITED',len(rows),flush=True)

if __name__=='__main__':main()
