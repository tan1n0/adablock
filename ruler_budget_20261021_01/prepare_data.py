import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import nltk
import html2text
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

MODEL='/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb'
TASKS=['niah_single_1','niah_multivalue','vt']
LIMITS={'niah_single_1':128,'niah_multivalue':128,'vt':30}
LENGTHS=[8192,16384,32768]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    nltk.data.path.insert(0,str(ROOT/'nltk_data'))
    for resource in ['punkt','punkt_tab']:
        with zipfile.ZipFile(ROOT/'nltk_source/packages/tokenizers'/f'{resource}.zip') as z:
            z.extractall(ROOT/'nltk_data/tokenizers')
    base=ROOT/'RULER/scripts/data/synthetic/json'
    cache=ROOT/'essay_sources';cache.mkdir(exist_ok=True)
    urls=(base/'PaulGrahamEssays_URLs.txt').read_text().splitlines()
    def fetch(url):
        filename=url.rsplit('/',1)[-1]
        target=cache/filename
        actual=url.replace('http://www.paulgraham.com/','https://www.paulgraham.com/')
        if 'github.com/gkamradt/' in url:
            target.write_bytes((ROOT/'essay_repo/needlehaystack/PaulGrahamEssays'/filename).read_bytes())
        if not target.exists():
            for attempt in range(3):
                try:
                    content=urllib.request.urlopen(actual,timeout=25).read()
                    target.write_bytes(content);break
                except Exception:
                    if attempt==2:raise
        raw=target.read_bytes()
        if '.html' in url:
            h=html2text.HTML2Text();h.ignore_images=True;h.ignore_tables=True
            h.escape_all=True;h.reference_links=False;h.mark_code=False
            # Match the upstream extractor's decoding and font selection.
            content=h.handle(str(BeautifulSoup(raw.decode('unicode_escape'),'html.parser').find('font')))
            kind=1
        else:content=raw.decode('utf-8');kind=0
        return kind,filename,content,dict(url=url,download_url=actual,sha256=sha(target))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        essays=list(pool.map(fetch,urls))
    essays.sort(key=lambda e:(e[0],e[1]))
    (base/'PaulGrahamEssays.json').write_text(json.dumps({'text':''.join(e[2] for e in essays)}))
    (ROOT/'essay_manifest.json').write_text(json.dumps([e[3] for e in essays],indent=2))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'deps'),NLTK_DATA=str(ROOT/'nltk_data'),
             PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    for length in LENGTHS:
        for task in TASKS:
            cmd=[sys.executable,str(ROOT/'RULER/scripts/data/prepare.py'),'--save_dir',str(ROOT/'data'/str(length)),
                 '--benchmark','synthetic','--task',task,'--tokenizer_path',MODEL,'--tokenizer_type','hf',
                 '--max_seq_length',str(length),'--num_samples','50','--random_seed','20261021',
                 '--model_template_type','meta-llama3']
            subprocess.run(cmd,env=env,check=True)
            source=ROOT/'data'/str(length)/task/'validation.jsonl'
            assert source.exists() and len(source.read_text().splitlines())==50, str(source)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rows=[]
    for length in LENGTHS:
        for task in TASKS:
            source=ROOT/'data'/str(length)/task/'validation.jsonl'
            batch=[json.loads(line) for line in source.read_text().splitlines()]
            assert len(batch)==50
            for r in batch:
                ids=tok.encode(r['input'],add_special_tokens=False)
                assert len(ids)+LIMITS[task]<=length,(task,length,len(ids))
                rows.append(dict(task=task,length=length,sample_index=r['index'],input_ids=ids,
                    answers=r['outputs'],max_new=LIMITS[task],prompt_tokens=len(ids),
                    source_sha256=sha(source),input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    (ROOT/'selected.json').write_text(json.dumps(rows))
    old=Path('/root/adablock_sab90_20261019_01')
    original=json.loads((old/'result_hashes.json').read_text())
    hashes={n:sha(old/'branches'/n/'best_pair.pt') for n in ['S','A','B']}
    assert all(h==original[f'branches/{n}/best_pair.pt'] for n,h in hashes.items())
    (ROOT/'manifest.json').write_text(json.dumps(dict(checkpoints=hashes,selected_sha256=sha(ROOT/'selected.json'),
        ruler_commit=subprocess.check_output(['git','-C',str(ROOT/'RULER'),'rev-parse','HEAD'],text=True).strip(),
        n=len(rows),protocol_sha256=sha(ROOT/'protocol.json'),
        length_stats={str(l):[min(r['prompt_tokens'] for r in rows if r['length']==l),max(r['prompt_tokens'] for r in rows if r['length']==l)] for l in LENGTHS}),indent=2))
    print('PREPARED',len(rows),flush=True)

if __name__=='__main__':main()
