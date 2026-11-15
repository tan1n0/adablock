import os,sys,json,time,hashlib,random,subprocess,traceback,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/adablock_sab90_20261019_01')
SOURCE=Path('/root/adablock_ruler_autonomous_20261022_01')
PY='/root/miniconda3/envs/ada/bin/python'

def dump(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(stage,**extra):dump('status.json',dict(stage=stage,time_unix=time.time(),**extra))
def run_stage(stage):
    jobs=[]
    for gpu in range(2):
        with (ROOT/f'{stage}_{gpu}.log').open('w') as f:
            jobs.append(subprocess.Popen([PY,'-u',str(ROOT/'evaluate_small.py'),stage,str(gpu)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false'),stdout=f,stderr=subprocess.STDOUT))
    status(stage,pids=[p.pid for p in jobs])
    while any(p.poll() is None for p in jobs):
        if any(p.poll() not in [None,0] for p in jobs):
            for p in jobs:
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'{stage} failed; workers stopped; inspect logs')
        if stage=='evaluation':
            completed=len(list((ROOT/stage).glob('partial_*.json')))
            status(stage,pids=[p.pid for p in jobs],started_questions=completed,total_questions=450)
        time.sleep(20)
    assert all(p.returncode==0 for p in jobs)

def main():
    assert not (ROOT/'status.json').exists()
    (ROOT/'selected.json').write_bytes((SOURCE/'selected.json').read_bytes())
    rows=json.loads((ROOT/'selected.json').read_text());assert len(rows)==450
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['task'],r['length'])].append(i)
    rng=random.Random(20261028);fixed=[];smoke=[]
    for key,ids in sorted(groups.items()):
        assert len(ids)==50
        smoke.append(ids[0]);fixed.extend([ids[0]]+rng.sample(ids[1:],9))
    assert len(fixed)==90 and len(smoke)==9
    dump('selection.json',dict(smoke=sorted(smoke),fixed=sorted(fixed),fixed_seed=20261028))
    pilot=Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
    protected=[SOURCE/'selected.json',SOURCE/'comparison.json',SOURCE/'gpu_controller.py',SOURCE/'baseline_controller.py',
        OLD/'core.py',OLD/'train.py',OLD/'qa_selected.json',OLD/'raw_data_hashes.json',pilot/'ranker/best.pt',pilot/'ranker/run_config.json',
        pilot/'source/scripts/train_block_ranker_batched.py',pilot/'source/models/block_ranker.py',pilot/'source/utils/ranking_batch.py']
    protected += [OLD/'branches'/n/'best_pair.pt' for n in ['S','B']]
    hashes={str(p):sha(p) for p in protected}
    dump('protocol.json',dict(architectures={'original':{'projection':64,'hidden':128,'input':213},'H64':{'projection':64,'hidden':64,'input':213},'P32H64':{'projection':32,'hidden':64,'input':117}},
        branches=['S','B'],hardware='two RTX3090 24GB, one process per GPU',
        training='Same original 3-epoch pilot pretraining, then original S/B schedule and seeds. Fresh small rankers, no weight slicing or test-QA distillation.',
        pretraining_coverage=.95,adaptation_and_budget_coverage=.90,deployment_quantile=.95,
        budget_initialization='Same original rho90 checkpoint and frozen normalizer for all; labels/features refreshed for each new ranker, same 40 epochs and final-stage validation CE checkpoint selection.',
        qa='Same frozen 450 RULER inputs as original autonomous evaluation, no truncation/prompt changes, max_new128 and same EOS/scorer.',
        fixed_qa='90 stratified questions, 10 per task/length; same questions for all six rankers; fixed32 and fixed128 middle blocks per KV head plus identical sink/local/current.',
        autonomous_qa='Full and all six rankers on 450 questions; each ranker uses its own trained budget model.',
        timing='9 task/length smoke prompts, equal fixed128 middle blocks, 16 forced decode steps, 3 rotating repeats. Also natural-output timings on all QA.',
        unchanged='FP32 rankers, original fast Gram summaries, mean_proj cache disabled, original padded attention, same budget features and quantile policy.',
        validation='Architecture/absolute budget checks, longest Full/all-read check, original Full/S/B output regression; training and QA split preserved.',
        keepalive='Idle GPU pulse watchdog; explicit pause marker throughout measured evaluation, then resumed.',
        protected_sha256=hashes,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    status('waiting_for_training')
    while True:
        states={}
        for arch in ['H64','P32H64']:
            p=ROOT/arch/'status.json';states[arch]=json.loads(p.read_text()) if p.exists() else {}
            assert states[arch].get('stage')!='failed',(arch,states[arch])
        if all(v.get('stage')=='complete' for v in states.values()):break
        status('training',architectures=states);time.sleep(20)
    (ROOT/'PAUSE_KEEPALIVE').write_text('Evaluation timing in progress; watchdog must not launch pulses.')
    # Allow any previously launched short pulse and completed trainers to exit.
    time.sleep(8)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not busy,('GPU still occupied before evaluation',busy)
    status('preflight')
    with (ROOT/'preflight.log').open('w') as f:
        subprocess.run([PY,'-u',str(ROOT/'evaluate_small.py'),'preflight'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false'),stdout=f,stderr=subprocess.STDOUT,check=True)
    run_stage('smoke')
    smoke=[json.loads(p.read_text()) for p in (ROOT/'smoke').glob('partial_*.json')]
    assert len(smoke)==9 and all(len(r['methods'])==19 and len(r['fixed128_profile'])==6 for r in smoke)
    assert not any(v['hit_generation_limit'] for r in smoke for v in r['methods'].values()),'Output cap hit in smoke; inspect before full evaluation'
    run_stage('evaluation')
    with (ROOT/'summarize.log').open('w') as f:
        subprocess.run([PY,str(ROOT/'evaluate_small.py'),'summarize'],stdout=f,stderr=subprocess.STDOUT,check=True)
    assert json.loads((ROOT/'comparison.json').read_text())['completed']==450
    assert all(sha(p)==h for p,h in hashes.items())
    dump('verification.json',dict(old_files_unchanged=True,protected_files=len(hashes),original_answers_verified=True))
    status('complete',old_files_unchanged=True,summary='report.txt',comparison='comparison.json')
    (ROOT/'PAUSE_KEEPALIVE').unlink()

if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc())
        if (ROOT/'PAUSE_KEEPALIVE').exists():(ROOT/'PAUSE_KEEPALIVE').unlink()
        raise
