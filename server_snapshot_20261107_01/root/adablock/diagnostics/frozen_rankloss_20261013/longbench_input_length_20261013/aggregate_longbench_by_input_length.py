"""Only saved prediction files: aligned question buckets and explicit protocol separation."""
import collections
import hashlib
import json
from pathlib import Path
import re
import string
import sys

ROOT=Path('/root/adablock');RESULTS=ROOT/'results'
sys.path.insert(0,str(ROOT))
from utils.longbench_eval import score_prediction
BINS=((0,8000),(8000,12000),(12000,16000),(16000,20000))
EXPECTED={'hotpotqa':[25,32,66,77],'musique':[2,7,25,166],'multifieldqa_en':[86,49,14,1]}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(paths):
    rows={}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);index=int(row.get('index',row.get('idx')))
            if index in rows:raise ValueError(f'Duplicate question index{index}')
            rows[index]=row
    return rows


def norm(text):
    text=str(text).lower();text=''.join(c for c in text if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def qa_f1(prediction,answer):
    pred,truth=norm(prediction).split(),norm(answer).split()
    common=sum((collections.Counter(pred)&collections.Counter(truth)).values())
    if not common:return 0.
    p,r=common/len(pred),common/len(truth)
    return 2*p*r/(p+r)


def compute(task,protocol,files,known_totals):
    datasets={m:read_rows(paths) for m,paths in files.items()}
    reference=datasets.get('Full') or datasets['RankLoss']
    canonical=[json.loads(line) for line in (ROOT/'data/longbench'/f'{task}.jsonl').read_text().splitlines()]
    assert sorted(reference)==list(range(len(canonical)))
    lengths={i:int(row.get('retained_prompt_tokens',row.get('input_tokens',row.get('used_tokens',row.get('used_prompt_tokens'))))) for i,row in reference.items()}
    ids={i:canonical[i]['_id'] for i in reference}
    counts=[sum(lo<=n<hi for n in lengths.values()) for lo,hi in BINS]
    assert counts==EXPECTED[task],(task,protocol,counts)
    per_method={};scores={};ems={}
    for method,rows in datasets.items():
        assert set(rows)==set(reference)
        values={};em={};input_hash_equal=True;length_deltas=[]
        for i,row in rows.items():
            assert row['answers']==canonical[i]['answers']==reference[i]['answers']
            supplied=row.get('sample_id',row.get('id'))
            if supplied is not None:assert str(supplied)==ids[i]
            if 'input_ids_sha256' in row and 'input_ids_sha256' in reference[i]:
                assert row['input_ids_sha256']==reference[i]['input_ids_sha256']
            else:input_hash_equal=False
            pred=row['prediction']
            value=score_prediction(task,pred,row['answers'])
            saved=row.get('F1',row.get('score'))
            if saved is not None:assert abs(value-float(saved))<1e-9,(method,i,value,saved)
            values[i]=value;em[i]=float(any(norm(pred)==norm(a) for a in row['answers']))*100
            other=int(row.get('retained_prompt_tokens',row.get('input_tokens',row.get('used_tokens',row.get('used_prompt_tokens',row.get('prompt_tokens'))))))
            length_deltas.append(other-lengths[i])
        total=sum(values.values())/len(values)
        assert abs(total-known_totals[method])<1e-9,(method,total,known_totals[method])
        scores[method]=values;ems[method]=em
        per_method[method]={'N':len(rows),'F1':total,'EM':sum(em.values())/len(em),
                            'files':[{'path':str(p),'sha256':sha(p)} for p in files[method]],
                            'input_hash_equal_to_reference':input_hash_equal,
                            'input_length_delta_min':min(length_deltas),'input_length_delta_max':max(length_deltas)}
    bins=[]
    for lo,hi in BINS:
        selected=[i for i,n in lengths.items() if lo<=n<hi]
        bins.append({'input_tokens_min_inclusive':lo,'input_tokens_max_exclusive':hi,'reference_N':len(selected),
                     'question_indices':selected,'question_ids':[ids[i] for i in selected],
                     'methods':{m:{'valid_N':len(selected),'F1':sum(scores[m][i] for i in selected)/len(selected),
                                   'EM':sum(ems[m][i] for i in selected)/len(selected)} for m in datasets}})
    regressions={m:sum(b['methods'][m]['F1']*b['methods'][m]['valid_N'] for b in bins)/len(reference) for m in datasets}
    for m in datasets:assert abs(regressions[m]-per_method[m]['F1'])<1e-9
    return {'task':task,'protocol':protocol,'k':1000,'boundary':'lower inclusive, upper exclusive',
            'scorer':'utils.longbench_eval.score_prediction (existing repository)',
            'scorer_sha256':sha(ROOT/'utils/longbench_eval.py'),
            'reference_method':'Full' if 'Full' in datasets else 'RankLoss',
            'input_length_field':'retained_prompt_tokens or input_tokens or used_tokens, fixed from reference for all methods',
            'question_identity':'canonical test index and raw dataset _id; answers equal; input hashes asserted where stored',
            'total':per_method,'weighted_bucket_F1_regression':regressions,'bins':bins}


def main():
    sdpa=RESULTS/'hotpotqa_curve50k_threeway_20261005'
    group=compute('hotpotqa','SDPA_manual_32k_gen128_all28',{
        'Full':[RESULTS/'longbench_ranker_hotpotqa_full_20261004/full/full.jsonl'],
        'COBS':[sdpa/f'cobs/shard_{i}/cobs.jsonl' for i in range(4)],
        'Quest':[sdpa/f'quest/shard_{i}/quest.jsonl' for i in range(4)],
        'RankLoss':[sdpa/f'learned/shard_{i}/learned.jsonl' for i in range(4)]},
        {'Full':58.146025465235944,'COBS':56.897860314860566,'Quest':56.44081815626111,'RankLoss':57.65032410905455})
    target=ROOT/'diagnostics/frozen_rankloss_20261013/longbench_input_length_20261013'
    target.mkdir(exist_ok=True)
    path=target/'sdpa_hotpot_by_length.json'
    if path.exists():
        if json.loads(path.read_text())!=group:raise FileExistsError('Different existing aggregate preserved')
    else:path.write_text(json.dumps(group,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(path),'task':group['task'],'protocol':group['protocol'],
                      'total':{m:{k:v[k] for k in ('N','F1','EM','input_hash_equal_to_reference','input_length_delta_min','input_length_delta_max')} for m,v in group['total'].items()},
                      'bins':[{k:v for k,v in b.items() if k not in ('question_indices','question_ids')} for b in group['bins']]},ensure_ascii=False),flush=True)
    fa2root=RESULTS/'fasa_rankloss_musique_multifieldqa_20261006'
    specs=[('hotpotqa','FA2_HF4.43_gen32_all28_50k',{
            'Full':[RESULTS/'fasa_full_reproduction_20261005_multigpu/run/predictions.jsonl'],
            'RankLoss':[RESULTS/'rankloss_fasa_same_protocol_20261005/all_layers/run/predictions.jsonl']},
            {'Full':58.183626099231695,'RankLoss':58.12854460714301}),
           ('musique','FA2_HF4.43_gen32_all28_50k',{'RankLoss':[fa2root/'musique/all_layers/run/predictions.jsonl']},
            {'RankLoss':30.298314112436568}),
           ('multifieldqa_en','FA2_HF4.43_gen64_all28_50k',{'RankLoss':[fa2root/'multifieldqa_en/all_layers/run/predictions.jsonl']},
            {'RankLoss':53.309049046178565})]
    for task,protocol,files,totals in specs:
        g=compute(task,protocol,files,totals)
        g['missing_same_protocol_methods']=[m for m in ('Full','COBS','Quest','RankLoss') if m not in files]
        p=target/(task+'_fa2_by_length.json')
        if p.exists():
            if json.loads(p.read_text())!=g:raise FileExistsError('Different existing aggregate preserved')
        else:p.write_text(json.dumps(g,indent=2),encoding='utf-8')
        print(json.dumps({'output':str(p),'task':task,'protocol':protocol,
                          'missing_same_protocol_methods':g['missing_same_protocol_methods'],
                          'total':{m:{k:v[k] for k in ('N','F1','EM','input_hash_equal_to_reference','input_length_delta_min','input_length_delta_max')} for m,v in g['total'].items()},
                          'bins':[{k:v for k,v in b.items() if k not in ('question_indices','question_ids')} for b in g['bins']]},ensure_ascii=False),flush=True)
    old_full_roots={'hotpotqa':'full_hotpotqa_32k_gen128','musique':'full_musique_32k_gen128','multifieldqa_en':'full_multifieldqa_32k_gen128'}
    old_totals={'hotpotqa':{'Full':56.11673574665634,'COBS':55.27263438386045,'Quest':54.64450071236823},
                'musique':{'Full':28.005944352876128,'COBS':24.974517138489794,'Quest':25.777948986670378},
                'multifieldqa_en':{'Full':52.221463888281065,'COBS':50.20623521705718,'Quest':50.83923419213214}}
    for task in old_full_roots:
        files={'Full':[RESULTS/old_full_roots[task]/(task+'.jsonl')],
               'COBS':[RESULTS/'longbench_qk_3layer_cobs_louiskv_qk/policy'/(task+'.jsonl')],
               'Quest':[RESULTS/'longbench_qk_3layer_quest_louiskv_qk/policy'/(task+'.jsonl')]}
        g=compute(task,'LEGACY_inventory_NOT_VERIFIED_SAME_PROTOCOL',files,old_totals[task])
        g['limitation']='Full128 vs old3layer-controller results: generation/backend/exact runtime protocol not fully retained. Do not add new FA2 RankLoss or claim equal-condition comparison.'
        p=target/(task+'_legacy_inventory_by_length.json')
        if p.exists():
            if json.loads(p.read_text())!=g:raise FileExistsError('Different existing aggregate preserved')
        else:p.write_text(json.dumps(g,indent=2),encoding='utf-8')
        print(json.dumps({'output':str(p),'task':task,'protocol':g['protocol'],'limitation':g['limitation'],
                          'bins':[{k:v for k,v in b.items() if k not in ('question_indices','question_ids')} for b in g['bins']]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
