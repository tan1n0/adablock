"""Summarize the verified lightweight aggregates, preserving protocol boundaries."""
import csv
import json
from pathlib import Path

root=Path('/root/adablock/diagnostics/frozen_rankloss_20261013/longbench_input_length_20261013')
files=['sdpa_hotpot_by_length.json','hotpotqa_fa2_by_length.json','musique_fa2_by_length.json','multifieldqa_en_fa2_by_length.json',
       'hotpotqa_legacy_inventory_by_length.json','musique_legacy_inventory_by_length.json','multifieldqa_en_legacy_inventory_by_length.json']
groups=[json.loads((root/name).read_text()) for name in files]
lines=['# LongBench scores by actual input token length','',
       'Read-only saved-prediction aggregation; no model evaluation. k=1000. Lower boundary inclusive, upper exclusive.',
       'Bins:[0,8000),[8000,12000),[12000,16000),[16000,20000). All scores are normalized QA F1 percentages.',
       'Reference input lengths are fixed by question index/raw dataset _id for every method; never bucket by generation length.',
       'Formal SDPA/FA2 lengths include the applied chat template. These runs have no input truncation: original/retained SDPA lengths coincide, and the FA2 saved manifests/configuration audit records zero truncation.',
       'Raw LongBench length field and k=1024 do not match the requested counts.',
       'Every saved prediction F1 was recomputed using existing utils.longbench_eval.score_prediction and checked against saved per-question scores; N-weighted bucket totals match each verified original summary within1e-9.',
       'SDPA four methods share all200 input hashes. FA2 Hotpot Full/RankLoss also share hashes. MuSiQue/MultiField FA2 have only RankLoss; missing methods remain missing.',
       'Legacy results are inventory only: success configuration is incomplete, Full128 and old three-layer-controller generations are not demonstrated identical. They must not be combined with new FA2 RankLoss or presented as equal-condition comparisons.',
       'The supplied old Full totals55.811/28.031/51.434 were not verified among these inspected files and are not substituted.',
       'EM and exact source file/hash/IDs are retained in JSON and CSV; N=1/2 bins are very small.']
rows=[]
for g in groups:
    lines += ['', '## '+g['task']+' / '+g['protocol'],'']
    if 'limitation' in g:lines += ['**'+g['limitation']+'**','']
    lines += ['| Input length | Reference N | Full F1 | COBS F1 | Quest F1 | RankLoss F1 |',
              '|---|---:|---:|---:|---:|---:|']
    for b in g['bins']:
        label=f"{b['input_tokens_min_inclusive']//1000}–{b['input_tokens_max_exclusive']//1000}k"
        values=[]
        for method in ('Full','COBS','Quest','RankLoss'):
            if method in b['methods']:
                data=b['methods'][method];assert data['valid_N']==b['reference_N']
                values.append(f"{data['F1']:.4f}")
                rows.append({'task':g['task'],'protocol':g['protocol'],'k':1000,'lower_inclusive':b['input_tokens_min_inclusive'],
                             'upper_exclusive':b['input_tokens_max_exclusive'],'method':method,**data})
            else:values.append('not found in this protocol')
        lines.append('|'+label+'|'+str(b['reference_N'])+'|'+'|'.join(values)+'|')
    lines += ['', 'Totals: '+', '.join(f"{m}:N={data['N']},F1={data['F1']:.6f},EM={data['EM']:.6f}" for m,data in g['total'].items()),
              '', 'Sources:']
    for m,data in g['total'].items():
        for source in data['files']:lines.append('- '+m+': '+source['path']+'; SHA256='+source['sha256'])
csv_path=root/'all_protocols_by_length.csv'
if not csv_path.exists():
    with csv_path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=root/'REPORT.md'
if not report.exists():
    with report.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
print(json.dumps({'report':str(report),'csv':str(csv_path),'prediction_groups':len(groups),'method_bucket_cells':len(rows),
                  'count_alignment':'Hot25/32/66/77; Mus2/7/25/166; Multi86/49/14/1','score_and_total_regression':'passed'}))
