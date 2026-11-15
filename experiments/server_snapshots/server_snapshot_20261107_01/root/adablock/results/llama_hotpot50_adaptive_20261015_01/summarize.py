import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from prepare import ROOT, PILOT, sha


def padded_fraction(record, method):
    values=[]
    for step in record['steps']:
        pos=step['position']
        if method=='full':
            values.append(1.);continue
        n=max(0,(pos+15)//16-34)
        fixed=pos+1-n*16
        if method!='adaptive':
            values.append((fixed+16*min(n,int(method.removeprefix('fixed'))))/(pos+1));continue
        classes=np.array(step['budget_classes']).reshape(32,8)
        budgets=np.array([0,4,8,16,32,64,128,256,n])
        take=np.minimum(n,budgets[classes])
        logical=float((fixed+16*take).mean())
        assert abs(logical-step['mean_selected_tokens'])<1e-5
        values.append(float((fixed+16*take.max(-1)).mean()/(pos+1)))
    return float(np.mean(values)) if values else None


def summarize():
    assert json.loads((ROOT/'status.json').read_text())['stage'] in {'generation_complete','complete'}
    config=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in json.loads((ROOT/'source_sha256_frozen.json').read_text()).items():
        assert sha(ROOT/name)==digest, f'Inference source changed: {name}'
    assert sha(PILOT/'ranker/best.pt')==config['ranker_sha256']
    assert sha(PILOT/'budget/best.pt')==config['budget_sha256']
    selected=json.loads((ROOT/'selected.json').read_text())
    rows=[]
    for shard in range(4):
        assert json.loads((ROOT/f'worker_{shard}_status.json').read_text())['stage']=='complete'
        rows += [json.loads(line) for line in (ROOT/f'worker_{shard}.jsonl').read_text().splitlines()]
    rows.sort(key=lambda r:r['source_index'])
    assert len(rows)==50 and len({r['source_index'] for r in rows})==50
    for row, manifest in zip(rows,selected):
        assert row['source_index']==manifest['source_index']
        assert row['input_ids_sha256']==manifest['input_ids_sha256']
        assert set(row['methods'])==set(config['methods'])
        assert len({v['ids'][0] for v in row['methods'].values()})==1
        for method,r in row['methods'].items():
            r['mean_padded_length_fraction']=padded_fraction(r,method)
    groups={'overall':rows,'untruncated':[r for r in rows if not r['truncated']],
            'truncated':[r for r in rows if r['truncated']]}
    summary={}
    for group, subset in groups.items():
        summary[group]={}
        for method in config['methods']:
            records=[r['methods'][method] for r in subset]
            valid=[r for r in records if r['decode_steps']>0]
            steps=sum(r['decode_steps'] for r in records)
            seconds=sum(r['decode_seconds'] for r in records)
            summary[group][method]={'samples':len(subset),'f1':float(np.mean([r['f1'] for r in records])),
                'mean_logical_token_fraction':float(np.mean([r['mean_read_fraction'] for r in valid])),
                'mean_padded_length_fraction':float(np.mean([r['mean_padded_length_fraction'] for r in valid])),
                'mean_middle_blocks':float(np.mean([r['mean_middle_blocks'] for r in valid])),
                'mean_decode_seconds':seconds/len(subset),'decode_ms_per_token':1000*seconds/max(steps,1),
                'median_question_decode_ms_per_token':float(np.median([1000*r['seconds_per_decode_step'] for r in valid])),
                'p95_question_decode_ms_per_token':float(np.quantile([1000*r['seconds_per_decode_step'] for r in valid],.95)),
                'mean_total_seconds':float(np.mean([r['total_seconds'] for r in records])),
                'mean_new_tokens':float(np.mean([len(r['ids']) for r in records])),
                'output_limit_count':sum(r['hit_generation_limit'] for r in records),
                'zero_decode_steps':len(records)-len(valid)}
    rng=np.random.default_rng(20261015)
    indices=rng.integers(0,50,(10000,50))
    paired={}
    for baseline in ['full','fixed32','fixed64']:
        diff=np.array([r['methods']['adaptive']['f1']-r['methods'][baseline]['f1'] for r in rows])
        boots=diff[indices].mean(1)
        paired[baseline]={'mean_f1_delta':float(diff.mean()),'paired_bootstrap_95_interval':np.quantile(boots,[.025,.975]).tolist(),
                          'wins':int((diff>1e-8).sum()),'ties':int((np.abs(diff)<=1e-8).sum()),'losses':int((diff< -1e-8).sum()),
                          'identical_output_ids':sum(r['methods']['adaptive']['ids']==r['methods'][baseline]['ids'] for r in rows)}
    per_document_layers=[]
    class_hist=np.zeros(9,dtype=np.int64)
    for row in rows:
        layers=[]
        for step in row['methods']['adaptive']['steps']:
            n=max(0,(step['position']+15)//16-34)
            classes=np.array(step['budget_classes']).reshape(32,8)
            class_hist+=np.bincount(classes.ravel(),minlength=9)
            take=np.minimum(n,np.array([0,4,8,16,32,64,128,256,n])[classes])
            layers.append(take.mean(-1))
        if layers:per_document_layers.append(np.mean(layers,axis=0))
    layer_mean=np.mean(per_document_layers,axis=0)
    result={'protocol':config,'groups':summary,'adaptive_paired_comparison':paired,
            'adaptive_budget_diagnostics':{'mean_middle_blocks_by_layer_question_macro':layer_mean.tolist(),
                'class_histogram_over_decode_layer_head_calls':class_hist.tolist()},
            'mean_prefill_seconds':float(np.mean([r['prefill_seconds'] for r in rows])),
            'mean_index_seconds':float(np.mean([r['index_seconds'] for r in rows])),
            'peak_gpu_gib':max(r['peak_gpu_gib'] for r in rows),
            'notes':['Read fractions refer only to decode after the shared first token; full-attention prefill excluded.',
                     'Logical token count is selected valid K/V; padding length reflects actual rectangular attention shape, not hardware HBM counters.',
                     'Full KV cache retained on GPU; no claim of KV storage compression.',
                     'All methods are scored on identical selected input IDs; this is not the full official 200-question score.',
                     'Each sparse method is charged the same measured one-time index construction even though evaluation reuses immutable prefix summaries.',
                     'Bootstrap covers this length-selected subset only and does not establish broad generalization.']}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    fields=['group','method',*summary['overall']['full'].keys()]
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for group,methods in summary.items():
            for method,values in methods.items():w.writerow(dict(group=group,method=method,**values))
    with (ROOT/'per_question.csv').open('w',newline='') as f:
        fields=['source_index','source_id','original_prompt_tokens','prompt_tokens','truncated','answers']
        fields += [f'{m}_{metric}' for m in config['methods'] for metric in ['prediction','f1','mean_read_fraction','mean_padded_length_fraction','mean_middle_blocks','decode_seconds','total_seconds']]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            entry={k:row[k] for k in fields[:6]};entry['answers']=json.dumps(entry['answers'],ensure_ascii=False)
            for m in config['methods']:
                for metric in ['prediction','f1','mean_read_fraction','mean_padded_length_fraction','mean_middle_blocks','decode_seconds','total_seconds']:
                    entry[f'{m}_{metric}']=row['methods'][m][metric]
            w.writerow(entry)
    ablation=json.loads((ROOT/'ablation_results.json').read_text())
    lines=['Llama HotpotQA 50题：固定预算与自适应预算对照（2026-10-15）','',
           '四组生成及标签来源的离线消融均已完成；旧权重、旧结果未覆盖。',
           '本次为8K长度内的小规模验证，不是完整200题LongBench，也不能与此前55.70等Full结果直接对照。','',
           '主要判断',
           '自适应F1为57.43，Full为57.05，固定32为55.76，固定64为55.72；差值的配对bootstrap区间均跨0，尚无稳定优势证据。',
           '自适应平均读取111.08个中间块，高于固定32/64；目前缺少相近实际预算的固定预算对照，不能把F1差异归因于动态分配。',
           '自适应对Full仅1题F1提高、2题降低、47题持平。提高的一题把“Saturday, the day of the Kentucky Derby.”缩短为“Saturday”，主要是少输出解释带来的F1变化。',
           '未截断29题中自适应F1为58.23，低于Full的59.88；总体略高于Full的现象不能表述为通用质量提升。',
           '上一轮离线覆盖试验的省预算结果尚未迁移到真实生成；当前实现也没有实际加速。',
           '下一步应补充约112块的近似等预算固定对照，并采集真实解码位置的教师注意力，区分预算目标过于保守和预算预测偏差。','',
           '真实生成结果',
           '方法 | F1 | 有效选中token比例 | 补齐后计算长度比例 | 平均中间块数 | 解码ms/token | 含预填充和索引的秒/题']
    for method,r in summary['overall'].items():
        lines.append(f'{method} | {r["f1"]:.2f} | {100*r["mean_logical_token_fraction"]:.2f}% | {100*r["mean_padded_length_fraction"]:.2f}% | '
                     f'{r["mean_middle_blocks"]:.2f} | {r["decode_ms_per_token"]:.2f} | {r["mean_total_seconds"]:.2f}')
    lines+=['','截断分组：']
    for group in ['untruncated','truncated']:
        lines.append(f'{group}（{len(groups[group])}题）：'+', '.join(f'{m} F1={r["f1"]:.2f}' for m,r in summary[group].items()))
    lines+=['','自适应与同题固定64档的逐题比较：',json.dumps(paired['fixed64'],ensure_ascii=False),
            '差值区间来自50题配对bootstrap，仅作为本小样本的不确定性描述。','',
            '协议',
            'Meta-Llama-3.1-8B-Instruct；BF16；单模型单卡，4张3090按题并行，各题都跑四组。',
            '在200题中按完整提示token长度选最短50题：29题未截断，21题保留前4080和后4080 tokens。',
            '完整聊天提示一次分词，不重复添加BOS；所有组逐题input_ids哈希相同。',
            '最大生成32 tokens，贪心，按模型全部EOS token终止。提示全注意力预填充，首个输出token共享，之后才切换稀疏注意力。',
            '固定块大小16、sink两块、local32块；所有32层和8个KV头使用同一冻结排序器。',
            '预算模型与0.95概率分位数取自上一轮独立文档验证集；未按HotpotQA调参。',
            '最大上下文加输出不超过8192。选择50题时未使用答案或任何方法的输出分数。','',
            '耗时与读取量的含义',
            f'平均预填充 {result["mean_prefill_seconds"]:.2f} 秒/题；排序块摘要索引构建 {result["mean_index_seconds"]:.2f} 秒/题。',
            '有效选中token比例包含sink/local/self，按题平均，仅统计后续解码，不包含全注意力预填充。',
            '变长预算使用带有效位掩码的矩形张量；补齐后长度按每层8个KV头的最大选中长度计，通常高于有效选中比例。',
            '没有测GPU硬件内存流量；上述比例不能称作HBM字节减少。完整KV仍驻留显存。',
            '解码耗时包含排序、预算分类、选择、gather、注意力和缓存更新；总耗时另外加入预填充和索引。',
            '4张GPU并行评测共享主机，少数解码测时出现秒级离群点；results.json另保留按题解码ms/token的中位数和P95，不能将本原型耗时当作严格独占设备性能基准。',
            '索引在评测执行时共享一次，统计每个稀疏方法单独运行成本时均计入这份构建时间。',
            '这是未融合的正确性原型，必须以本表实际耗时判断，不能由token比例推导加速比。','',
            '标签来源消融（离线测试集，未增加HotpotQA生成组）',
            '只把预算标签改成理想注意力排序下所需预算；架构、输入、数据划分、随机种子、优化器、训练轮数和校准规则相同。',
            '训练标签改变1557/25344（约6.14%）；其他训练输入完全相同。',
            '标签来源 | 预测方式 | 平均中间块数 | 95%覆盖达标率']
    for name,arm in ablation['results'].items():
        for method,row in arm['splits']['test'].items():
            lines.append(f'{name} | {method} | {row["mean_middle_blocks"]:.2f} | {100*row["hit_95"]:.2f}%')
    lines+=['实际排序标签模型的验证选择分位数为0.95，理想排序标签模型为0.975；不同读取量下的覆盖率不能解释为同预算胜负。',
            '两者都是第5轮验证CE最优，后续过拟合；此单种子消融不能证明稳定优越性。','',
            '正确性与存档',
            '修正GQA调用路径后，两个真实提示的全部读取模式与原生全注意力生成token和logits完全一致。',
            '变长掩码与逐头独立注意力的FP32最大误差1.19e-7；在线批量排序与逐样本排序误差2.86e-6。',
            '验证全部32层均调用排序器/预算模型，验证新完成块的均值与协方差二次项与直接重算一致。',
            '每题答案、逐token记录、预算档位、输入哈希、模型调用计数、截断标记均保留。',
            f'远程目录：{ROOT}',
            '本地附件：comparison.csv、per_question.csv、results.json、protocol.json、preflight.json、ablation_results.json、reproduction_bundle.zip。']
    (ROOT/'report.txt').write_text('\n'.join(lines),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'reproduction_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.glob('*.py'):z.write(p,p.name)
        for name in ['protocol.json','selected.json','source_sha256_frozen.json','preflight.json','ablation_results.json']:
            z.write(ROOT/name,name)
        z.write(PILOT/'budget_pilot.py','pilot/budget_pilot.py')
        for name in ['models/block_ranker.py','utils/block_oracle.py','utils/qk_block_oracle.py','utils/ranking_data.py',
                     'utils/ranking_batch.py','utils/ranking_loss.py','utils/longbench_eval.py']:
            z.write(PILOT/'source'/name,'pilot/source/'+name)
    (ROOT/'artifact_sha256.json').write_text(json.dumps({p.name:sha(p) for p in [ROOT/'report.txt',ROOT/'results.json',ROOT/'comparison.csv',ROOT/'per_question.csv',ROOT/'reproduction_bundle.zip']},indent=2))
    (ROOT/'status.json').write_text(json.dumps({'stage':'complete','samples':50,'methods':4,'ablation':'complete'},indent=2))
    print(json.dumps({'groups':summary,'paired':paired},indent=2))


if __name__=='__main__':summarize()
