import json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'regression_report.json').read_text())
trace=json.loads((ROOT/'trace_analysis.json').read_text())
diagnosis=json.loads((ROOT/'diagnosis.json').read_text())
probe=json.loads((ROOT/'guard_probe.json').read_text())
rows=[json.loads(s) for p in ROOT.glob('regression_*.jsonl') for s in p.read_text().splitlines()]
summary['max_read_fraction_delta']=max(abs(r['reference']['mean_read_fraction']-r['guarded']['mean_read_fraction']) for r in rows)
assert summary['max_read_fraction_delta']<1e-6
selected=json.loads(Path('/root/adablock_runtime_optimization_20261022_01/selected.json').read_text())
previous=json.loads(Path('/root/adablock_runtime_optimization_20261022_01/comparison.json').read_text())
fractions=[r['guarded']['audit']['refined_blocks']/(32*8*(len(selected[r['index']]['input_ids'])//16)) for r in rows]
summary['mean_refined_fraction']=statistics.mean(fractions)
summary['max_refined_fraction']=max(fractions)
lines=['摘要优化掉分诊断与修复验证','',
    '结论以当前回归集为限，不宣称所有未来输入无损。',
    f"旧 54 组回归：{summary['tokens_equal']}/54 逐 token 一致；得分改变 {summary['score_changes']} 组。",
    f"平均得分：{summary['mean_scores']['reference']:.4f} → {summary['mean_scores']['guarded']:.4f}。",'',
    '一、掉分题与首次差异',
    '8K CWE，B，25% 读取配额。原版 80 分，快速 Gram 70 分。',
    '原版结尾：9. self-esteem 10. plight；Gram 结尾：9. pleasure 10. self-esteem。',
    '第 21 解码步、第 23 层、第 6 个 KV 头，query、预算类别与 73 块配额均相同。',
    '块 377（包含 plight）与块 382 处在选块边界，原分差仅 4.768e-7。']
for r in trace['first_selection_difference']['candidate_scores']:
    lines.append(f"块 {r['block']}：原分数 {r['reference']:.12f}，Gram 分数 {r['gram']:.12f}。")
lines.extend(['原排序是 377 在前，Gram 排序是 382 在前；微小舍入误差翻转离散决策。',
    '第一处输出 token 分歧出现在第 38 个 token。选块改变后，后续 query、预算、KV 状态继续分化。',
    '单独修复第一次块替换仍为 70 分，所以不能把最终掉分归因于一个块；连续边界复核才恢复原轨迹。','',
    '二、修复尝试'])
for name,r in diagnosis['versions'].items():lines.append(f"{name}: {r['score']:.0f} 分，初始摘要 {r['index_seconds']:.4f}s。")
lines.extend(['更高精度更接近数学分解，但不保证复现原 FP32 SVD 的临界排序。',
    'guarded：大多数块用快速摘要；选块分差或预算 CDF 距离阈值不超过 1e-4 时，用原 SVD 复核并缓存。',
    '阈值不读取任务答案、题目编号或指定块；预算、模型和提示词不变。',
    f"掉分题修复后 {probe['guarded']['score']:.0f} 分，回答逐 token 恢复原版。",'',
    '三、速度（54 组均值，含复核成本）'])
lines.append(f"原版：解码 {summary['mean_decode_seconds']['reference']:.3f}s，端到端 {summary['mean_e2e_seconds']['reference']:.3f}s。")
lines.append(f"仅 GPU 决策：解码 {previous['comparisons']['gpu']['decode_seconds_after']:.3f}s，端到端 {previous['comparisons']['gpu']['e2e_seconds_after']:.3f}s。")
lines.append(f"未加保护的 Gram：解码 {previous['comparisons']['gpu_digest']['decode_seconds_after']:.3f}s，端到端 {previous['comparisons']['gpu_digest']['e2e_seconds_after']:.3f}s。")
lines.append(f"边界复核：解码 {summary['mean_decode_seconds']['guarded']:.3f}s，端到端 {summary['mean_e2e_seconds']['guarded']:.3f}s。")
lines.extend([f"平均复核 {100*summary['mean_refined_fraction']:.2f}% 的前缀 head-block，最多 {100*summary['max_refined_fraction']:.2f}%。",
    '复核费用计入解码；与未加保护的 Gram 相比会变慢。计时为单次运行，没有重复计时置信区间。','',
    '四、额外未用题验证（B，25%）'])
for r in summary['fresh']:lines.append(f"{r['length']//1024}K：{r['scores']}，逐 token 一致 {r['tokens_equal']}。")
lines.extend(['','旧结果受保护哈希检查通过。全部原始日志、诊断和生成结果保留。',
    '局限：边界保护是经验阈值，不是严格的浮点误差证明。小样本通过不等于总体质量永不下降。'])
(ROOT/'report.txt').write_text('\n'.join(lines),encoding='utf-8')
(ROOT/'final_report.json').write_text(json.dumps(summary,indent=2))
(ROOT/'final_source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
print('\n'.join(lines))
