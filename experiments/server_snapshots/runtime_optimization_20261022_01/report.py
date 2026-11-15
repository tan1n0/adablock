import hashlib, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=sorted([json.loads(line) for p in ROOT.glob('results_*.jsonl') for line in p.read_text().splitlines()],key=lambda r:r['index'])
assert len(rows)==9
comparison=json.loads((ROOT/'comparison.json').read_text())
differences=[]
for r in rows:
    for key,value in r['methods'].items():
        if key.startswith('baseline/'):continue
        original=r['methods']['baseline/'+key.split('/',1)[1]]
        if value['ids']!=original['ids']:
            differences.append(dict(index=r['index'],task=r['task'],length=r['length'],method=key,
                answers=r['answers'],baseline_prediction=original['prediction'],optimized_prediction=value['prediction'],
                baseline_score=original['score'],optimized_score=value['score']))
(ROOT/'quality_changes.json').write_text(json.dumps(differences,indent=2,ensure_ascii=False))
lines=['运行时优化对照结果','', '范围：9 道官方 RULER 题 × S/A/B × 两个配额；54 组比较，每组三个实现。',
       '这是实现回归小样本，不构成 S/A/B 优劣或质量无损的统计证明。','']
for version,report in comparison['comparisons'].items():
    lines.extend([version,
        f"逐 token 相同：{report['outputs_identical']}/{report['n']}；得分改变：{report['score_changes']} 组。",
        f"平均得分：{report['score_before']:.4f} → {report['score_after']:.4f}。",
        f"平均自由生成解码耗时：{report['decode_seconds_before']:.4f}s → {report['decode_seconds_after']:.4f}s。",
        f"平均端到端耗时：{report['e2e_seconds_before']:.4f}s → {report['e2e_seconds_after']:.4f}s。",
        f"输出截断：{report['truncated']}。",''])
for length in [8192,16384,32768]:
    r=json.loads((ROOT/f'verification_{length}.json').read_text())
    lines.extend([f'{length//1024}K 正式长度复核：',
        f"原版/优化版摘要构建：{r['index_seconds']['baseline']:.4f}s / {r['index_seconds']['gpu_digest']:.4f}s。",
        '固定 Full 输入轨迹、B、25% 配额，16 步，轮换顺序重复三次：'])
    for v,t in r['timing'].items():lines.append(f"  {v}: {t['median_ms_per_step']:.3f} ms/token（中位数）。")
    for version in ['gpu','gpu_digest']:
        tests=[v for k,v in r['decisions'].items() if k.endswith('/'+version)]
        lines.append(f"  {version} 首步选块相同层数：{sum(t['equal_layers'] for t in tests)}/{sum(t['total_layers'] for t in tests)}；预算类别一致：{sum(t['classes_equal'] for t in tests)}/{len(tests)}。")
    lines.append('')
lines.extend(['重要限制：','GPU 决策仍保留最大宽度标量读回与原 SDPA mask 分支同步。',
    'Gram 摘要保持数学上的 rank-4 目标，但浮点结果与 SVD 不完全相同，实际输出差异见 quality_changes.json。',
    '摘要构建每题计时一次；固定轨迹解码重复三次。旧评测与旧结果未修改。',
    '原始记录和逐题结果均留在本目录。'])
(ROOT/'report.txt').write_text('\n'.join(lines),encoding='utf-8')
(ROOT/'final_source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
print('\n'.join(lines))
sys.path.insert(0,'/root/adablock_ruler_harder_20261022_01/deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
values=[json.loads((ROOT/f'verification_{n}.json').read_text()) for n in [8192,16384,32768]]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for version in ['baseline','gpu','gpu_digest']:
    axes[0].plot([8,16,32],[v['index_seconds']['gpu_digest' if version=='gpu_digest' else 'baseline'] for v in values],marker='o',label=version)
    axes[1].plot([8,16,32],[v['timing'][version]['median_ms_per_step'] for v in values],marker='o',label=version)
axes[0].set_ylabel('Summary construction (seconds)');axes[1].set_ylabel('Decode (ms/token, median of 3)')
for ax in axes:ax.set_xlabel('Context (K tokens)');ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'runtime_comparison.png',dpi=170);fig.savefig(ROOT/'runtime_comparison.pdf')
(ROOT/'verification_status.json').write_text(json.dumps(dict(stage='complete',initial_32k_oom_preserved=True),indent=2))
