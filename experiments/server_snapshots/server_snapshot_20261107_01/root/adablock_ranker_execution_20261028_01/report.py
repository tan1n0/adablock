"""Summarize only completed paired measurements; no new model computation."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    micro = json.loads((ROOT/'micro_summary.json').read_text())
    version = 'v3' if (ROOT/'verification_v3.json').exists() else 'v2'
    decode = json.loads((ROOT/f'decode_{version}_summary.json').read_text())
    verification = json.loads((ROOT/('verification_v3.json' if version=='v3' else 'verification.json')).read_text())
    assert micro['completed']==18 and decode['completed']==18
    assert verification['old_files_unchanged'] and verification['original_answers_verified']
    cases = [json.loads(p.read_text()) for p in (ROOT/f'decode_{version}').glob('case_*.json')]
    assert len(cases)==18
    lines = ['排序器执行开销优化：完整结果', '',
        '独立目录：/root/adablock_ranker_execution_20261028_01',
        f'最终实现为 graph_ranker_{version}.py；第一版部分诊断记录保留在 decode/，未纳入正式结果。',
        '旧权重、训练、摘要、预算特征、排序规则及注意力实现均不变；仍然为每个新 query 重新算分。',
        '通过 CUDA Graph 复用原 FP32 运算的提交过程；不是缓存旧排序，不是改变预算，也没有降低数值精度。', '',
        '相同真实32K输入、3层、S/B，排序器单独测量（每种架构6组）：',
        '架构 | 原执行 ms/调用 | 执行图 ms/调用 | GPU kernel时间和 ms/调用 | kernel数/调用']
    for arch in ['original','H64','P32H64']:
        rr = [r for r in micro['cases'] if r['name'].startswith(arch+'_')]
        vals = [statistics.mean(r['eager']['median_wall_ms'] for r in rr),
                statistics.mean(r['graph']['median_wall_ms'] for r in rr),
                statistics.mean(r['kernel_sum_ms'] for r in rr),
                statistics.mean(r['kernel_count_per_call'] for r in rr)]
        lines.append(arch+' | '+' | '.join(f'{v:.4f}' for v in vals))
    lines += ['', f'上面的排序器微测试使用第一版图执行器，已检查分数一致；最终解码使用{version}。',
        'GPU kernel时间之和不等于墙钟耗时，其百分比不能作为总延迟占比。',
        '原版排序器GPU内部构成（实际kernel时间，不含CPU调用间隙）：']
    rr = [r for r in micro['cases'] if r['name'].startswith('original_')]
    total = statistics.mean(r['kernel_sum_ms'] for r in rr)
    for name in rr[0]['parts']:
        value = statistics.mean(r['parts'][name]['gpu_ms'] for r in rr)
        lines.append(f'  {name}: {value:.4f} ms/调用，{100*value/total:.2f}%')
    lines += ['', '完整解码：9道原始题 × S/B；各版本3次交替运行、每次16步，使用中位数后汇总。',
        '每次解码从空图缓存开始；包含图创建和历史增长后的重建耗时，不含预填充和摘要构建。',
        '长度 | 原版 ms/token | 优化 ms/token | 加速倍数 | 耗时下降 | 峰值allocated增量MiB | 峰值reserved增量MiB']
    for length, v in sorted(decode['by_length'].items(), key=lambda x:int(x[0])):
        lines.append(f"{length} | {v['eager_ms']:.3f} | {v['graph_ms']:.3f} | {v['speedup']:.3f} | "
                     f"{v['time_reduction_pct']:.2f}% | {v['peak_allocated_extra_MiB']:.2f} | {v['peak_reserved_extra_MiB']:.2f}")
    lines += ['', '自然回答的解码耗时（同一份原始答案、包含建图，按解码步数加权）：']
    for length in [8192,16384,32768]:
        rr = [r for r in cases if r['length']==length]
        vals = {v:1000*sum(r['answers'][v]['decode_seconds'] for r in rr)/
                    sum(r['answers'][v]['decode_steps'] for r in rr) for v in ['eager','graph']}
        lines.append(f"{length}: {vals['eager']:.3f} -> {vals['graph']:.3f} ms/token，下降{100*(1-vals['graph']/vals['eager']):.2f}%")
    lines += ['', f"逐层验证：{decode['audited_layer_steps']}个层步，原版与优化版排序分数、预算类别、选块和强制续写全部一致。",
        '32步验证跨越新历史块产生与图重建；正式回答沿用原官方prompt、EOS和128输出上限。',
        '18份自然回答与此前保存的原版回答逐token一致，逻辑读取量和补齐比例一致；没有输出上限截断。',
        '本轮是执行实现的回归验证，不是新增450题质量评测，也不用于证明S/B的质量优越性。',
        f"旧文件校验：{verification['protected_files']}个受保护文件保持不变。"]
    if (ROOT/'long_decode_check.json').exists():
        long = json.loads((ROOT/'long_decode_check.json').read_text())
        lines += ['', f"长生成检查：{long['prompt_tokens']}个输入token，128个解码步，{long['graph']['builds']}次图构建，4096次排序调用；",
            '每次排序分数与原实现完全一致，128步生成token也完全一致。该检查包含额外参考计算，不用作性能结论。']
    if (ROOT/'long_decode_memory_only.json').exists():
        long = json.loads((ROOT/'long_decode_memory_only.json').read_text())
        a,b = long['results']['eager'],long['results']['graph']
        lines += ['', '另测无额外参考计算、无逐层审计的128步空间占用：',
            f"峰值已分配显存：{a['peak_allocated_MiB']:.2f} -> {b['peak_allocated_MiB']:.2f} MiB，增加{b['peak_allocated_MiB']-a['peak_allocated_MiB']:.2f} MiB。",
            f"峰值框架预留显存：{a['peak_reserved_MiB']:.2f} -> {b['peak_reserved_MiB']:.2f} MiB，增加{b['peak_reserved_MiB']-a['peak_reserved_MiB']:.2f} MiB。",
            '预留显存包含已分配显存，两者不能相加。形状重建会增加缓存池预留，不能只报约40MiB张量增量；未验证超过128步的空间上限。']
    if (ROOT/'stage_profile_corrected.json').exists():
        stage = json.loads((ROOT/'stage_profile.json').read_text())
        corrected = json.loads((ROOT/'stage_profile_corrected.json').read_text())
        lines += ['', '补充分项剖析见 stage_profile_corrected.json 和 decode_trace_*.json。',
            '原始torch.profiler汇总漏归属了图中的kernel，已通过cudaGraphLaunch correlation逐个归属，原始记录保留。',
            '该剖析重复同一个32K首步，图已预热，单独报告；不能代替上面的含建图完整解码计时。']
        for name, item in stage.items():
            lines.append(f"{name}: warm wall {item['median_warm_wall_ms']:.3f} ms/token; GPU kernel sum {item['total_gpu_kernel_ms']:.3f} ms/token")
            if name in corrected:
                for label, v in corrected[name]['stages'].items():
                    lines.append(f"  {label}: GPU {v['gpu_ms']:.3f} ms/token ({v['gpu_percent']:.2f}%)")
        lines += ['CPU区间会等待前序图执行，预算特征CPU区间不能直接解释成预算特征纯计算耗时。']
    (ROOT/'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
