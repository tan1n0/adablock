"""Generate a compact, auditable Chinese report from completed experiment artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    assert json.loads((ROOT / 'status.json').read_text())['stage'] == 'complete'
    results = json.loads((ROOT / 'results.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    rank_history = json.loads((ROOT / 'ranker/history.json').read_text())
    best_rank = max(rank_history, key=lambda r: r['mean_validation_coverage'])
    overall = results['splits']['test']['groups']['overall']
    baseline, adaptive = overall['fixed_64'], overall['adaptive_calibrated']
    counts = {s: json.loads((ROOT / f'extract_{s}_complete.json').read_text())['scenes']
              for s in ('rank_train', 'budget_train', 'validation', 'test')}
    lines = [
        'Llama 排序器—预算模型小规模试验（2026-10-15）', '',
        '状态：排序训练、按冻结排序生成预算标签、预算训练和固定预算对照均已完成。',
        '范围：独立文档上的离线注意力覆盖试验；未运行本轮 HotpotQA 答案生成评测，也未测端到端加速。', '',
        '主要发现',
        f'相对固定64档，自适应校准版平均中间块数从 {baseline["mean_middle_blocks"]:.2f} 降到 {adaptive["mean_middle_blocks"]:.2f}，减少 {100*(1-adaptive["mean_middle_blocks"]/baseline["mean_middle_blocks"]):.2f}%。',
        f'达到95%覆盖的场景比例从 {100*baseline["hit_95"]:.2f}% 提升到 {100*adaptive["hit_95"]:.2f}%，但仍略低于严格的95%场景目标。',
        f'直接取预算分类最大概率时，达标率只有 {100*overall["adaptive_argmax"]["hit_95"]:.2f}%；不能把校准版收益归到未经校准的CE分类器上。',
        '预算模型最佳验证轮为第5轮，后续训练损失继续下降而验证交叉熵上升，存在明显过拟合。',
        '这支持继续验证“先训练排序、再按实际排序训练预算”的方向；下一步应增加独立文档并检查真实生成质量，不能只增加训练轮数。', '',
        '模型与数据',
        'Meta-Llama-3.1-8B-Instruct；BF16；完整文本先分词，再按 token 截断。',
        '30 篇文档：排序训练 12，预算训练 6，验证 6，测试 6；覆盖 2048/4096/8192 tokens。',
        '四组原始组成文档 ID 和完整文本哈希互不重叠；没有使用 LongBench 题目训练。',
        '采样全部 32 层、8 个 KV 头，每 256 tokens 取一处位置；场景粒度为位置×层×KV 头。',
        f'实际样本数：{counts}', '',
        '训练方法',
        '排序器沿用现有 BlockRanker 与注意力质量差加权的 pairwise RankLoss；单卡训练 3 轮，batch=64，lr=2e-4。',
        f'按验证集固定均匀采样4096个场景的覆盖选择第 {best_rank["epoch"]} 轮排序器，然后冻结；预算训练及固定预算对照使用相同 SHA256 权重。',
        '块大小 16 tokens；固定保留开头 32 tokens、最近 32 个历史块和当前 token。',
        '预算标签：按冻结排序器给出的顺序累计真实注意力质量，选择达到 95% 覆盖的最小预算档。',
        '预算档：0/4/8/16/32/64/128/256/ALL；实际读取量按候选数截断，ALL 明确代表全部候选。',
        '本轮新建一个轻量预算分类器，目标为纯交叉熵；输入是当前 query、层/头身份、候选数量和排序分数摘要。',
        '它不是直接续训原有 previous_hidden_only/reuse 控制器；本轮不训练复用决策，也不加入成本损失。',
        '真实注意力权重与 fixed_mass 仅用于标签、评估，不进入预算模型输入。特征标准化只拟合预算训练集。',
        f'预算训练 40 轮，选择验证交叉熵最低的第 {results["selected_budget_epoch"]} 轮。',
        '保留原始 argmax 预测；另在验证集选择一个保守概率分位数，使至少 95% 验证场景达到 95% 注意力覆盖。',
        f'选定的概率分位数为 {results["calibrated_quantile"]}；测试集未参与模型或阈值选择。', '',
        '独立测试集结果（按场景平均）',
        '方法 | 平均中间块数 | 平均注意力覆盖 | 达到95%覆盖的场景比例 | 平均读取token比例',
    ]
    for name, row in overall.items():
        lines.append(f'{name} | {row["mean_middle_blocks"]:.2f} | {100*row["mean_coverage"]:.2f}% | '
                     f'{100*row["hit_95"]:.2f}% | {100*row["mean_attention_token_fraction"]:.2f}%')
    lines += ['', '解释与限制',
        '平均中间块数只统计额外检索的历史块；读取token比例另包含固定 sink/local/self 部分。',
        '覆盖指标是每个 KV 组内各 query 头注意力质量的平均值，不代表每个 query 头都达到同一阈值。',
        'oracle_budget_for_frozen_ranker 使用真实注意力选择预算，只是诊断上限，不能部署。',
        'results.json 同时保留固定部分不足95%的困难场景、每层、每文档及各位置区间的结果。',
        '同平均块数比较采用固定预算曲线的线性插值，仅供诊断；不是实际运行的混合策略，也不是速度测试。',
        '这些 query 来自全注意力 teacher-forcing 文本轨迹，尚未验证自由生成时的状态漂移。',
        '测试只有6篇独立文档，2.56万个相关场景不能视为2.56万个独立实验；不能据此宣称论文级泛化优势。',
        '该预算模型在获得所有候选排序分数之后决策，因此当前节省目标是注意力读取量，未节省排序计算。',
        '本轮还改变了预算输入与层/头粒度，不能把全部差异单独归因于“标签按排序器重算”；进一步因果消融需要统一架构。', '',
        '正确性与保留',
        '微型 Llama 的采集与原生 eager 注意力质量最大误差 8.94e-8。',
        '实际 BF16 Llama 抽查第0/15/31层最大误差约0.00133；包含低精度概率舍入与重算数值差异。',
        '标签测试覆盖错误排序、固定窗口已足够、预算超出候选数和禁止真实标签进入输入。',
        '每个场景的 fixed+candidate 注意力质量约为1；所有候选块在采样位置前已完整结束，避免未来信息。',
        '源代码快照、配置、数据划分、两个 best/last checkpoint、固定预算表和逐场景预测均独立保存。',
        '旧 Qwen/Llama 权重、旧 LongBench 结果和原仓库源代码未覆盖。', '',
        f'远程完整目录：{ROOT}',
        f'排序器 SHA256：{results["ranker_sha256"]}',
        '本地主要附件：report.txt、test_comparison.csv、results.json、protocol.json、ranker_best.pt、budget_best.pt、reproduction_bundle.zip。',
    ]
    (ROOT / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    fingerprints = {}
    for split in counts:
        path = ROOT / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        fingerprints[split] = {'jsonl_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'token_sequences': [{'doc_id': r['doc_id'], 'tokens': len(r['pilot_input_ids']),
                                'ids_json_sha256': hashlib.sha256(json.dumps(r['pilot_input_ids']).encode()).hexdigest()} for r in rows]}
    (ROOT / 'data_fingerprints.json').write_text(json.dumps(fingerprints, indent=2))
    files = [p for p in ROOT.glob('*.py')] + [ROOT / 'protocol.json', ROOT / 'source_sha256.json', ROOT / 'data_fingerprints.json']
    files += [ROOT / 'source' / name for name in (
        'models/block_ranker.py', 'utils/block_oracle.py', 'utils/qk_block_oracle.py',
        'utils/ranking_data.py', 'utils/ranking_batch.py', 'utils/ranking_batch_eval.py',
        'utils/ranking_eval.py', 'utils/ranking_loss.py', 'scripts/build_oracle_dataset.py',
        'scripts/build_ranking_dataset.py', 'scripts/train_block_ranker.py', 'scripts/train_block_ranker_batched.py')]
    with zipfile.ZipFile(ROOT / 'reproduction_bundle.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
    artifacts = [ROOT / 'ranker/best.pt', ROOT / 'budget/best.pt', ROOT / 'results.json', ROOT / 'protocol.json']
    (ROOT / 'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in artifacts}, indent=2))
    print(json.dumps({'test': overall, 'matched_block_cost': results['splits']['test']['matched_block_cost']}, indent=2))


if __name__ == '__main__':
    main()
