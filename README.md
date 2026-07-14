# AdaBlock 实验指南

本项目训练一个轻量 KV block retrieval policy。当前版本不再预测 token/block
类别，而是使用更明确的三段式缓存：

```text
固定保留最早的 sink tokens
        +
固定保留最近的 local sliding window
        +
Policy 动态预测需要召回的中间 block 数
```

默认基座模型为 `Qwen/Qwen2.5-7B-Instruct`，默认 block size 为 16。

## 1. 当前方法流程

```text
非 LongBench 训练数据
    ↓ 文档去重、按 source 分层、文档级 train/validation 切分
train.jsonl + validation.jsonl
    ↓ 分别生成 full-attention Oracle
train_oracle.jsonl + validation_oracle.jsonl
    ↓ 选择 policy 架构并训练
MLP / Residual MLP / Feature Transformer
    ↓
固定 sink + 固定滑窗 + 动态中间块稀疏解码
    ↓
LongBench 测试
```

LongBench 默认只能用于最终测试。数据准备脚本会拒绝把
`longbench_*` 源放入训练集；只有显式添加
`--allow-longbench-training` 才允许进行有意的诊断实验。

旧 Oracle 包含 category 标签，预算含义也是“全部 block 数”；当前版本的预算
专指固定 sink/local 之外的中间 block 数。因此旧 Oracle 和旧 checkpoint
不应与当前版本混用。

## 2. 环境

推荐 Python 3.10：

```bash
pip install -r requirement.txt
```

Oracle 生成需要 full attention，是最消耗显存的阶段；policy 训练本身较轻。

## 3. 构造严格非 LongBench 训练数据

当前正式语料只使用与 LongBench 子任务不同的来源：

| 来源 | 总文档 | Train | Validation |
|---|---:|---:|---:|
| PG-19 | 750 | 675 | 75 |
| BookSum | 750 | 675 | 75 |
| arXiv Summarization | 1000 | 900 | 100 |
| PubMed Summarization | 1000 | 900 | 100 |
| WikiHop | 1000 | 900 | 100 |
| Multi-Needle | 1500 | 1350 | 150 |
| 合计 | 6000 | 5400 | 600 |

稳定复现命令：

```bash
bash scripts/run_build_strict_training_corpus.sh
```

构造器会下载约 1.2GB 的受控训练分片，逐来源生成后再合并，输出：

```text
data/train_strict_clean.jsonl
data/validation_strict_clean.jsonl
data/train_strict_clean_metadata.json
data/strict_sources/*
```

处理包括：

- 仅使用各数据源的训练分片，不读取其 test split；
- 按格式化文本 SHA-256 去重并生成稳定 `doc_id`；
- 在每个 source 内按整篇文档做 90/10 划分；
- 使用 6K/12K/20K/32K 字符长度课程；
- 对本地六个 LongBench 任务的 context 构建 13-word n-gram 指纹；
- 候选文档命中至少 3 个 LongBench 指纹时删除并继续补足来源配额；
- 合并后再次检查 train/validation 文档交集和 LongBench 指纹交集。

当前最终检查结果为：

```text
document_overlap = 0
longbench_ngram_overlap = 0
LongBench fingerprints = 1,431,429
```

详细来源、长度和删除统计记录在
`data/train_strict_clean_metadata.json`。

## 4. 分别生成训练和验证 Oracle

训练 Oracle：

```bash
python scripts/build_oracle_dataset.py \
  --input-jsonl data/train_strict_clean.jsonl \
  --output-jsonl data/oracle_train_clean.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-length 2048 \
  --block-size 16 \
  --sink-tokens 32 \
  --local-window-blocks 4 \
  --budget-buckets 0 4 8 16 32 64 128 \
  --sample-stride 8 \
  --oracle-layer-indices 12 20 27 \
  --dtype float32
```

验证 Oracle 使用完全相同的 Oracle 参数：

```bash
python scripts/build_oracle_dataset.py \
  --input-jsonl data/validation_strict_clean.jsonl \
  --output-jsonl data/oracle_validation_clean.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-length 2048 \
  --block-size 16 \
  --sink-tokens 32 \
  --local-window-blocks 4 \
  --budget-buckets 0 4 8 16 32 64 128 \
  --sample-stride 8 \
  --oracle-layer-indices 12 20 27 \
  --dtype float32
```

Oracle 标签含义：

- `budget_label`：固定 sink/local 后还需要多少个中间 block 才能达到目标 coverage；
- `required_k`：未离散化的中间 block 数；
- `reuse_label`：当前固定块加上一步中间块能否达到 reuse coverage；
- `fixed_sink_blocks`、`fixed_local_blocks`、`eligible_middle_blocks`：用于检查标签。

检查分布：

```bash
python scripts/inspect_oracle_dataset.py --jsonl data/oracle_train_clean.jsonl
python scripts/inspect_oracle_dataset.py --jsonl data/oracle_validation_clean.jsonl
```

## 5. 参数化选择 Policy 架构

`scripts/train_adablock_policy.py` 支持：

- `--architecture mlp`：原始轻量 MLP 基线；
- `--architecture residual_mlp`：可配置深度的残差 MLP；
- `--architecture feature_transformer`：把 hidden、query drift、各 score
  feature 和 feedback 编码成 feature tokens，再用小型 Transformer 聚合。

共有结构参数：

- `--projection-size`
- `--mlp-hidden-size`
- `--num-layers`
- `--num-attention-heads`
- `--dropout`
- `--budget-buckets`

训练 MLP：

```bash
python scripts/train_adablock_policy.py \
  --train-jsonl data/oracle_train_clean.jsonl \
  --val-jsonl data/oracle_validation_clean.jsonl \
  --output-dir checkpoints/policy_mlp \
  --architecture mlp \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-4
```

训练残差 MLP：

```bash
python scripts/train_adablock_policy.py \
  --train-jsonl data/oracle_train_clean.jsonl \
  --val-jsonl data/oracle_validation_clean.jsonl \
  --output-dir checkpoints/policy_residual_mlp \
  --architecture residual_mlp \
  --num-layers 3 \
  --mlp-hidden-size 256 \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-4
```

训练 Feature Transformer：

```bash
python scripts/train_adablock_policy.py \
  --train-jsonl data/oracle_train_clean.jsonl \
  --val-jsonl data/oracle_validation_clean.jsonl \
  --output-dir checkpoints/policy_feature_transformer \
  --architecture feature_transformer \
  --projection-size 64 \
  --num-layers 2 \
  --num-attention-heads 4 \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-4
```

必须保证训练时的 `--budget-buckets` 与 Oracle 构造时一致。

## 6. 每个 Epoch 的损失记录

每个输出目录会生成：

```text
best.pt
last.pt
metrics.jsonl
history.json
```

`metrics.jsonl` 每行对应一个 epoch，包含训练和验证的：

- `loss`
- `budget_loss`
- `reuse_loss`
- `cost_loss`
- `budget_acc`
- `reuse_acc`

示例：

```json
{
  "epoch": 1,
  "architecture": "residual_mlp",
  "lr": 0.0002,
  "train": {
    "loss": 1.2,
    "budget_loss": 0.7,
    "reuse_loss": 0.49,
    "cost_loss": 0.1
  },
  "validation": {
    "loss": 1.3,
    "budget_loss": 0.75,
    "reuse_loss": 0.54,
    "cost_loss": 0.1
  }
}
```

判断收敛时应分别观察：

- train/validation budget loss 是否都下降；
- reuse loss 是否真正下降，而不只是多数类准确率；
- cost loss 是否下降过快并把预算压到 0；
- validation total loss 是否开始回升。

最佳模型仍按 validation total loss 保存。

自动检查各项损失趋势并比较架构：

```bash
python scripts/summarize_policy_training.py \
  checkpoints/policy_mlp/metrics.jsonl \
  checkpoints/policy_residual_mlp/metrics.jsonl \
  checkpoints/policy_feature_transformer/metrics.jsonl \
  --output-json results/policy_architecture_comparison.json
```

输出会给出每项损失的首轮、末轮、最低值、最低 epoch、总变化和逐轮下降比例，
并按最佳 validation total loss 对架构排序。最终仍应结合 LongBench 质量与平均
选择 token 数，而不能只按验证损失决定模型。

## 7. 稀疏评测

```bash
python scripts/eval_longbench_adablock_sparse.py \
  --policy-checkpoint checkpoints/policy_residual_mlp/best.pt \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --tasks hotpotqa musique multifieldqa_en \
  --local-data-dir data/longbench \
  --output-dir results/policy_residual_mlp_b512 \
  --max-input-length 32768 \
  --block-size 16 \
  --sink-tokens 32 \
  --local-window-blocks 4 \
  --budget-tokens 512 \
  --selection-mode policy
```

选择顺序是：

1. 固定加入开头 `sink-tokens`；
2. 固定加入最近 `local-window-blocks`；
3. Policy 预测动态中间块预算；
4. 中间候选按 cosine score 选择；
5. 若预测 reuse，则复用上一步仍有效的中间块；
6. 可选 `--fill-budget-with-score` 补到总预算。

比较架构时必须使用同一份 train/validation Oracle、相同随机种子和相同 LongBench
样本，并同时报告任务分数、平均动态中间块数、平均总 token 数和 reuse rate。
