# AdaBlock 实验指南

## 可学习 COBS 块排序器（v1）

新增 `Query + COBS 摘要 → 共享 MLP → 每层/每 KV-head 块分数` 的训练与推理闭环。
本版先使用固定中间块预算，不启用 reuse，也不重训练预算 controller。
基座和原始 Key 摘要冻结；训练额外的 Query/均值投影、交互 MLP 和输出头。
每个 Query head 先独立形成非线性匹配特征，再聚合共享 KV 的头，避免直接平均 Query。

数据构造复用 `SampledAttentionCollector`：真实标签为每个 Query head 完整因果
softmax 后的块质量，再在 GQA 组内取平均。输入 Query 已经过 RoPE，并且只乘一次
attention scaling。均值、低秩协方差方向和特征值按文档/层共享存储；真实质量和
固定部分质量按采样位置保存。分片为 PyTorch 张量，加载时按需缓存，不保存完整
token attention 矩阵。所有元数据随 checkpoint 保存，训练/验证及推理会检查兼容性。

**v1 约束：**仅支持当前项目的 Qwen2/Qwen2.5 路径，且 `local-window-blocks >= 1`，
保证未完成的历史块固定保留、可排序块全部已完成。数据读取还会检查块末尾不晚于
当前可见历史。不要把本版称为 COBS 论文完整复现：本版使用 post-RoPE 摘要，固定
COBS 对照保留项目原有的 GQA 原始分数平均口径。

### 小规模构造、训练和评测

以下命令在装好项目依赖的环境运行（本机为 `conda activate ada`）。先用少量文档
验证，正式实验再扩大 `--max-docs`、`--max-length`。所有输出目录应为新的空目录。

```bash
python scripts/build_ranking_dataset.py \
  --input-jsonl data/train_strict_packed_tokens.jsonl \
  --output-dir data/ranker_train_probe \
  --layers 12 20 27 --block-size 16 --sink-tokens 32 \
  --local-window-blocks 32 --cobs-rank 4 \
  --sample-stride 16 --max-length 1024 --max-docs 2

python scripts/build_ranking_dataset.py \
  --input-jsonl data/validation_strict_packed_tokens.jsonl \
  --output-dir data/ranker_validation_probe \
  --layers 12 20 27 --block-size 16 --sink-tokens 32 \
  --local-window-blocks 32 --cobs-rank 4 \
  --sample-stride 16 --max-length 1024 --max-docs 2

python scripts/train_block_ranker.py \
  --train-dir data/ranker_train_probe --val-dir data/ranker_validation_probe \
  --output-dir checkpoints/ranker_cobs_probe \
  --epochs 10 --batch-size 8 --lr 2e-4 --coverage 0.95

python scripts/eval_block_ranker.py \
  --data-dir data/ranker_validation_probe \
  --checkpoint checkpoints/ranker_cobs_probe/best.pt \
  --budgets 4 8 16 32 64 128 --coverage 0.95 \
  --output-json results/ranker_cobs_probe.json
```

训练时按真实质量累计到 `coverage` 划分正例。每个场景最多抽 16 个正例，每个配
4 个高预测分数的困难负例及 4 个随机负例，另抽最多 32 个正例内部有序对。
这些数量可配置，与推理预算 K 无关。近似并列对过滤；无有效对的场景不参与 loss。
损失为 `sum((mass_i-mass_j)*softplus(score_j-score_i))/sum(mass_i-mass_j)`，
先按场景归一化，再对有效场景平均。不会生成 B×B 比较矩阵。

训练/验证严格检查原始文档组件和文本 SHA-256 交集；LongBench 来源不能用于构造
训练数据。采样器先打乱分片，再打乱片内场景，避免随机逐行读取反复装载大分片。
`history.json` 保存真实训练损失及验证指标，`best.pt` 根据各固定 K 的平均验证
coverage 选择；不同损失类型的数值不可直接比较。`--no-position` 可关闭位置特征。

### 固定预算稀疏解码

```bash
python scripts/eval_longbench_adablock_sparse.py \
  --selection-mode learned_qk \
  --ranker-checkpoint checkpoints/ranker_cobs_probe/best.pt \
  --ranker-untrained-layers cobs --fixed-middle-blocks 32 \
  --block-size 16 --sink-tokens 32 --local-window-blocks 32 --qk-cobs-rank 4 \
  --tasks hotpotqa --max-samples 2 --max-input-length 1024 \
  --max-new-tokens-override 16 --ranker-profile \
  --output-dir results/ranker_cobs_decode_probe
```

只训练 12/20/27 层时，其余层需要显式指定 `--ranker-untrained-layers cobs`；默认
会报错，避免未训练 embedding 静默参与预测。固定基线使用
`--selection-mode fixed_qk --qk-score-selector cobs`（也可选 `quest_aabb` 或
`mean_pool`），其余参数和输入保持一致，并使用另一个输出目录。

`learned_qk` / `fixed_qk` 中，`--fixed-middle-blocks` 决定中间块数，历史总上限
自动按 `sink + local + K` 推导；旧的 `--budget-tokens` 不参与这两种模式。
两种模式都在全部层执行检索，不使用旧 `--qk-layer-indices` 的限制；学习排序只
作用于 checkpoint 中的层，其他层执行显式的 COBS 回退。固定部分重叠、候选不足时
会减少实际读取量。历史 token 读取统计不包括始终追加的当前 token。

评测 JSON 报告 Coverage@K、中间块 mass、相对同预算 oracle 的 mass gap、平均
required_k 和预算饱和比例，并按当前历史长度分桶。required_k 是逐场景诊断，
并非已经重训练的共享预算模型输出。推理每条记录包含 checkpoint、实际层集合、
回退方式、有效预算和可选的阶段计时。`--ranker-profile` 会同步设备、增加计时
开销；当前 CPU KV 搬运实现是质量原型，不能据此宣称融合 kernel 的加速比。

### CPU 流程检查与后续预算衔接

```bash
python -m unittest discover -s tests -p 'test_block_ranker.py' -v
python scripts/smoke_block_ranker.py --output-dir /tmp/adablock_ranker_smoke_new
```

smoke 使用随机初始化的微型 Qwen 和合成 token，覆盖分片生成、训练、checkpoint
及评测，**不代表预训练模型的任务质量**。真实模型 CPU 数据探测可在构造命令上加
`--allow-cpu --device-map none --device cpu --local-files-only --cpu-threads 8`，
并将长度和文档数设得很小；数据构造只加载 backbone，不需要生成头。

后续冻结排序器后，可按 manifest 中的 source/doc_id 和 token_position 关联
现有预算 Oracle 的 source/source_doc_id/token_index。逐层逐头沿学习排序累计
真实质量，按既定层/头口径平均覆盖率曲线，再寻找达到阈值所需的共享 K、映射预算桶，
重新训练预算模型。不要直接沿用旧 COBS/Quest 标签，也不要把 ranker 的分数当作
校准后的注意力概率。本版尚未实现这一步预算重训练。

本次实现验证记录见 `results/block_ranker_v1_verification.json`：6 项核心测试通过，
随机微型 Qwen 完成训练和 CLI 验证；本地 Qwen2.5-7B 在 CPU 上完成了训练/验证各
1 篇、192 tokens 的提取，各得到 120 个场景，并完成 3 epoch 排序训练与 4-token
稀疏解码探测。探测 checkpoint 在 `checkpoints/block_ranker_cobs_v1_cpu_probe/best.pt`，
其配置为 block=16、sink=16 tokens、local=1 block、COBS rank=4、层 12/20/27，
与上方 local=32 的示例不同。该微型训练的验证效果尚未超过固定 COBS，不能作为
质量提升或推理加速结论；正式长上下文训练和 GPU 验证仍需另行运行。

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

### 3.1 按 Qwen Token 构造真实 4K/8K/16K/32K 序列

原始严格语料按字符数控制长度，`32K characters` 通常只有约6K～9K Qwen
tokens。使用 token packing 脚本在已经完成文档级划分的 train/validation
内部独立切块与拼接：

```bash
python -u scripts/pack_strict_training_corpus.py \
  --train-jsonl data/train_strict_clean.jsonl \
  --validation-jsonl data/validation_strict_clean.jsonl \
  --train-output-jsonl data/train_strict_packed_tokens.jsonl \
  --validation-output-jsonl data/validation_strict_packed_tokens.jsonl \
  --metadata-json data/strict_packed_tokens_metadata.json \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --length-buckets 4096 8192 16384 32768 \
  --min-fill-ratio 0.9 \
  --seed 42
```

默认 pack 数比例是 `4K:8K:16K:32K = 8:4:2:1`。由于 Oracle 每隔固定
stride 生成一行，这个反长度比例使四个长度桶贡献近似相同数量的训练行；可以
通过 `--bucket-pack-weights` 显式覆盖。每个 pack 记录原始组件文档、token
起止区间、目标长度和实际长度。脚本先检查原始文档 id，再在输出后检查组件
交集，保证 train/validation component overlap 为0。

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

### 32K 指定层 Sampled Attention Oracle

传统 `eager_full` 会先生成每一层完整的 `[heads, seq, seq]` attention，再选择
`--oracle-layer-indices`，因此不能扩展到 32K。`sampled_recompute` 使用 SDPA
完成正常模型前向，并只在指定层重算 `sample-stride` 对应的 query 行；每个
query chunk 会立即聚合成 block mass，不在 GPU 或 CPU 保存完整 attention 矩阵。

先运行小规模 probe：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/build_oracle_dataset.py \
  --input-jsonl data/train_strict_packed_tokens.jsonl \
  --output-jsonl data/oracle_train_packed_32k_causal_probe.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --max-docs 15 \
  --max-length 32768 \
  --block-size 16 \
  --sink-tokens 32 \
  --local-window-blocks 32 \
  --budget-buckets 0 4 8 16 32 \
  --sample-stride 64 \
  --transition-mode causal_next_token \
  --anchor-offset-mode document_hash \
  --oracle-layer-indices 12 20 27 \
  --attention-mode sampled_recompute \
  --attn-implementation sdpa \
  --attention-query-chunk-size 8 \
  --dtype bfloat16 \
  --device-map balanced_low_0 \
  --log-every-doc
```

该模式保持“全文内部采样 token”Oracle 定义，不要求文本具有 QA 或 Summary
target。`sample-stride=64` 只控制输出锚点密度；文档哈希 offset 会分散采样相位，
避免永远落在相同的 block 边界。`transition-mode=causal_next_token` 把 token `t`
的注意力需求作为监督目标，但 Policy 输入严格使用 `hidden[t-1]`。query drift
使用 `hidden[t-1]` 与 `hidden[t-2]`，score query 也只能访问 `t-1`，因此决策时
不会读取 `hidden[t]` 或 `attention[t]`。这版暂不输入 `embedding[t]`。
训练和验证必须使用完全相同的层、stride、后端、transition 和预算配置。
BF16 下 SDPA 与 eager 的中间 hidden 可能在预算桶边界产生少量标签差异；正式
train/validation Oracle 固定使用同一 `--attn-implementation sdpa`，不要混合。

Oracle 标签含义：

- `budget_label`：`required_k` 所落入的离散预算桶编号；
- `required_k`：达到目标 coverage 理论上需要的中间 block 数，不受最大预算限制；
- `budget_bucket_k`：该标签对应的中间 block 预算上限；
- `allocated_k`：考虑当前可选中间块数量后，实际分配的 block 数；
- `budget_saturated`：`required_k > allocated_k`，表示当前预算不能达到目标 coverage；
- `reuse_label`：当前固定块加上一步实际分配的中间块能否达到 reuse coverage；
- `oracle_policy_input_mode=previous_hidden_only`：Policy 状态输入为 `hidden[t-1]`；
- `attention_target_token_index=t`、`controller_token_index=t-1`：监督目标与决策状态错开一步；
- `oracle_transition_gap=1`：reuse 和 drift 均为相邻 token；
- `reuse_previous_selector=causal_cosine_topk`：前一步选块方式与推理代码一致；
- `fixed_sink_blocks`、`fixed_local_blocks`、`eligible_middle_blocks`：用于检查标签。

reuse 标签按照 `attention(t-1)` 推导出的预算桶分配 `allocated_k`，再使用
`hidden(t-2)`（在处理 `t-1` 前已可用）的 cosine top-k 选块，最后由
`attention(t)` 判断这些历史块能否复用。预算标签本身同样由 `attention(t)`
产生，但该值只作为训练监督，绝不作为 Policy 输入。
主 Oracle JSONL 已直接包含最终标签，不再需要 correction sidecar：

```bash
python scripts/train_adablock_policy.py \
  --train-jsonl data/oracle_train_packed_32k.jsonl \
  --val-jsonl data/oracle_validation_packed_32k.jsonl \
  --output-dir checkpoints/policy_mlp_packed_32k_causal \
  --architecture mlp \
  --budget-buckets 0 4 8 16 32 \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-4
```

训练默认拒绝不带 `causal_next_token_v1`、不是
`previous_hidden_only`、gap 不为1或不用 `causal_cosine_topk` 的旧 Oracle，
防止静默混入 gap64 或读取 `hidden[t]` 的标签。旧 correction sidecar 仅保留
用于历史消融，不用于新的完整构造。
新 checkpoint 同时写入 `policy_input_mode=previous_hidden_only`；LongBench Policy
评估默认拒绝缺少该标记的旧 checkpoint。仅做历史消融时才传
`--allow-legacy-policy-checkpoint`。

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

### 7.1 当前 Query 的逐层逐 KV-head 检索

实验性模式 `layer_head_qk_topk` 在处理当前 token 时进入每层 attention，使用
该层经过 RoPE 的真实 Query 与各 KV head 的历史 Key block centroid 做 cosine
评分；Qwen2.5-7B 中每个 KV head 对应的7个 Query heads先各自在 block 维度
归一化，再做 share-average。每层、每个 KV head独立 top-k，但每头的 sink/local
块和总预算完全相同。

`policy_layer_head_qk` 保留已有 Policy checkpoint 及其
`hidden[t-1] + query_drift + legacy score_features` 输入，仅用 Policy 预测中间块
数量；块的位置由当前 `query[t]` 在层内检索。该实验模式暂不复用上一 token 的
块集合。实现保留完整 KV 在 CPU 并按层/头 gather，目标是比较准确率，不代表
优化后的推理速度。

在同一批样本上比较原全局方案和逐层逐头方案：

```bash
MAX_SAMPLES=20 TASKS="hotpotqa" \
bash scripts/run_layer_head_qk_compare.sh
```

脚本同时运行固定512-token预算和兼容旧 Policy 的动态预算两组对照，并输出
`comparison.tsv/json`。固定预算对照用于隔离选块位置的影响；Policy 对照关闭
旧路径 reuse，以减少额外变量。

### 7.2 LongBench 块数趋势（快速抽样）

下面的脚本会在同一批 LongBench 样本上依次测试动态中间块数
`0, 4, 8, 16, 32, 64, ...`，直到达到 `MAX_INPUT_LENGTH` 对应的块数，最后
追加一个 full-attention baseline。开头 sink 和最近 local window 块始终保留，
因此这里的数字指“额外动态中间块”，不是总 KV token 数；结果会写入每个点的
目录，并汇总到 `trend_summary.tsv/json`。

脚本默认 `MAX_INPUT_LENGTH=32768`，与仓库的 32K LongBench 配置一致；它不是
模型的上下文限制。若只想先做低成本 smoke test，可以显式传入
`MAX_INPUT_LENGTH=8192`。32K、`BLOCK_SIZE=16` 时最大扫描点是 2048 块，运行
时间和显存占用会明显增加。

```bash
MAX_SAMPLES=5 \
MAX_INPUT_LENGTH=32768 \
TASKS="hotpotqa musique" \
bash scripts/run_longbench_block_trend.sh
```

默认使用不依赖 checkpoint 的 `score_topk`。可用
`METHOD=quest_topk` 或 `METHOD=policy POLICY_CHECKPOINT=...` 切换方法；也可以用
`BLOCK_POINTS="0 4 8 16 32 64"` 指定扫描点。正式跑完整数据前建议先用较小的
`MAX_SAMPLES` 检查显存和生成长度。

## 8. 实验配置与预算口径

仓库中先后存在不同阶段的实验配置，比较结果时必须同时记录 sink、local、
动态预算和总预算。`sink` 和 `local` 是固定保留区域；`dynamic middle` 是在
两者之外由 Oracle 或 Policy 选择的中间 block。

| 实验/数据 | sink | local | 动态 middle 预算 | 备注 |
|---|---:|---:|---:|---|
| 早期 Oracle、3072 LongBench | 2 blocks = 32 tokens | 4 blocks = 64 tokens | 0/4/8/16/32/64/128 | 历史方案 |
| 90% `oracle_train_packed_32k_causal.jsonl` | 2 blocks = 32 tokens | 32 blocks = 512 tokens | 0/4/8/16/32 | 32K causal Oracle |
| 95% `oracle_train_packed_32k_causal_cov095.jsonl` | 2 blocks = 32 tokens | 32 blocks = 512 tokens | 0/4/8/16/32 | 95% target、32-block cap |
| 之前部分六任务评测 | 2 blocks = 32 tokens | 4 blocks = 64 tokens | 总上限 512 tokens | 与 32K causal MLP 训练配置不一致 |
| LouisKV 论文（长输入任务） | 32 tokens | 512 tokens | 额外 512 tokens | 前两层保留完整 KV |

### 8.1 90% packed 32K Oracle 的预算标签

下面的分布来自 `data/oracle_train_packed_32k_causal.jsonl`，共 238,716 行：

```text
budget 0  ：30.40%
budget 4  ：10.47%
budget 8  ： 5.92%
budget 16 ： 9.45%
budget 32 ：43.76%
```

这些比例对应 `sink=2 blocks`、`local=32 blocks`，不是 `local=4 blocks`。标签
中的 `required_k` 是达到 90% attention-mass coverage 理论上需要的 middle block
数量；`budget_bucket_k` 是离散桶上限，`allocated_k` 还会受当前可用 block 数
限制。序列开头 sink 与 local 可能重叠，所以单行的 `fixed_local_blocks` 可以
小于 32，但配置仍是 `local_window_blocks=32`。

### 8.2 95% Oracle 统计

95% Oracle 使用与 90% packed 版本相同的模型、block、sink、local、采样 stride
和指定层，仅把 coverage target 从 0.90 改为 0.95。完整结果为 238,716 行、0
个损坏行：

| 长度 | budget 0 | budget 4 | budget 8 | budget 16 | budget 32 | `required_k > 32` |
|---|---:|---:|---:|---:|---:|---:|
| 0–4K | 31.37% | 7.55% | 5.70% | 9.52% | 45.87% | 30.42% |
| 4–8K | 9.53% | 4.40% | 3.32% | 6.05% | 76.70% | 64.75% |
| 8–16K | 2.04% | 5.48% | 2.35% | 4.67% | 85.46% | 75.94% |
| 16–24K | 0.00% | 3.01% | 1.42% | 2.53% | 93.04% | 87.29% |
| 24–32K | 0.00% | 0.64% | 0.73% | 1.14% | 97.49% | 94.68% |
| 总体 | 17.13% | 5.75% | 3.97% | 6.88% | 66.27% | 54.10% |

95% 版本的平均动态预算从 16.41 增至 22.85 blocks，平均总保留从 49.44 增至
55.88 blocks，约为 791→894 tokens。由于动态上限仍是 32 blocks，长上下文中
95% target 通常无法真正达到；它应称为“95% coverage target、32-block capped
Oracle”。

### 8.3 LouisKV 的预算换算

LouisKV 论文长输入任务使用 `S=32`、`W=512` 和 `B=512` tokens，并在前两层
保留完整 KV。按 token 数换算：

```text
32 sink + 512 local + 512 retrieval = 1056 tokens（最大可见缓存）
```

这与 AdaBlock 的 `sink=2 blocks`、`local=32 blocks`、`budget-tokens=1056`
上限在总 token 数上对齐，但 90% MLP 不一定每一步都输出 32 个动态 block。因此
应区分：

- **最大预算对齐**：AdaBlock 总上限为 1056 tokens；
- **平均预算对齐**：需要报告实际 `avg_selected_tokens`，不能仅看命令行上限；
- **实现完全对齐**：当前 AdaBlock 未复现 LouisKV 的前两层 full-KV 和语义段落
  触发机制，不能称为 LouisKV 的完整实现。

### 8.4 常用实验的准确命令口径

块数趋势脚本默认使用 `local-window-blocks=4`，扫描点表示额外 dynamic middle
blocks；因此例如 `m32` 的总预算不是 32 blocks，而是：

```text
2 sink + 4 local + 32 dynamic = 38 blocks = 608 tokens
```

逐层逐头受控对比脚本同样默认 `local-window-blocks=4`、`budget-tokens=512`，
用于隔离选块位置的影响。若要与 LouisKV 的长输入预算进行对比，应显式使用：

```bash
--sink-tokens 32 \
--local-window-blocks 32 \
--budget-tokens 1056
```

旧的 `local=4` 评测结果不能直接当作 `local=32` 训练配置下的结果；报告中应
把两者分开命名。

### 8.5 逐层逐头受控对比的解释

逐层逐头实验在相同输入、样本、预算和 `local=4 blocks` 下，将全局 block
选择改为每层、每个 KV head 独立选择。当前 Query `q_t` 只负责检索 block 位置，
因果 Policy 仍使用 `hidden[t-1]` 预测预算。已有 20 样本 HotpotQA 对比中：

- 固定预算下，逐层逐头选择比全局选择提高约 9.55 分；
- 动态预算下，逐层逐头选择比全局选择提高约 12.70 分；
- 动态方案平均读取量约从 396 降至 384 tokens，提升不是由读取更多 KV 导致；
- 后续 HotpotQA 200 样本验证应沿用同一 `local=4`、预算和输入配置单独报告。

这些结果属于 `local=4` 的历史受控实验，不能直接与 `local=32` 的 90%/95%
packed Oracle 结果合并平均。
