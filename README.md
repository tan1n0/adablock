# AdaBlock 实验指南

本项目用于训练 AdaBlock 的轻量 block retrieval policy。核心流程是：

```text
非 LongBench 长上下文数据
        ↓
生成 full-attention oracle 标签
        ↓
训练 AdaBlock policy MLP
        ↓
后续接入 KV retrieval 推理并在 LongBench 上测试
```

注意：如果 LongBench 要作为最终测试集，不建议用 LongBench 数据训练 policy。训练阶段建议使用 Qasper、GovReport、NarrativeQA 和 Needle-in-a-Haystack 合成数据。

## 1. 环境准备

推荐 Python 版本：

```text
Python 3.10
```

创建环境：

```powershell
conda create -n adablock python=3.10 -y
conda activate adablock
```

安装依赖：

```powershell
pip install -r requirement.txt
```

默认基准模型使用：

```text
Qwen/Qwen2.5-7B-Instruct
```

如果 Hugging Face 下载较慢或需要访问私有模型，可以再执行：

```powershell
huggingface-cli login
```

## 2. 硬件建议

实验分为两部分：oracle 数据生成和 policy 训练。

`oracle 数据生成` 最吃显存，因为需要加载 Qwen2.5-7B-Instruct 并开启：

```python
output_attentions=True
output_hidden_states=True
use_cache=False
```

推荐配置：

```text
1 x RTX 3090 24GB:
    适合 debug
    max_length 建议 512 - 1024

2 x RTX 3090 24GB:
    适合小规模 oracle
    max_length 建议 1024 - 2048

4 x RTX 3090 24GB:
    适合第一版实验
    max_length 可尝试 2048 - 4096

A100/H100 80GB:
    更适合正式 4096 或更长上下文 oracle 生成
```

`policy 训练` 很轻，1 张普通 GPU 或 CPU 都能跑。

## 3. 准备非 LongBench 训练数据

使用脚本：

```text
scripts/prepare_training_data.py
```

支持数据源：

```text
qasper
govreport
narrativeqa
needle
```

其中 `needle` 是脚本内置合成数据，不需要下载。

先跑一个只含 Needle 的最小测试：

```powershell
python scripts/prepare_training_data.py `
  --output-jsonl data/train_needle_debug.jsonl `
  --sources needle `
  --needle-samples 20
```

正式准备非 LongBench 训练数据：

```powershell
python scripts/prepare_training_data.py `
  --output-jsonl data/train_non_longbench.jsonl `
  --sources qasper govreport needle `
  --max-samples-per-source 100 `
  --needle-samples 100 `
  --min-chars 1500 `
  --max-chars 24000
```

输出格式为 JSONL，每行类似：

```json
{"text": "Question:\n...\n\nContext:\n...\n\nAnswer:\n...", "source": "qasper"}
```

## 4. 生成 Oracle 数据

Oracle 数据是用 full attention 自动生成的监督标签。它告诉 policy：

```text
当前 token 应该召回多少 block
当前 token 能不能复用上一步 block set
当前 token 更依赖哪些 block 类别
```

相关脚本：

```text
scripts/build_oracle_dataset.py
```

先用小配置 debug：

```powershell
python scripts/build_oracle_dataset.py `
  --input-jsonl data/train_needle_debug.jsonl `
  --output-jsonl data/adablock_oracle_debug.jsonl `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --max-length 512 `
  --block-size 16 `
  --sample-stride 8 `
  --max-docs 5 `
  --dtype float16
```

确认能跑后，再生成正式 oracle：

```powershell
python scripts/build_oracle_dataset.py `
  --input-jsonl data/train_non_longbench.jsonl `
  --output-jsonl data/adablock_oracle.jsonl `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --max-length 2048 `
  --block-size 16 `
  --sample-stride 4 `
  --dtype float16
```

如果显存不够，按顺序降低压力：

```text
1. max_length: 4096 -> 2048 -> 1024 -> 512
2. sample_stride: 4 -> 8 -> 16
3. max_docs: 减少样本数
4. dtype: 使用 float16
```

Oracle JSONL 每行包含：

```text
hidden_state
query_drift
score_features
prev_feedback
budget_label
required_k
reuse_label
category_target
```

## 5. 训练 AdaBlock Policy

相关脚本：

```text
scripts/train_adablock_policy.py
```

先跑 debug：

```powershell
python scripts/train_adablock_policy.py `
  --train-jsonl data/adablock_oracle_debug.jsonl `
  --output-dir checkpoints/adablock_policy_debug `
  --epochs 1 `
  --batch-size 8
```

正式训练：

```powershell
python scripts/train_adablock_policy.py `
  --train-jsonl data/adablock_oracle.jsonl `
  --output-dir checkpoints/adablock_policy `
  --epochs 3 `
  --batch-size 64 `
  --lr 2e-4
```

训练完成后会保存：

```text
checkpoints/adablock_policy/last.pt
checkpoints/adablock_policy/best.pt
```

## 6. Policy 输入与输出

Policy 模型位于：

```text
models/adablock_policy.py
```

输入：

```text
hidden_state
query_drift
score_features
prev_feedback
```

输出：

```text
reuse_prob
budget_prob
category_prob
```

含义：

```text
reuse_prob:
    是否复用上一轮 retrieved block set

budget_prob:
    当前 token 召回多少 block
    bucket = {0, 4, 8, 16, 32, 64, 128}

category_prob:
    当前 token 对不同 block 类别的需求分布
    categories = local / input / generated / anchor / high_hit
```

## 7. 检查训练是否合理

训练结束后需要检查：

```text
loss 是否下降
budget_acc 是否高于随机
reuse_acc 是否稳定
```

还要检查 oracle 标签分布：

```text
budget_label 是否几乎全是小 K
reuse_label=1 的比例是否过高
category_target 是否几乎全是 local
```

如果标签几乎全是 local 或小 K，说明训练数据长程依赖不够，应增加：

```text
Qasper
Needle
更长的 GovReport
更大的 max_length
```

## 8. 后续 LongBench 测试设计

LongBench 建议只作为最终测试集，不参与 policy 训练。

推荐对比：

```text
Full attention
Fixed Top-K, K=16
Fixed Top-K, K=32
Fixed Top-K, K=64
AdaBlock without reuse
AdaBlock full
```

评估指标：

```text
LongBench task score
平均召回 block 数
retrieval frequency
attention FLOPs
KV transfer volume
latency
```

核心实验问题：

```text
1. AdaBlock 是否比固定 K 更省 block？
2. AdaBlock 是否能保持接近 full attention 的任务质量？
3. reuse predictor 是否减少 retrieval 次数？
4. category-aware allocation 是否优于 global Top-K？
```

## 9. 推荐从这里开始

如果是第一次跑，按这个顺序：

```powershell
python scripts/prepare_training_data.py `
  --output-jsonl data/train_needle_debug.jsonl `
  --sources needle `
  --needle-samples 20

python scripts/build_oracle_dataset.py `
  --input-jsonl data/train_needle_debug.jsonl `
  --output-jsonl data/adablock_oracle_debug.jsonl `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --max-length 512 `
  --block-size 16 `
  --sample-stride 8 `
  --max-docs 5 `
  --dtype float16

python scripts/train_adablock_policy.py `
  --train-jsonl data/adablock_oracle_debug.jsonl `
  --output-dir checkpoints/adablock_policy_debug `
  --epochs 1 `
  --batch-size 8
```

这三步跑通后，再扩大数据和上下文长度。
