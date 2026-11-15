# AdaBlock 当前代码架构

本文描述 2026-11-15 的工作树。项目在冻结的 Qwen2/Qwen2.5 基座上研究历史 KV 块的选择；Policy 和 BlockRanker 是两条独立的学习路线。

## 目录职责

| 目录 | 职责 | Git |
|---|---|---|
| models/ | 预算 Policy、块排序器、逐层逐头稀疏解码控制器 | 跟踪 |
| utils/ | Oracle、Q/K 摘要、排序数据与损失、LongBench 输入与评分 | 跟踪 |
| scripts/ | 可改参数复跑的数据、训练、评测与汇总入口 | 跟踪 |
| tests/ | unittest 自动测试；manual/ 放手动冒烟检查 | 跟踪 |
| experiments/presets/ | 实验运行参数与调度脚本 | 跟踪 |
| experiments/one_off/ | 依赖固定数据或 GPU 配置的单次实验脚本 | 跟踪 |
| experiments/code/ | 从旧运行目录收集、按内容去重的历史实验源码 | 跟踪 |
| experiments/runs/、experiments/archives/ | 历史运行快照与结果压缩包 | 忽略，保留本地文件 |
| server_snapshot_20261107_01/ 与根目录中带日期的实验目录 | 远端已提交的服务器源码与协议快照；索引见 CODE_SNAPSHOT.md | 跟踪，历史归档 |
| tools/reports/ | PPT 和文档生成器 | 跟踪 |
| docs/ | 架构说明与实验记录 | 跟踪 |
| data/、checkpoints/、results/、logs/ | 数据、权重、结果、运行日志 | 忽略，保留本地文件 |
| diagnostics/、.cache/、.venvs/、tmp/ | 诊断快照、下载缓存、环境和临时清单 | 忽略，保留本地文件 |

README.md 是使用说明；requirement.txt 列出主要 Python 依赖。源码采用从项目根目录导入 models、utils 和 scripts 的方式，目前还没有安装式 Python 包。

## 主流程

    非 LongBench 文档
      → build_strict_training_corpus / merge / pack
      → data/ 中的 train、validation JSONL
      → build_oracle_dataset：采样完整注意力与块质量
           ├─ 预算与复用标签 → train_adablock_policy → Policy checkpoint
           └─ 当前 Query、块摘要与真实质量 → build_ranking_dataset
                                                   → train_block_ranker 或 DDP 版本
                                                   → Ranker checkpoint
      → eval_longbench_adablock_sparse
      → results/longbench、results/policy、results/ranker 等

LongBench 默认用于最终评测，不进入正式训练集。eval_longbench_full.py 提供完整注意力基线。

### Policy 路线

models/adablock_policy.py 的 AdaBlockPolicy 输入上一 token 的隐藏状态、query drift、块评分特征与反馈，输出中间块预算桶和是否复用的概率。utils/block_oracle.py 将历史划分为固定 sink、固定最近窗口和可选中间块，并从完整注意力质量生成预算标签。推理入口是 scripts/eval_longbench_adablock_sparse.py 的 policy 等模式。

### Ranker 路线

scripts/build_ranking_dataset.py 复用 build_oracle_dataset.py 的采样器，按文档、层和 KV head 写排序分片。utils/ranking_data.py 管理元数据、分片和训练/验证文档隔离；utils/ranking_loss.py 生成成对比较并计算质量加权损失。models/block_ranker.py 将当前 Query、COBS 块摘要、位置和层头身份组合成候选块分数。train_block_ranker.py、批量版和 DDP 版是不同执行方式。

在线的 learned_qk 与 fixed_qk 模式使用 models/layer_head_sparse_qwen.py：每层每 KV head 选块，再从完整 KV 历史中 gather 所需 K/V。目前该控制器将完整 KV 历史放在 CPU，是准确率原型；不能把它的运行时间当作融合稀疏 kernel 的性能。学习排序器目前主要使用固定中间块预算，尚未接上为其重新训练的动态预算 Policy。

## 路径约定

1. 维护中脚本以项目根目录作为相对路径基准。直接运行 scripts/ 中的命令时，先切换到项目根目录。
2. experiments/presets/ 的 Shell 脚本会自行切换到项目根目录；Python 预设从自身位置计算根目录，启动子进程时以根目录为工作目录。
3. 新数据写入 data/，模型写入 checkpoints/，结果按主题写入 results/{longbench,policy,ranker,oracle,comparisons,llama,diagnostics,summaries,reports}/，运行日志写入 logs/。临时盘点和试跑放入 tmp/。
4. results/ 顶层仍有被旧命令引用的兼容路径；主题目录中的相对链接指向同一份内容。路径对应关系保存在 tmp/results_organization_manifest.json。
5. experiments/code/ 与 experiments/runs/ 是历史快照。其中的旧绝对路径和协议哈希保留原样，不作为新实验的路径模板。

## 验证与已知边界

- tests/test_block_ranker.py 检查排序标签、因果采样、GQA 真值、checkpoint 和在线/离线一致性。
- tests/test_ranking_batch.py 检查批量执行与逐场景执行的采样、梯度及更新等价性。
- tests/manual/smoke_block_ranker.py 用随机微型 Qwen 检查完整排序器流程。
- 当前自动测试主要覆盖 Ranker；Policy 与完整 LongBench 评测仍缺少相应的自动化检查。
- scripts/build_oracle_dataset.py 与 scripts/eval_longbench_adablock_sparse.py 各约 1500 行，是后续提取可复用模块的主要位置。
- experiments/presets/run_louiskv_accuracy_proxy_table2.py 仍引用缺失的 eval_longbench_louiskv_accuracy_proxy.py，当前不能作为可运行的完整预设。
