# 当前代码快照（2026-11-03）

2026-11-07 已直接连接 gpu027，补充 [服务器实时同步快照](server_snapshot_20261107_01/README.md)。下文为 11 月 3 日本地归档的说明；最新服务器版本及实际 JSSA 适配内核请查看新快照。

本次归档保留各轮实验的独立目录和原有实现，未改变训练目标、模型结构或评测配置。

| 内容 | 代码位置 |
|---|---|
| S/A/B 训练 | `sab90_20261019_01/` |
| 冻结排序器的覆盖目标对照 | `budget_rho_20261019_01/` |
| 32K LongBench 评测 | `longbench_sab90_32k_20261020_01/` |
| 固定读取预算 RULER 评测 | `ruler_official_prompt_20261021_01/` |
| 自主预算 RULER 评测 | `ruler_autonomous_20261022_01/` |
| JSSA 校准、适配与评测 | `jssa_20261023_01/` |
| GPU 预算特征与选块、快速摘要 | `runtime_optimization_20261022_01/` |
| 摘要稳定性与边界复核 | `digest_stability_20261022_01/` |
| 小排序器训练与评测 | `small_ranker_20261028_01/` |
| CUDA Graph 排序器与分项计时 | `ranker_execution_20261028_01/`，其中 `graph_ranker_v3.py` 为第三版 |
| 核查过的模型、训练和推理源码 | `method_audit_20261102_01/source/` |

各目录中的 `protocol.json` 和 README 记录对应实验配置。历史修正分支与诊断脚本保留在原目录，不能把不同分支的配置混用。

## 运行依赖

这些脚本是服务器实验代码快照，包含原有 `/root/adablock*` 绝对路径和跨实验目录导入。运行前需恢复对应目录布局，或按本机路径调整入口脚本。模型权重、数据集、缓存、中间张量和逐题输出保留在原存储位置，没有上传到 Git。

`sab90_20261019_01/vendor/` 和 `budget_rho_20261019_01/vendor/` 中的源码从已有本地实验归档恢复，仅恢复源码与配置，不覆盖现有文件。`method_audit_20261102_01/source_manifest.json` 记录核查时源码的原始路径和校验值。

JSSA 的第三方仓库单独获取，避免把完整 vLLM 仓库重复提交：

- JSSA：https://github.com/jiwonsong-dev/JSSA ，版本 `34ce5abd82453b16ca6d5f16a1a8ae50fac5b920`。
- vLLM：https://github.com/vllm-project/vllm ，版本 `39910f2b25aacc09f5e7f166cdf0030b19f8b9e8`。
- 官方补丁和安装入口：JSSA 仓库的 `vllm_patch/setup_vllm.sh`。
- 本地 `jssa_adapter.py` 使用 `kernel_adapter/` 下的三个 Triton 内核；运行时需要从对应官方补丁版本准备这些文件。当前适配器用于共同框架质量与耗时对照，官方 vLLM 吞吐结果采用另一套执行协议。

本次上传验证 Python 源码语法，不重新执行 GPU 训练或评测。
