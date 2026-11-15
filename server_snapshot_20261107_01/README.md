# gpu027 服务器代码快照（2026-11-07）

本目录直接通过 SSH 从 `gpu-027-llm` 读取，主机为 `autodl-container-54c14dadfc-d9cafdc0`。`root/` 镜像服务器 `/root/` 下的 AdaBlock 主项目及独立实验目录，只保存源码、配置、依赖说明和许可证。

## 检查结果

- 共同步 1,043 份文件，8,286,458 字节；每份文件均记录原始路径、SHA-256、大小、修改时间和权限。
- 下载后逐文件校验，再回服务器重新计算全部 1,043 份文件的哈希，全部一致。
- 与已有本地对应文件相比：294 份完全一致，18 份不同，731 份此前不存在。差异包括服务器主项目中的模型、oracle 构建、训练、评测代码和一份解码剖析脚本，逐项见 `comparison.json`。这些差异不表示文件在上次上传后才修改。
- 11 月 2 日核查的排序器、预算器、S/A/B 训练、快速摘要、自主预算控制器及 CUDA Graph v3 的源文件和配置，此次均找到并与原校验值一致。
- 在服务器仅使用 CPU 导入 S/A/B 核心模块，确认实际导入路径及哈希均包含在本快照中，未初始化 CUDA。记录见 `remote_recheck.json`。
- Python 源码已做语法检查。历史 `fasa_full_gap_audit_20261005/` 中五份文件是保留缩进的类方法摘录，独立解析会报缩进错误；去掉公共缩进后均通过，原文保持不变。详见 `verification.json`。

本次没有执行训练、生成回答或 GPU 性能评测。

## 主要入口

| 内容 | 本快照位置 |
|---|---|
| 服务器主项目 | `root/adablock/` |
| 原始排序器、预算器及源码依赖 | `root/adablock/results/llama_budget_rank_pilot_20261015_01/` |
| S/A/B 训练 | `root/adablock_sab90_20261019_01/` |
| LongBench 32K | `root/adablock_longbench_sab90_32k_20261020_01/` |
| RULER 自主预算 | `root/adablock_ruler_autonomous_20261022_01/` |
| JSSA 适配与校准 | `root/adablock_jssa_20261023_01/`，含实际使用的 `kernel_adapter/` |
| 小排序器对照 | `root/adablock_small_ranker_20261028_01/` |
| CUDA Graph 与解码剖析 | `root/adablock_ranker_execution_20261028_01/` |

服务器目录结构和源码字节原样保留，仓库原有目录也保留。运行时仍需准备原模型权重、数据集和环境，并恢复 `/root/adablock*` 的目录布局或调整脚本中的绝对路径。`manifest.json` 记录原始可执行权限；从 Git 恢复 shell 脚本时可按记录恢复权限，或使用 `bash` 调用。

## 归档范围

包含服务器主项目、历史实验里的项目源码，以及各独立实验目录的源码、协议和配置。`results/` 中仅归档符合源码或配置筛选条件的文件，不包含逐题结果或训练张量。

权重、数据、虚拟环境、安装包、缓存、日志和完整第三方仓库不在上传范围内。第三方版本信息记录在 `manifest.json` 的 `third_party`，对应许可证位于 `third_party_licenses/`。JSSA 的实际适配内核随本次源码一起归档。

本目录的 `.gitattributes` 关闭文本换行转换，以便 Git 中的源码字节与服务器 SHA-256 一致。
