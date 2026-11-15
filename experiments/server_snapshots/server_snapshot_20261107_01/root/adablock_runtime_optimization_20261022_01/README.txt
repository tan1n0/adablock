运行时优化独立实验（2026-10-22）

远端目录：/root/adablock_runtime_optimization_20261022_01
现有评测代码、模型权重和结果均不修改。protocol.json 保存其 SHA256，并在结束时复核。
启动前发现旧 harder 评测已自行停止：32K CWE 的 Full 得分为 0，触发原有检查门槛。
本轮没有解除该门槛或重启旧评测。

三个实现版本：
baseline：原 controller.py 的独立副本。
gpu：预算特征、分类、预算分配与选块索引在 GPU 上；按原始配额与稳定排序规则执行。
gpu_digest：在 gpu 基础上，用 16x16 Gram 矩阵分解计算原来的 rank-4 摘要。
Gram 与 SVD 的数学目标一致，浮点结果不保证逐位一致；必须记录实际决策与输出差异。
fast_digest 的 svd 选项保留精简后的原分解，可用作保守回退，不是本轮 gpu_digest 结果。

保留的开销：
每层仍读回一个最大预算标量，用于确定现有矩形 SDPA 输入尺寸。
沿用原 selected_attention，其 valid.all() 分支仍存在一次同步。
尚未替换 gather/padding、GQA 展开、SDPA 或跨层流水线。
GPU 版本没有累计旧 audit.ragged_attention_calls 字段；其默认 0 不表示没有 padding，不用于本轮比较。
不重新训练模型，不改变 90% 训练覆盖目标或部署分位数 0.95。

验证与比较：
1. check_ops.py：GPU 特征误差、1000 组预算分配、边界选块、摘要二次型检查。
2. experiment.py preflight：短上下文缓存与新增摘要更新一致性。
   该短探针中固定窗口主导预算，不能替代正式长度选块检查。
3. 正式配对：旧 harder 的 9 道官方 smoke 输入原样复用。
   niah_multikey_3 / cwe / fwe，各 8K、16K、32K 一道。
   S/A/B，各 10% 与 25% 读取配额，各三个实现版本，共 162 个输出。
   使用原题的官方输出上限，EOS、预填充、评分、检查点保持一致。
   同一题共享预填充和原始摘要；优化摘要单独构建。
   所有自由生成计时均关闭 capture，保留逐题输出及得分，逐版本统计 token 是否相同。
4. verify_timing.py：三个正式长度上，每个分支、两个配额的首个解码步选块和预算类别对照。
   B、25% 配额，固定 Full 轨迹的 16 步解码，三个版本轮换顺序，各重复三次。
   报告中位数，避免输出长度差异影响速度判断。

时间口径：
摘要时间为完整前缀摘要构建的同步墙钟耗时。
解码耗时包含决策、索引、注意力、模型其他计算及新增摘要更新，不含前缀预填充和初始摘要。
端到端 = 同题共享预填充 + 对应摘要构建 + 解码。首个输出 token 来自共享完整预填充。
摘要构建本轮每题一次；固定轨迹解码重复三次。没有测在线服务多请求吞吐量。
初次 32K 重复计时因同时保留两份摘要而显存不足；首步复核通过，失败日志保留。
补测改为仅把当前版本摘要放在 GPU，切换传输放在计时之外，符合单版本实际运行的显存占用。
这 9 道题用于实现回归，不能据此宣称 S/A/B 的统计显著优劣或质量完全不受影响。

主要文件：
ops_checks.json / preflight.json：基础检查。
results_*.jsonl / partial_*.json：配对质量与计时原始结果。
comparison.json：总体比较。
verification_8192.json / verification_16384.json / verification_32768.json：正式长度决策与重复计时。
runtime_comparison.png / .pdf：摘要与固定轨迹解码耗时曲线。
status.json：配对质量阶段状态。
verification_status.json：后续正式长度验证与重复计时状态。

运行：使用 /root/miniconda3/envs/ada/bin/python。
experiment.py launch 完成基础探针后启动 4 GPU 质量比较。
finish.py 利用质量分片结束后空闲的 GPU，启动三个正式长度验证；全部完成后作图。
脚本拒绝覆盖已有运行；重新实验请创建新的独立目录。
