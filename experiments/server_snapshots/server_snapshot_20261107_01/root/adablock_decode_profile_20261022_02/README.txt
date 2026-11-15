独立解码耗时剖析，保留现有自主预算评测及结果。

远端目录：/root/adablock_decode_profile_20261022_01
启动器等待自主预算 status.json 为 complete，并等待 GPU 0 无其他计算进程。
如果自主预算失败，停止剖析，不自动继续。

范围：3 个任务 × 8K/16K/32K，每格取原评测第一道题；每题 S/A/B，共 27 个剖析配置。
每配置固定 16 个实际解码步（强制忽略 EOS），预热 1 次、无插桩计时 3 次、profiler 1 次。
逐配置检查重复输出和插桩前后输出一致；检查 GPU 独占；最后检查旧代码、结果和权重哈希。
这不是新一轮回答质量评测，也不是完整长输出的耗时统计。

六类主要范围：排序网络、预算特征、预算网络、选块索引、KV gather/复制、注意力。
预算特征包含分数排序；排序网络范围仅包含模型打分。
预算网络范围不包含网络输出后的 softmax/CDF/档位映射，后者归入其他开销。
摘要整理、Transformer 其他层、摘要增量更新等未标记工作归入其他 GPU 开销。
掩码判断及同步单列；补齐成本已经包含在 KV 与注意力中，不冒称其独立耗时。

GPU 百分比：各范围内 CUDA kernel duration 总和 / 全部 CUDA activity duration。
这不是端到端墙钟时间百分比；CPU 范围时间包含调度和等待，不能与 GPU 时间相加。
同时报告未插桩 ms/token 中位数及 profiler 本身的减速倍数。
无法在 CUDA kernel 总和占比与墙钟占比之间直接换算。

输出：status.json、protocol.json、case_*.json、operators_*.json、summary.json、report.txt。
每种长度保存 B 单值检索的一份 Chrome trace。
本轮只测量，不应用优化修改。
