# Ranker 全量训练与排序评测记录（2026-10-03）

> 当前脚本位置（2026-11-15）：All28 单次训练与诊断脚本已移至 experiments/one_off/ranker_all28_20261003_05/。下文命令和 SHA-256 保留实验当时的原始记录。


本文记录本次工程内操作、复现入口和当前状态。原数据、原 checkpoint、用户已有未提交源码均保留。
平台核实时刻：2026-10-03 07:13:38 UTC。远端日志时钟为 +08:00，观察到比平台时间快约 10 分 26 秒；远端时刻不直接当作准确 UTC。

## 当前状态（自动维护）

<!-- RANKER_STATUS_START -->
最近观察的远端时刻：2026-10-04T03:11:12.961985+08:00（远端钟偏快约10分26秒；不是准确UTC）。

当前阶段：completed；流程PID：8339；子进程PID：无/已退出。

最近训练进度：epoch 10，2691192/2691192 场景；loss=0.12314279236051989；速度=885.9375561386976场景/秒。

已完成 epoch 数：10。

| Epoch | Train loss | 验证平均 coverage@K | 训练秒数 |
| --- | ---: | ---: | ---: |
| 1 | 0.171413 | 0.893756 | 3038.8 |
| 2 | 0.141009 | 0.894264 | 2994.4 |
| 3 | 0.134393 | 0.894823 | 2979.5 |
| 4 | 0.130525 | 0.894838 | 2977.5 |
| 5 | 0.128266 | 0.894970 | 2990.2 |
| 6 | 0.126627 | 0.895402 | 2980.3 |
| 7 | 0.125438 | 0.895519 | 2990.6 |
| 8 | 0.124592 | 0.895119 | 3041.5 |
| 9 | 0.123625 | 0.895362 | 3042.4 |
| 10 | 0.123143 | 0.894662 | 3037.7 |

独立测试已完成；场景数 44448；选中epoch 7。

| 方法 | Coverage@8 | Coverage@32 | Coverage@128 | Required k |
| --- | ---: | ---: | ---: | ---: |
| mean_pool | 0.831889 | 0.886512 | 0.946768 | 183.08 |
| quest_aabb | 0.828556 | 0.880438 | 0.949334 | 124.06 |
| cobs_legacy_mean_logits | 0.838231 | 0.900297 | 0.963372 | 91.45 |
| oracle | 0.860287 | 0.916784 | 0.970618 | 74.51 |
| learned | 0.852257 | 0.910366 | 0.967395 | 81.76 |
<!-- RANKER_STATUS_END -->

## 已完成的数据构造与检查

最初 32 序列小集已完成。随后全量命令取消了 --max-docs 32，但沿用非空输出目录，触发保护报错：
`Use a new output directory; existing ranking data will not be overwritten.`
该全量尝试没有开始。终端证据保存在
`results/ranker_audit_20261003_v1/screen_ada_snapshot.txt`。

修复只更换输出目录，保留原命令其余参数。全量指全部 packed 序列，不是所有 query；sample-stride 仍为 64。
成功运行退出码 0，用时 10680.29 秒（2小时58分）。

| 数据 | 路径 | 序列 | 分片 | 场景 |
| --- | --- | ---: | ---: | ---: |
| 旧小集，保留 | data/ranker_cobs_train_32k_v1 | 32 | 96 | 48,180 |
| 正式全量训练 | data/ranker_cobs_train_full_32k_20261003 | 1,751 | 5,253 | 2,691,192 |
| checkpoint 选择用验证 | data/ranker_cobs_validation16_20261003 | 16 | 48 | 44,364 |
| 最终独立测试（构造待执行） | data/ranker_cobs_test16_20261003 | 16 个已选源序列 | 待构造 | 待构造 |

全量训练分片共 59,455,553,394 bytes。manifest 文档、层组合、分片存在性、无临时分片、源文本哈希已校验。
验证提取用时 216.21 秒；全部验证分片检查通过：标签有限非负、候选不含未来 token，
fixed mass + candidate mass 归一化最大误差 3.58e-7。
全量构造写入器本身也检查摘要/query有限性、候选因果性和质量归一化。
旧 32 序列做过全分片额外复查；不要将其检查范围误写成对全量训练分片的额外逐张量复扫。

提取配置：Qwen/Qwen2.5-7B-Instruct；层 12/20/27；block=16；sink=32 tokens；
local=32 blocks；COBS rank=4；stride=64；query chunk=8；max length=32768；
BF16、SDPA、post-RoPE、device-map auto，仅使用已有本地模型。
GQA 标签为各 query head 完整因果 softmax 块质量，再在 KV-head 组内平均。
固定部分包含 sink、local 和 self，self 仅计一次。不启用 reuse。

### 划分与防泄漏

- 训练源为 data/train_strict_packed_tokens.jsonl 的全部 1,751 条（5,398 个原始组件）。
- 验证从既有独立验证源按 4K/8K/16K/32K 每档 4 条选择，seed=42；共 103 个组件。
- 剩余序列中，34 条因与所选验证共享原始组件而隔离，142 条保留作测试来源。
- 全量训练、16 条验证、142 条保留测试两两组件 ID 和完整文本 SHA256 交集均为 0。
- 在正式训练前，从保留测试用 seed=43 冻结每档 4 条，共 16 条。该测试不参与 checkpoint 选择或调参。
- 不按相邻 token 随机划分；token 场景仍随其原文档/序列归属。

证据：
`results/ranker_audit_20261003_v1/split_manifest.json`、
`full_source_isolation.json`、`full_manifest_audit.json`、`validation_audit.json`。
冻结最终测试：`results/ranker_audit_20261003_v1/batch_optimization/final_test_split.json`。
所有选择的输入 JSONL 与清单放在上述审计目录。

## 模型、目标与实现改动

排序器有 61,313 个参数。Query + COBS 摘要进入共享 MLP，每个 query-head 先做非线性匹配，
再聚合 GQA 组。没有训练基座模型。原入口 `scripts/train_block_ranker.py` 和
`models/block_ranker.py` 未修改，可以继续选择使用。

新增文件：

- `utils/ranking_batch.py`：变长候选拼接计算；连续候选使用 CPU 视图，非连续候选保留回退；
  批量搬运及一次 scores 回传；CPU 采样等值稳定排序/过滤；不传输训练用不到的 Quest 极值。
- `scripts/train_block_ranker_batched.py`：独立训练入口，复用原命令参数；新增进度、阶段计时、
  配置与中途状态保存。
- `utils/ranking_batch_eval.py`：批量 learned scorer；对同一批场景计算项目原有基线；
  增加 fixed-below-rho 分组、阈值命中率和 unreachable 说明。
- `tests/test_ranking_batch.py`：数值、采样和随机数状态对照。

保留全量 2,691,192 场景/epoch、batch=8、10 epoch、lr=2e-4、AdamW weight decay=0.01、
seed=42、梯度裁剪 1.0、模型维度与 FP32 ranker。不引入 AMP，不改变数据规模。
CPU threads=1 是显式实测后选择的执行参数。

损失仍为每场景 `sum((mass_i-mass_j)*softplus(score_j-score_i))/sum(mass_i-mass_j)`，
再对有效场景平均；空 pair 场景不参与 loss。保留 coverage=.95、正例最多16、困难/随机负例各4、
正例内部 pair 最多32、delta=1e-6、原场景顺序与 CPU torch RNG 消费顺序。
未冻结输出 bias 或改动 loss。

原实现备份：
`results/ranker_audit_20261003_v1/batch_optimization/reference/`。
正式启动的源码哈希：
`results/ranker_audit_20261003_v1/batch_optimization/source_sha256_at_formal_launch.json`。

## 等价性与数值限制

新增 3 项测试和原有 6 项核心测试通过。包含 CPU/CUDA、变长候选、空候选/空pair、
fixed 已达 rho、混合层/head、并列分数、固定 pair 的 loss/梯度/单步更新。
100 组随机与边界采样测试的 pair 和 RNG 状态精确一致。

真实数据 1,008 场景：特征精确一致，初始分数最大差 5.96e-8，pair 无差异，RNG一致。
32 步独立更新对照：pair/RNG 一致，非输出 bias 参数最大差 1.50e-6，
六个预算的验证 coverage 差异均为 0；同一模型的新旧 evaluator 已有指标误差均不超过1e-5。

**不是 bitwise 等价。**Pairwise 排序对统一分数偏置不敏感。输出 bias 的舍入级梯度经 Adam
可放大；32 步后该 bias 差约 0.001203，但中心化分数、pair、loss 和验证排序指标通过检查。
未把这一偏置差掩盖为参数逐位一致；正式进度记录继续输出 bias 供观察。
近并列分数仍可能因浮点舍入改变排序；现有实测没有采样 pair 差异，不能外推为永远零差异。

证据在 batch_optimization 目录：
`test_ranking_batch.py.log`、`test_block_ranker.py.log`、
`benchmark_v1.json`、`benchmark_v2.json`、`multistep_verification.json`。

## 性能与实际入口成本

原先 140 小时是小规模样本外推，存在冷加载放大等不确定性，不是实际完成时间。
第一轮分段测量观察到 CPU gather/加载、采样、逐 scene GPU launch/同步均有开销，
不能笼统说全部由磁盘导致。第二轮将热缓存训练与每 epoch 每分片一次加载分别估计。

四长度桶各252场景、三层、两次重复的长度加权预测：
原路径 threads=8 约82.15小时；批处理 threads=8 约37.79小时；
批处理 threads=1 约26.86小时（10 epoch 训练，不含验证）。
小样本和系统波动仍会影响估计，不作为推理加速比。

真实新入口另跑 32,768 训练场景：130.98秒、250.18场景/秒，loss=0.25414；
完整 44,364 场景验证：163.30秒。长度加权训练加10次验证估计 **27.31小时**。
实际入口前缀速度外推可能更慢，宜按约27–31小时规划并根据正式日志更新。
该 probe 不算正式模型，正式训练已重新初始化。

检查点：`checkpoints/ranker_batched_entry_probe_20261003`（probe）。
详细配置、时间与结果：`batch_optimization/entry_probe_command.json`、
`entry_cost_review.json`、`entry_probe.log`。
训练期间单卡 GPU0，其余GPU不被 ranker占用；测试提取后续沿用原 auto 分布方式。
不下载大模型、不使用付费 API、不新增机器。

## 可复现命令

在 /root/adablock，Python 为 /root/miniconda3/envs/ada/bin/python。
输出必须为新的空目录；重跑时替换输出路径，不能直接覆盖本次结果。

```bash
/root/miniconda3/envs/ada/bin/python scripts/build_ranking_dataset.py \
  --input-jsonl data/train_strict_packed_tokens.jsonl \
  --output-dir data/ranker_cobs_train_full_32k_20261003 \
  --model-name Qwen/Qwen2.5-7B-Instruct --layers 12 20 27 \
  --block-size 16 --sink-tokens 32 --local-window-blocks 32 --cobs-rank 4 \
  --sample-stride 64 --query-chunk-size 8 --max-length 32768 \
  --dtype bfloat16 --device-map auto --local-files-only

/root/miniconda3/envs/ada/bin/python -u scripts/train_block_ranker_batched.py \
  --train-dir data/ranker_cobs_train_full_32k_20261003 \
  --val-dir data/ranker_cobs_validation16_20261003 \
  --output-dir checkpoints/ranker_batched_full_20261003 \
  --epochs 10 --batch-size 8 --lr 2e-4 --coverage 0.95 \
  --seed 42 --cpu-threads 1 --device cuda:0
```

正式配置的所有默认值另见 checkpoint 目录的 run_config.json。
验证提取只替换 input/output 路径，使用已保存的 validation_selected.jsonl；
最终测试提取使用 final_test_selected.jsonl。完整命令均保存在阶段 command/result JSON。

## 日志、产物和待办

审计根目录 A=`results/ranker_audit_20261003_v1`；
优化/正式流程目录 B=`A/batch_optimization`。

- 全量构造日志/退出状态：A/full_build.log、A/full_build_status.json。
- 旧接续完成了验证与初次sanity，停在成本审核；不再负责正式训练。
- 正式流程管理 PID 2166，训练启动 PID 2190；PID只代表启动时状态，以 B/formal_pipeline_status.json 和系统进程为准。
- 正式训练：B/formal_training.log、B/formal_training_result.json（退出后写）、B/formal_train_command.json。
- 正式 checkpoint：checkpoints/ranker_batched_full_20261003/{best.pt,last.pt,history.json,progress.json}。
  best/last 在epoch完成后写；没有文件不应误报训练完成。
- 中途每100,000场景写 progress_state.pt，包含模型/优化器与RNG状态；**尚未提供自动恢复入口**。
- 完整训练结束后写 TRAINING_COMPLETE；管理流程随后构造冻结测试、检查隔离、运行质量评测。
- 最终报告（未运行前不存在）：B/validation_comparison.json、B/heldout_test_comparison.json、
  B/final_comparison_summary.json。
- 流程结束标记 B/PIPELINE_COMPLETE；失败 B/FORMAL_PIPELINE_BLOCKED。失败不会覆盖或删除旧产物。

评测比较 learned、cobs_legacy_mean_logits、quest_aabb、oracle（另保留 mean_pool）。
相同query、层/head、候选块、blocksize、预算 K=4/8/16/32/64/128，rho=.95。
报告 coverage@K、middle mass、oracle mass gap、required_k、阈值命中率；
分别统计 all、fixed部分不足rho、各历史长度桶。
required_k 无法达标时上限截为候选数，并单独报告 unreachable_rate。
COBS 是项目 post-RoPE/GQA平均原始分数版本，不宣称论文完整复现。
验证用于选checkpoint；最终独立测试用于最终质量比较。
离线评测耗时不等于线上推理速度；当前不作推理加速结论。

待完成：全量10epoch训练、冻结独立测试提取与校验、正式模型统一比较、最终结论文档更新。

## 与此前预算模型训练耗时的对照（2026-10-03）

此次对照只读检查了既有 checkpoint/日志，没有修改或重启当前训练。
基准是最近的 `checkpoints/adablock_policy_qk_3layer_cobs/last.pt` 与
`logs/train_adablock_policy_qk_3layer_cobs.log`，不是推测旧模型的任务。

| 项目 | 既有 COBS 预算 policy | 当前 ranker |
| --- | --- | --- |
| 参数量 | 255,912 | 61,313 |
| 实际训练记录/epoch | 214,444 | 2,691,192 scenes |
| 验证记录 | 24,272 | 44,364 |
| batch | 64 | 8 |
| batch数/epoch | 3,351 | 336,399（全无有效pair的batch不更新） |
| epoch数 | 10 | 10 |
| 已保存运行device | CPU | CUDA:0，CPU采样/调度1线程 |
| 单个训练样本 | 固定长度特征 -> 一组预算/reuse输出 | query位置 × layer × KV-head，所有候选块评分 |
| 输出/损失 | 7预算档CE + reuse BCE + 0.01成本项 | 按attention mass差加权的 sampled pairwise softplus |
| 训练循环用时/epoch | 历史进度条82–91秒 | 当前吞吐外推约9,816秒（2.73小时），不含验证 |

预算 checkpoint 实际配置为 hidden_size=3584、projection=64、MLP=128、
7个预算档[0,4,8,16,32,64,128]；输入还有query_drift、7维score_features及8维feedback。
预算模型不是“只回归一个标量”。历史loss权重为 budget=1、reuse=1、cost=0.01；
历史训练确实包含reuse损失，当前ranker不启用reuse。

样本数之比为12.55倍，而batch从64变8又增加8倍批次数，总批次比约100.39倍。
当前scene按3层×4个KV-head展开，这与旧的一条controller记录不是同一计数单位；
此外旧任务记录含其内部train/validation划分，两者不是严格相同采样集。
旧训练并非每条输入都运行一次对所有块的可学习排序器。
原预算特征/标签已离线生成，训练阶段只读固定特征并预测少量类别。
现在每个scene要给所有候选块打分，并在7个共享KV的query-head上先做非线性匹配再聚合；
长上下文场景候选可达约2000块。模型参数更少，不表示每条样本的计算更少。

截至本次检查，当前平均约274scene/s、约29.2ms/有效optimizer step；
旧日志约3351步/82–91秒，即约24.5–27.2ms/step。
这组观测与“step数量增加约100倍”相吻合，而不是loss公式令每step慢100倍。
旧记录device=CPU，线程/历史环境不完全已知，所以该比值是运行记录对照，不是严格同硬件消融。

Rank loss并未枚举B×B所有块对。每scene最多16个正例，每正例最多4困难+4随机负例，
另最多32个正例内部pair，所以最终最多160个pair（会去重/过滤）。
本次进度累计平均约129.4pair/有效scene。额外成本包括候选评分、正例质量排序、
困难负例选择、CPU随机采样和数据搬运；pairwise softplus本身只是其中一部分。
当前固定部分已经达rho的scene仍沿用原训练语义处理，未偷偷改变样本规模/更新顺序。

此前分段profile证实存在CPU gather、采样和小kernel/同步开销；
最终路径已经优化这些环节。当前只读快照训练CPU约99%（约一个逻辑核），GPU利用率约7–14%、
显存约718MiB，提示仍有明显CPU/调度限制，不能认为GPU算力已饱和。
这里没有为最终路径新增占用训练GPU的benchmark，也没有据瞬时利用率编造精确耗时百分比。
分片缓存使每epoch主要按分片加载，不能未经IO测量就归咎磁盘。

证据：预算checkpoint的training_args/config、旧日志training_setup与各epoch最后进度条；
当前checkpoint目录progress.json；batch_optimization下benchmark_v1.json、benchmark_v2.json、
entry_cost_review.json。训练继续，配置未改动。

## 用户要求暂停并扩大 batch 的实验（进行中）

用户要求先暂停 batch8 正式训练，并尽量利用4张3090做更大batch/多卡测试。
仅向本任务训练 PID2190 和流程 PID2166 发送 SIGSTOP，两者均确认 T(stopped)；
没有终止或重启，没有推进最终测试。文档同步进程继续运行。

- 完整实时模型、optimizer、RNG和迭代器状态仍保留于暂停进程内，可原位 SIGCONT 恢复。
- 暂停前最后日志：epoch1，2,053,936 scenes，256,419 steps，loss0.14347；这不是精确磁盘恢复点。
- 备份磁盘状态：results/ranker_audit_20261003_v1/batch_optimization/larger_batch_trials/paused_disk_state.pt，
  精确记录 epoch1、2,000,000 scenes，含模型、optimizer、torch/CUDA/采样RNG。
  没有scheduler；当前没有自动磁盘resume入口。若远端进程丢失，不能声称能无损恢复到最后日志位置。
- 原位恢复命令（尚未执行）：kill -CONT 2166 2190。若原位恢复，日志elapsed wall time会包含暂停时长，需单独扣除后估算吞吐。
- pause_receipt.json 记录 PID核验、SIGSTOP状态、时刻和快照键。远端时钟偏差仍须注意。

GPU0保留暂停上下文约718MiB；其他卡无其他计算作业。
在GPU1/2/3各使用1个CPU线程并行短测batch8/32/64，统一冻结4096训练scenes（四长度各1024）、
512验证scenes和同一磁盘模型/optimizer起点，lr仍2e-4。
另测32K末端长候选的峰值显存；并行结果可能受主机资源争用影响，需隔离复测。
所有结果是短期batch实验，不是新的正式模型。增大batch减少更新次数、改变优化轨迹，不宣称与batch8等价。
随后评估4卡DDP：须说明per-rank/global batch、有效scene权重、数据不重复覆盖、RNG和恢复语义；
小模型/CPU采样限制下不预设线性加速。推荐明确之前不启动新长训。

实验入口、固定样本和进程：larger_batch_trials/trial_batch.py、trial_scenes.json、parallel_jobs.json。
各trial结果为对应子目录result.json，日志为parallel_b8.log、parallel_b32.log、parallel_b64.log。

## 大batch与四卡实测结论（已完成短测，未启动新长训）

推荐候选：4卡DDP，每卡16/global64，lr保持2e-4。旧PID2190/2166继续SIGSTOP。

以下统一4096训练scene、同一2M磁盘模型与optimizer起点、固定512验证scene；验证列为六个K的平均coverage。

| 配置 | scene/s | 更新数 | 训练loss | 验证平均coverage |
| --- | ---: | ---: | ---: | ---: |
| isolated_b8 | 181.2 | 512 | 0.151248 | 0.895003 |
| isolated_b32 | 363.6 | 128 | 0.157894 | 0.894965 |
| isolated_b64 | 353.4 | 64 | 0.163306 | 0.894883 |
| ddp_local8 | 597.0 | 128 | 0.157526 | 0.894865 |
| ddp_local16 | 1043.6 | 64 | 0.161536 | 0.894972 |
| ddp_local32 | 1045.3 | 32 | 0.166858 | 0.894581 |

所有试验前相同验证平均coverage为0.8933369。单卡batch8隔离短测有181scene/s的低值；并行短测259、原长训约273，不能拿低值夸大DDP加速。

每卡16/global64额外使用原seed42 shard顺序在2M快照之后的65536条唯一scene复测，cache2、不预加载，覆盖四长度。耗时72.13秒，908.61scene/s，1024次更新；各rank最终模型hash一致。训练loss0.14724，固定验证平均coverage约0.893446，required_k=93.354。该验证表现没有清晰证明收敛改善，不能把吞吐收益说成质量收益。

该实测外推全量10epoch纯训练约8.23小时，加入验证约8.68小时，建议按9–11小时规划并以实际日志更新。全局128初测与64同速，因此没有理由仅为更大batch再减半更新次数。全局64每完整epoch约42050个batch，相比batch8的336399减少约8倍；lr未线性放大。

### 显存（单卡真实各长度末端候选，含反向）

| 来源长度 | batch8 MiB | batch32 MiB | batch64 MiB |
| --- | ---: | ---: | ---: |
| 4096 | 71.3 | 224.7 | 387.9 |
| 8192 | 114.8 | 396.4 | 736.6 |
| 16384 | 203.0 | 747.5 | 1437.6 |
| 32768 | 378.4 | 1450.7 | 2841.1 |

4卡每卡16的32K长候选压力测试各卡约793–822MiB张量峰值；GPU0另保留旧暂停进程约718MiB。这里是torch allocated peak，不含所有driver/NCCL缓存，不等同nvidia-smi总量。

### DDP正确性与恢复边界

- 每个global batch按rank切成不相交片段，不使用补齐重复样本；4096和65536试验均验证唯一覆盖。尾部空rank仍参加collective。
- loss为本rank有效scene归一化loss之和乘 world_size/global有效scene数，再由DDP平均梯度；不是各rank局部均值简单平均。
- 有效数[0,6,7,6]、尾部[3,0,0,0]和全0三种情况与单卡固定pair参考对照通过，最大梯度差4.46e-7。
- 每rank CPU pair RNG seed=43+rank，明确不同于原单流轨迹；分布/损失定义保留，不声称与batch8更新轨迹等价。
- 短测均从同一磁盘模型和optimizer恢复，不改lr。正式切换前仍需实现/审查DDP训练恢复入口，保存每rank RNG、global已消费位置、epoch与sampler状态；不能只保存rank0 RNG。
- 若从磁盘2M状态切换，会重做旧内存暂停点之前约5万scene；这是明确回滚到checkpoint，不得称为精确从2,053,936继续。也可选择从头做纯DDP新实验。当前未做任何长期切换。

新增支持：utils/ranking_ddp.py；短测入口：scripts/benchmark_ranker_ddp.py。结果根目录：results/ranker_audit_20261003_v1/batch_optimization/larger_batch_trials；推荐汇总recommendation.json。其他证据包括各子目录result.json、length_peak_memory.json、ddp_extended.log、pause_receipt.json。


## 用户最终选择：从头四卡正式训练

用户在 2026-10-03 09:44:47 UTC 明确选择从头跑。正式运行采用随机初始化 seed=42，未加载旧 200 万场景权重或优化器；4 卡每卡 batch=16，全局 batch=64，lr=2e-4，10 epochs。旧 PID 2166/2190 保持暂停，旧数据与 checkpoint 保留。

新输出：checkpoints/ranker_ddp_fresh_global64_20261003；流水线及日志：results/ranker_audit_20261003_v1/batch_optimization/ddp_fresh_run/。训练结束后仅按验证集选择 checkpoint，然后自动构造预先冻结的16文档测试集并评估。预计训练加验证9–11小时，最终测试另计。

入口恢复验证：batch_optimization/ddp_resume_verification/result.json；连续与恢复两轮各60场景，120条索引顺序完全一致、无重复遗漏，四卡 RNG 与 sampler RNG 一致、优化器步数一致，非输出bias参数最大差7.45e-9，优化器moment差6.98e-10，验证mean coverage差0。输出bias为排序不变的常数平移，差4.94e-5；不宣称逐位等价。

启动前 GPU 检查最初遇到容器/宿主 PID 不同：暂停进程2190的Ngid为3358921。已读取/proc状态核实，同一旧任务占用718MiB且GPU利用率0，其他三卡空闲；修正检查后启动，未终止其他作业。


### 10:27 UTC 巡检及文档同步修复

四卡正式训练持续正常；第1轮已处理1,504,000/2,691,192场景，23,500步，累计loss0.18716，约890场景/秒。resume_latest.pt已生成且可读取。旧2166/2190仍为暂停状态。发现文档观察器在新manager写入运行状态前读到上次启动检查的blocked状态并退出；现重启观察器，训练未中断，未重复启动。


## 北京时间午夜断网安排与离线审计

用户确认2026-10-04 00:00北京时间（2026-10-03 16:00 UTC）仅断网、不关机。不得以远端快约10分26秒的时钟推算截止。管理进程8339已脱离SSH（PPID=1、独立session、无TTY），输入/dev/null、输出本地日志；四卡单机训练不依赖Windows在线。训练/验证数据和冻结16篇测试输入均本地；测试构造命令含--local-files-only，已核验Qwen模型4个权重分片、config及tokenizer缓存存在，无外部API或下载需求。磁盘剩余约1.8TiB。

训练后既有本地manager自动选验证best、构造冻结测试、评估并保存heldout_test_comparison.json、validation_comparison.json、final_comparison_summary.json及PIPELINE_COMPLETE。文档观察器同样脱离SSH。仅外网中断且本机通信/存储持续可用时，无需恢复操作；断网期间外部无法实时核验，待连通后读取结果。

每约10万场景原子保存resume_latest.pt，含模型、优化器、各rank RNG及sampler位置，已验证恢复入口。当前没有自动崩溃重试：若发生非网络相关运行错误，manager会记录blocked及堆栈，后续需从checkpoint恢复；不宣称断网期间自动修复任意异常。

截至约13:08 UTC，epoch4约154万/269万，约903场景/秒；前三轮验证best为epoch3。按每轮约53分钟估算，午夜前约完成6轮并进入第7轮，完整训练验证及最后测试预计北京时间次日约03:00，粗略区间02:45–03:30，取决于后续速度。完整审计：results/ranker_audit_20261003_v1/batch_optimization/ddp_fresh_run/offline_readiness_audit.json。

## 重启后最终完成核验（2026-10-04）

已重新连接服务器并核验持久产物：10轮训练全部完成，每轮2,691,192场景，训练退出码0；冻结16篇测试构造退出码0，48 shards、44,448场景；最终评测及PIPELINE_COMPLETE均存在。没有相关训练进程，无需恢复或重跑。容器内/proc/uptime反映宿主长时间运行，不据此否认用户报告的服务器重启。

训练加逐轮验证耗时31867.59秒（8.85小时），测试构造226.71秒；测试评估165.57秒，均已在断网期间落盘。远端记录完成时间2026-10-04 03:10:45+08:00，远端钟此前测得快约10分26秒，不能当作校准后的精确北京时间。

选模仅使用验证集六档coverage均值，最佳epoch7（0.895518581）；最后epoch10为0.894661507，故最终测试使用best.pt的epoch7，未使用test调参。best/last/resume均可读取，resume为epoch11 offset0。重验train/test与validation/test文档隔离通过。

### 冻结测试整体结果

| 方法 | Coverage@8 | Coverage@32 | Coverage@128 | 达到95%平均块数 |
| --- | ---: | ---: | ---: | ---: |
| Learned | 0.852257 | 0.910366 | 0.967395 | 81.757 |
| COBS（工程legacy mean logits） | 0.838231 | 0.900297 | 0.963372 | 91.451 |
| Quest AABB | 0.828556 | 0.880438 | 0.949334 | 124.064 |
| Oracle | 0.860287 | 0.916784 | 0.970618 | 74.506 |

非平凡子集fixed_below_rho共37,428场景（固定sink/local/self尚未达到95%）；其learned/COBS/Quest/Oracle所需平均块数分别为97.092/108.604/147.334/88.480；coverage@32分别为0.895281/0.883615/0.860302/0.902667。所有方法unreachable_rate为0。

### 按query历史长度分组（不是整篇文档长度）

| 历史长度 | 场景数 | Learned C@32 | COBS C@32 | Quest C@32 | Learned k95 | COBS k95 | Quest k95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-4K | 10656 | 0.969961 | 0.966637 | 0.955087 | 16.596 | 18.496 | 23.318 |
| 4-8K | 9216 | 0.911273 | 0.902622 | 0.876025 | 57.228 | 64.577 | 81.375 |
| 8-16K | 12288 | 0.905690 | 0.895908 | 0.872196 | 73.452 | 84.807 | 113.827 |
| 16-32K | 12288 | 0.862680 | 0.845412 | 0.827256 | 164.967 | 181.518 | 253.684 |

整体达到95%所需平均块数相比COBS减少10.60%，相比Quest减少34.10%。以上是同样query/layer/head/candidate/budget条件下排序质量结果，不是推理加速测量。评估时间包括多方法和聚合，不能当作单模型延迟。仅16篇独立测试，未做文档级置信区间，不宣称统计显著性；无额外NDCG或速度实验。

详细结果：results/ranker_audit_20261003_v1/batch_optimization/ddp_fresh_run/heldout_test_comparison.json、validation_comparison.json、final_comparison_summary.json；重启后核验：post_restart_completion_verification.json。模型：checkpoints/ranker_ddp_fresh_global64_20261003/best.pt。


## LongBench HotpotQA 8样本真实生成试跑（2026-10-04）

独立脚本experiments/presets/run_ranker_hotpot8_20261004.py，输出results/longbench_ranker_hotpot8_20261004。复用本地HotpotQA前8条、Qwen2.5-7B-Instruct BF16、官方HotpotQA prompt加Qwen聊天模板，统一中间截断8192、贪心最多32 tokens。稀疏方法固定sink32 tokens、local32 blocks、block16、middle K32，当前token self另加；prefill全部dense。比较full、COBS、Quest、Learned（仅12/20/27层用epoch7，其余层COBS），无reuse。官方QA token-F1（多参考取max、样本均值乘100）；记录预测、耗时、PyTorch峰值显存、真实选块和learned调用证据。已有用户改动不覆盖。

首次探针全候选token一致性通过；1024上下文小于66块预算无法形成稀疏，被assert拦下，未开始正式评测。保留probe_v1_failure.json和脚本快照，将learned探针改为2048上下文；正式8192配置不变。

### HotpotQA 八样本最终结果

四种方案均完成8/8；所有同样本prompt长度、截断长度与首token均一致。以下为官方LongBench QA token-F1（各参考答案取最大、再做样本均值乘100），不是离线attention coverage。

| 方法 | F1 | 8条总秒数 | PyTorch峰值allocated MiB | 峰值reserved MiB | 生成tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full attention | 35.42 | 15.19 | 16101.4 | 16316.0 | 67 |
| COBS | 35.42 | 290.40 | 16101.4 | 16474.0 | 67 |
| Quest AABB | 35.42 | 291.93 | 16101.4 | 16474.0 | 81 |
| Learned三层+其余COBS | 35.42 | 292.98 | 16101.4 | 16474.0 | 67 |

路径验证：1024-token全候选稀疏与full前4个生成token一致；2048-token Learned探针真实裁剪到最多1056历史tokens，调用只在12/20/27层。正式记录每层首个head的选块哈希，Quest与COBS不同；Learned训练层也改变选块，后续COBS层可受隐藏状态传播影响。每个稀疏生成step覆盖28层，无reuse。全部历史KV存在CPU，dense prefill与第一生成token保持full；不宣称端到端稀疏prefill或永久KV驱逐。

生成32-token上限记录：{"full": [{"index": 2, "last_token_id": 13, "last_token_text": "."}], "cobs": [{"index": 2, "last_token_id": 13, "last_token_text": "."}], "quest": [{"index": 2, "last_token_id": 279, "last_token_text": " the"}], "learned": [{"index": 2, "last_token_id": 13, "last_token_text": "."}]}。保留原输出，不基于结果延长生成重测。前7条输入中间截断到8192，最后1条2317；input_manifest.json保存输入token哈希。仅8条有序样本、单预算，不能据同分推断方法总体等价，也未证明此前coverage提升转化成任务得分提升。

计时为单次顺序运行，含tokenization、dense prefill、摘要构造、解码及trace记录，排除模型加载和路径探针；生成长度不同，不能直接当纯decode吞吐基准。稀疏原型CPU KV和逐层搬运/摘要构造开销明显，未展示加速。显存为PyTorch allocator峰值（含模型与dense prefill），不含CPU KV/驱动显存，不是稀疏cache独立占用。

产物：results/longbench_ranker_hotpot8_20261004/{summary.json,completion_audit.json,config.json,input_manifest.json,path_verification.json,full.jsonl,cobs.jsonl,quest.jsonl,learned.jsonl}；代码experiments/presets/run_ranker_hotpot8_20261004.py；官方评分来源https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py。尚未进行全量LongBench、更长上下文、多预算或重复速度测量。


## HotpotQA 同8条32K/128重跑

用户授权将上下文上限改为32768、输出128；Qwen实际max_position_embeddings=32768，无rope扩展，因此prompt上限32640，为128生成token预留空间。8条原始prompt长度为11787、17322、17388、16764、15170、15750、16128、2317，全部低于上限，预计无输入截断。复用旧selected_rows.json，checkpoint、预算、评分、生成温度和其余路径保持相同。检查旧config所记录4个评测源码hash均无变动；旧结果不覆盖。

新脚本experiments/presets/run_ranker_hotpot8_32k128_20261004.py；新输出results/longbench_ranker_hotpot8_32k128_20261004。最长17388-token样本先作单卡BF16/SDPA显存探针；记录逐条输入hash和输入/输出截断标志。预计25–40分钟，依实际耗时修正。

### 32K/128重跑完成结果

四种方案均完成8/8，实际prompt上限32640（模型位置32768减预留128），所有输入完整保留，四方案同样本input token哈希一致。训练三层Learned与其余COBS混合方式、epoch7 checkpoint、middle32预算、其余配置不变。

| 方法 | 原8K/32 F1 | 新32K/128 F1 | 新总秒数 | 峰值allocated MiB | 峰值reserved MiB | 输出截断样本（0起始） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full attention | 35.42 | 47.92 | 31.02 | 17858.4 | 18564.0 | [] |
| COBS | 35.42 | 47.92 | 462.48 | 17858.4 | 18602.0 | [] |
| Quest AABB | 35.42 | 47.92 | 461.40 | 17858.4 | 18602.0 | [] |
| Learned三层+其余COBS | 35.42 | 37.34 | 474.70 | 17858.4 | 18602.0 | [] |

逐条原始/保留prompt token数：

| 样本（0起始） | 原始 | 保留 | 输入截断 |
| --- | ---: | ---: | --- |
| 0 | 11787 | 11787 | False |
| 1 | 17322 | 17322 | False |
| 2 | 17388 | 17388 | False |
| 3 | 16764 | 16764 | False |
| 4 | 15170 | 15170 | False |
| 5 | 15750 | 15750 | False |
| 6 | 16128 | 16128 | False |
| 7 | 2317 | 2317 | False |

评分仍为官方QA token-F1多参考取max再做均值乘100。此次同时扩大输入与输出上限，分数变化不能单独归因于其中一项。仅8条有序样本、单预算，非全量LongBench，不作统计显著性结论。计时含tokenization、dense prefill、摘要构造、decode和trace，不含模型加载及探针；单次顺序运行且各方法生成长度可不同，不能作为纯decode吞吐基准。显存为PyTorch峰值，包括模型和prefill、不含CPU完整KV与驱动内存；稀疏仅作用decode（第一token为dense prefill），不宣称稀疏prefill或永久KV驱逐。

本轮产物results/longbench_ranker_hotpot8_32k128_20261004：config.json、input_manifest.json、memory_preflight.json、path_verification.json、四方法jsonl、summary.json、completion_audit.json、comparison_with_8k.json。代码experiments/presets/run_ranker_hotpot8_32k128_20261004.py，旧8K结果完整保留。


32K/128逐样本退分核验：[{"index": 6, "answers": ["Tharangambadi"], "methods": {"full": {"prediction": "Tharangambadi", "score": 100.0, "tokens": 6, "ended_with_eos": true}, "cobs": {"prediction": "Tharangambadi", "score": 100.0, "tokens": 6, "ended_with_eos": true}, "quest": {"prediction": "Tharangambadi", "score": 100.0, "tokens": 6, "ended_with_eos": true}, "learned": {"prediction": "Thirukkalacherry is a village in Mayiladuthurai District, Tranquebar (Tharangambadi) Taluk, Tamil Nadu, India.", "score": 15.384615384615385, "tokens": 36, "ended_with_eos": true}}}]

所有32条输出均以EOS结束，输入/输出截断均为false。混合Learned相对同上下文COBS下降10.576923个F1分；差异是实际生成回答质量，不能用离线coverage优势代替端到端结论。仅8条样本，单条退分对均值影响较大；未追加调参或重跑。详见score_difference_cases.json。


## 完整LongBench HotpotQA 200条评测

用户授权完整子任务，官方任务卡200条。本地hotpotqa.jsonl与缓存官方snapshot 5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip内文件字节SHA256相同，200个唯一ID，完整有序匹配，无下载。全量prompt长度1927–17684、均值13475.455；prompt上限32640、预留输出128，全部无输入截断。

四张3090空闲后按full/COBS/Quest/三层Learned混合各占GPU0/1/2/3并行。沿用epoch7、greedy、block16、sink32tokens、local32blocks、middle32；不复用先前8条结果或耗时。预估3–5小时，四方法共享CPU/内存/IO，速度仅作竞争条件下参考。新脚本experiments/presets/run_ranker_hotpotqa_full_20261004.py；新目录results/longbench_ranker_hotpotqa_full_20261004。每条JSONL flush+fsync，恢复核验完整前缀和输入hash，未完成尾行备份后修复；配置不一致拒绝恢复。manager脱离SSH，失败最多同配置重试一次（OOM停止而不盲目重试），旧产物保留。

<!-- HOTPOT_FULL_STATUS_START -->
用户明确要求停止。本任务评测、管理、巡检及自动重试进程已停止；已有预测保留，未自动续跑。停止核验见results/longbench_ranker_hotpotqa_full_20261004/user_stop_receipt.json。
<!-- HOTPOT_FULL_STATUS_END -->


## 用户授权：原epoch7 ranker全28层实验

旧full/COBS/Quest/混合任务保持停止。本次独立全层实验使用同一epoch7共享ranker，真实layer_id=0..27，不映射层号、不改checkpoint或训练metadata、不重训。新增models/experimental_all_layer_ranker_qwen.py显式all_layer_ranker开关；原controller文件未修改。基类构造器复用原metadata校验和摘要构造，构造后启用全28层ranker；实际baseline评分函数在实验进程内设为调用即报错，确保无COBS fallback。

探针已通过：全候选稀疏与full四token一致；2048输入上每个decode step全部28层真实调用，最多1056历史tokens；17684-token最大样本显存allocated约17.50GiB。训练metadata仍为12/20/27，25层embedding未经本轮监督，此为原checkpoint跨层应用实验。

完整200条按index%4分成4份各50条，每卡加载同checkpoint；prompt32640、生成128、greedy、sink32/local32blocks/middle32blocks/block16均不变。对照复用旧full200，核验配置与input hash；不重跑full、不恢复旧任务。新目录results/longbench_ranker_all28_hotpotqa_20261004，新入口experiments/presets/run_ranker_all28_hotpotqa_20261004.py。预计约60–90分钟，以分片实测调整；并行时间有资源竞争限制。逐条fsync并支持按分片前缀恢复，manager后台运行，15分钟资源巡检。

<!-- HOTPOT_ALL28_STATUS_START -->
阶段：completed；已落盘：200/200。

| 分片/GPU | 样本数 | 状态 |
| --- | ---: | --- |
| 0 | 50/50 | completed |
| 1 | 50/50 | completed |
| 2 | 50/50 | completed |
| 3 | 50/50 | completed |
<!-- HOTPOT_ALL28_STATUS_END -->


### 全28层实验最终结果

{
  "samples": 200,
  "learned_all28_f1": 57.67030547993963,
  "full_reference_f1": 58.146025465235944,
  "input_truncated_indices": [],
  "output_truncated_indices": [],
  "sum_sample_seconds": 12105.596816329286,
  "peak_allocated_mib": 17915.6318359375,
  "peak_reserved_mib": 18666.0,
  "generated_tokens": 1554
}

完整200条无重复遗漏；各稀疏decode step均核验28层ranker调用，无baseline fallback。checkpoint训练metadata仍为12/20/27，其他25层embedding没有本轮训练监督，实验衡量原checkpoint直接跨层使用；不重训、不映射层号。full200对照复用已完成预测，配置和逐条input hash严格一致，不复用其时间做并行速度比。4卡ID分片共享CPU/内存/IO，样本耗时之和不是墙钟时间；峰值显存含模型和dense prefill，不含CPU完整KV与驱动内存。逐条差异、截断索引和完整预测见results/longbench_ranker_all28_hotpotqa_20261004。


### HotpotQA 128→32 token离线精确截断复评分（2026-10-04）

仅使用原full/all28各200条真实生成token trace，取前32个新增token（EOS占预算），原tokenizer decode(skip_special_tokens=True).strip()。400条完整trace回解码与原预测、原F1全部一致；同200输入哈希及答案对齐。未运行模型，未修改原结果。

{
  "samples": 200,
  "methods": {
    "full": {
      "128": {
        "f1": 58.146025465235944,
        "em_percent": 43.0,
        "em_count": 86
      },
      "32": {
        "f1": 57.9806499087555,
        "em_percent": 43.0,
        "em_count": 86
      },
      "delta_32_minus_128": {
        "f1": -0.16537555648044133,
        "em_percent": 0.0
      },
      "changed_prediction_indices": [
        48,
        59,
        63,
        72,
        95,
        117,
        188,
        189
      ],
      "changed_score_indices": [
        48,
        59,
        72,
        95,
        117,
        188
      ]
    },
    "all28": {
      "128": {
        "f1": 57.67030547993963,
        "em_percent": 45.0,
        "em_count": 90
      },
      "32": {
        "f1": 57.71291533445489,
        "em_percent": 45.0,
        "em_count": 90
      },
      "delta_32_minus_128": {
        "f1": 0.04260985451526267,
        "em_percent": 0.0
      },
      "changed_prediction_indices": [
        48,
        59,
        63,
        95,
        176,
        188,
        189
      ],
      "changed_score_indices": [
        48,
        59,
        95,
        176,
        188
      ]
    }
  },
  "ranker_vs_full": {
    "128": {
      "f1_delta": -0.47571998529630277,
      "wins": 7,
      "ties": 181,
      "losses": 12
    },
    "32": {
      "f1_delta": -0.26773457430059877,
      "wins": 7,
      "ties": 185,
      "losses": 8
    }
  }
}

产物：/root/adablock/results/longbench_ranker_hotpotqa_cap32_postprocess_20261004。脚本rescore.py、audit.json、changed_cases.json及两方法cap32 JSONL。
等价范围：既有手写greedy无长度相关logit处理或提前计划，128和32仅改变循环上限；本批首token无EOS，因此截前32精确对应同实现同数值执行的32上限输出。此为离线反事实，不是新推理实测，不复现HF generate重复惩罚或LouisKV未知协议。


### All28 HotpotQA前8条实际decode attention质量诊断

原epoch7、32K/128、sink32+local512+middle512，无COBS回退。独立重放前8条全部稀疏decode步，完整生成token序列及首步28层head0选块哈希与原结果一致。原日志未保存其他步/其他head选择，无法逐项追溯比较；新诊断保留所有所选middle块。

分母为稀疏模型自身轨迹真实query对该轨迹完整历史K与self的FP32 softmax总质量；不是full模型独立轨迹。fixed含sink/local/self，middle增量为实际选中middle的质量。GQA按每7个queryhead质量均值，相同middle块预算oracle按组均值选top块。总体按head-step等权，另存分层/样本/生成步结果；仅8条不代表200条。

{
  "samples": 8,
  "layer_steps": 840,
  "query_head": {
    "coverage": {
      "n": 23520,
      "mean": 0.8334401897020811,
      "median": 0.9094799160957336,
      "p10": 0.5470188200473786,
      "below90_fraction": 0.4769557823129252,
      "below95_fraction": 0.6321003401360544
    },
    "fixed_self": {
      "n": 23520,
      "mean": 0.7335623116090002,
      "median": 0.8329475522041321,
      "p10": 0.3182364344596863,
      "below90_fraction": 0.6329931972789116,
      "below95_fraction": 0.784311224489796
    },
    "middle_increment_mean": 0.09987787809740668,
    "self_mass_mean": 0.06674411692888949,
    "oracle_gqa_same_budget": {
      "n": 23520,
      "mean": 0.8667894818017031,
      "median": 0.9338912069797516,
      "p10": 0.6244272232055664,
      "below90_fraction": 0.40127551020408164,
      "below95_fraction": 0.563860544217687
    },
    "oracle_minus_selected_mean": 0.03334929209962197,
    "fixed_already95_fraction": 0.21568877551020407,
    "nontrivial_fixed_below95": {
      "coverage": {
        "n": 18447,
        "mean": 0.7910310515750385,
        "median": 0.8583340644836426,
        "p10": 0.5016562104225158,
        "below90_fraction": 0.6081205616089337,
        "below95_fraction": 0.8059305036049222
      },
      "oracle_coverage": {
        "n": 18447,
        "mean": 0.8329808848328986,
        "median": 0.8966439366340637,
        "p10": 0.5838817477226257,
        "below90_fraction": 0.5116279069767442,
        "below95_fraction": 0.7189244863663469
      },
      "oracle_gap_mean": 0.04194983325786012
    }
  },
  "gqa_mean": {
    "coverage": {
      "n": 3360,
      "mean": 0.8334401897020811,
      "median": 0.8626437059470585,
      "p10": 0.6476683514458792,
      "below90_fraction": 0.6163690476190476,
      "below95_fraction": 0.7863095238095238
    },
    "fixed_self": {
      "n": 3360,
      "mean": 0.7335623116090002,
      "median": 0.7725844181009701,
      "p10": 0.4714883600428168,
      "below90_fraction": 0.8101190476190476,
      "below95_fraction": 0.9354166666666667
    },
    "middle_increment_mean": 0.09987787809740668,
    "self_mass_mean": 0.06674411692888949,
    "oracle_gqa_same_budget": {
      "n": 3360,
      "mean": 0.8667894818017031,
      "median": 0.8995668377195085,
      "p10": 0.6953126613582884,
      "below90_fraction": 0.5008928571428571,
      "below95_fraction": 0.7104166666666667
    },
    "oracle_minus_selected_mean": 0.03334929209962197,
    "fixed_already95_fraction": 0.06458333333333334,
    "nontrivial_fixed_below95": {
      "coverage": {
        "n": 3143,
        "mean": 0.8228968271671364,
        "median": 0.8514933586120605,
        "p10": 0.6402757027319499,
        "below90_fraction": 0.6589245943366211,
        "below95_fraction": 0.8405981546293351
      },
      "oracle_coverage": {
        "n": 3143,
        "mean": 0.8583184465253579,
        "median": 0.890699028968811,
        "p10": 0.6892032401902335,
        "below90_fraction": 0.5354756601972638,
        "below95_fraction": 0.7594654788418709
      },
      "oracle_gap_mean": 0.035421619358221544
    }
  },
  "wall_seconds": 499.2735047917813
}

独立产物：/root/adablock/results/longbench_all28_attention_coverage8_20261004，含diagnose.py、coverage_steps.jsonl、summary.json、audit.json、sample_audits.json。


### All28轨迹同Q/K选择器覆盖对比：ranker / COBS / Quest

同前8条全部真实decode步；各方法共享ranker自身轨迹Q/K、sink/local/self、middle32块和block16。COBS、Quest直接旁路调用原工程score_qk_block_digests公式，GQA按7个queryhead score均值选择，未反馈到生成。完整token序列与原结果一致，840层步的全部4个KVhead选块哈希与上一诊断一致。非各基线独立端到端轨迹。

{
  "samples": 8,
  "sparse_steps": 30,
  "query_head_observations": 23520,
  "overall": {
    "ranker": {
      "query_head": {
        "coverage": {
          "n": 23520,
          "mean": 0.8334401897020811,
          "median": 0.9094799160957336,
          "p10": 0.5470188200473786,
          "below90_fraction": 0.4769557823129252,
          "below95_fraction": 0.6321003401360544
        },
        "fixed_self": {
          "n": 23520,
          "mean": 0.7335623116090002,
          "median": 0.8329475522041321,
          "p10": 0.3182364344596863,
          "below90_fraction": 0.6329931972789116,
          "below95_fraction": 0.784311224489796
        },
        "middle_increment_mean": 0.09987787809740668,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 23520,
          "mean": 0.8667894818017031,
          "median": 0.9338912069797516,
          "p10": 0.6244272232055664,
          "below90_fraction": 0.40127551020408164,
          "below95_fraction": 0.563860544217687
        },
        "oracle_minus_selected_mean": 0.03334929209962197,
        "fixed_already95_fraction": 0.21568877551020407,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 18447,
            "mean": 0.7910310515750385,
            "median": 0.8583340644836426,
            "p10": 0.5016562104225158,
            "below90_fraction": 0.6081205616089337,
            "below95_fraction": 0.8059305036049222
          },
          "oracle_coverage": {
            "n": 18447,
            "mean": 0.8329808848328986,
            "median": 0.8966439366340637,
            "p10": 0.5838817477226257,
            "below90_fraction": 0.5116279069767442,
            "below95_fraction": 0.7189244863663469
          },
          "oracle_gap_mean": 0.04194983325786012
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3360,
          "mean": 0.8334401897020811,
          "median": 0.8626437059470585,
          "p10": 0.6476683514458792,
          "below90_fraction": 0.6163690476190476,
          "below95_fraction": 0.7863095238095238
        },
        "fixed_self": {
          "n": 3360,
          "mean": 0.7335623116090002,
          "median": 0.7725844181009701,
          "p10": 0.4714883600428168,
          "below90_fraction": 0.8101190476190476,
          "below95_fraction": 0.9354166666666667
        },
        "middle_increment_mean": 0.09987787809740668,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 3360,
          "mean": 0.8667894818017031,
          "median": 0.8995668377195085,
          "p10": 0.6953126613582884,
          "below90_fraction": 0.5008928571428571,
          "below95_fraction": 0.7104166666666667
        },
        "oracle_minus_selected_mean": 0.03334929209962197,
        "fixed_already95_fraction": 0.06458333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 3143,
            "mean": 0.8228968271671364,
            "median": 0.8514933586120605,
            "p10": 0.6402757027319499,
            "below90_fraction": 0.6589245943366211,
            "below95_fraction": 0.8405981546293351
          },
          "oracle_coverage": {
            "n": 3143,
            "mean": 0.8583184465253579,
            "median": 0.890699028968811,
            "p10": 0.6892032401902335,
            "below90_fraction": 0.5354756601972638,
            "below95_fraction": 0.7594654788418709
          },
          "oracle_gap_mean": 0.035421619358221544
        }
      }
    },
    "cobs": {
      "query_head": {
        "coverage": {
          "n": 23520,
          "mean": 0.8491483470969865,
          "median": 0.9246717989444733,
          "p10": 0.5765960812568665,
          "below90_fraction": 0.4344812925170068,
          "below95_fraction": 0.589328231292517
        },
        "fixed_self": {
          "n": 23520,
          "mean": 0.7335623116090002,
          "median": 0.8329475522041321,
          "p10": 0.3182364344596863,
          "below90_fraction": 0.6329931972789116,
          "below95_fraction": 0.784311224489796
        },
        "middle_increment_mean": 0.11558603548237319,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 23520,
          "mean": 0.8667894818017031,
          "median": 0.9338912069797516,
          "p10": 0.6244272232055664,
          "below90_fraction": 0.40127551020408164,
          "below95_fraction": 0.563860544217687
        },
        "oracle_minus_selected_mean": 0.017641134704716604,
        "fixed_already95_fraction": 0.21568877551020407,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 18447,
            "mean": 0.8105099311771642,
            "median": 0.881030797958374,
            "p10": 0.5320919275283813,
            "below90_fraction": 0.5539654144305307,
            "below95_fraction": 0.7513958909307746
          },
          "oracle_coverage": {
            "n": 18447,
            "mean": 0.8329808848328986,
            "median": 0.8966439366340637,
            "p10": 0.5838817477226257,
            "below90_fraction": 0.5116279069767442,
            "below95_fraction": 0.7189244863663469
          },
          "oracle_gap_mean": 0.022470953655734375
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3360,
          "mean": 0.8491483470969865,
          "median": 0.8802665897778102,
          "p10": 0.6649414477603777,
          "below90_fraction": 0.5636904761904762,
          "below95_fraction": 0.75
        },
        "fixed_self": {
          "n": 3360,
          "mean": 0.7335623116090002,
          "median": 0.7725844181009701,
          "p10": 0.4714883600428168,
          "below90_fraction": 0.8101190476190476,
          "below95_fraction": 0.9354166666666667
        },
        "middle_increment_mean": 0.11558603548237319,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 3360,
          "mean": 0.8667894818017031,
          "median": 0.8995668377195085,
          "p10": 0.6953126613582884,
          "below90_fraction": 0.5008928571428571,
          "below95_fraction": 0.7104166666666667
        },
        "oracle_minus_selected_mean": 0.017641134704716604,
        "fixed_already95_fraction": 0.06458333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 3143,
            "mean": 0.8395869897135897,
            "median": 0.8698679889951434,
            "p10": 0.6594781994819641,
            "below90_fraction": 0.60260897231944,
            "below95_fraction": 0.8017817371937639
          },
          "oracle_coverage": {
            "n": 3143,
            "mean": 0.8583184465253579,
            "median": 0.890699028968811,
            "p10": 0.6892032401902335,
            "below90_fraction": 0.5354756601972638,
            "below95_fraction": 0.7594654788418709
          },
          "oracle_gap_mean": 0.01873145681176829
        }
      }
    },
    "quest_aabb": {
      "query_head": {
        "coverage": {
          "n": 23520,
          "mean": 0.8223748019792307,
          "median": 0.9101371169090271,
          "p10": 0.5008469164371491,
          "below90_fraction": 0.47551020408163264,
          "below95_fraction": 0.6307397959183674
        },
        "fixed_self": {
          "n": 23520,
          "mean": 0.7335623116090002,
          "median": 0.8329475522041321,
          "p10": 0.3182364344596863,
          "below90_fraction": 0.6329931972789116,
          "below95_fraction": 0.784311224489796
        },
        "middle_increment_mean": 0.08881249038457434,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 23520,
          "mean": 0.8667894818017031,
          "median": 0.9338912069797516,
          "p10": 0.6244272232055664,
          "below90_fraction": 0.40127551020408164,
          "below95_fraction": 0.563860544217687
        },
        "oracle_minus_selected_mean": 0.044414679822472374,
        "fixed_already95_fraction": 0.21568877551020407,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 18447,
            "mean": 0.7769052685329448,
            "median": 0.8556686639785767,
            "p10": 0.45675066709518436,
            "below90_fraction": 0.6062774434867458,
            "below95_fraction": 0.8041958041958042
          },
          "oracle_coverage": {
            "n": 18447,
            "mean": 0.8329808848328986,
            "median": 0.8966439366340637,
            "p10": 0.5838817477226257,
            "below90_fraction": 0.5116279069767442,
            "below95_fraction": 0.7189244863663469
          },
          "oracle_gap_mean": 0.05607561629995378
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3360,
          "mean": 0.8223748019792307,
          "median": 0.858000133718763,
          "p10": 0.612363921744483,
          "below90_fraction": 0.6208333333333333,
          "below95_fraction": 0.7985119047619048
        },
        "fixed_self": {
          "n": 3360,
          "mean": 0.7335623116090002,
          "median": 0.7725844181009701,
          "p10": 0.4714883600428168,
          "below90_fraction": 0.8101190476190476,
          "below95_fraction": 0.9354166666666667
        },
        "middle_increment_mean": 0.08881249038457434,
        "self_mass_mean": 0.06674411692888949,
        "oracle_gqa_same_budget": {
          "n": 3360,
          "mean": 0.8667894818017031,
          "median": 0.8995668377195085,
          "p10": 0.6953126613582884,
          "below90_fraction": 0.5008928571428571,
          "below95_fraction": 0.7104166666666667
        },
        "oracle_minus_selected_mean": 0.044414679822472374,
        "fixed_already95_fraction": 0.06458333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 3143,
            "mean": 0.811140785796997,
            "median": 0.844970098563603,
            "p10": 0.6065534532070159,
            "below90_fraction": 0.6636971046770601,
            "below95_fraction": 0.8536430162265352
          },
          "oracle_coverage": {
            "n": 3143,
            "mean": 0.8583184465253579,
            "median": 0.890699028968811,
            "p10": 0.6892032401902335,
            "below90_fraction": 0.5354756601972638,
            "below95_fraction": 0.7594654788418709
          },
          "oracle_gap_mean": 0.047177660728361014
        }
      }
    }
  },
  "training_layers": {
    "ranker": {
      "query_head": {
        "coverage": {
          "n": 2520,
          "mean": 0.853177156683708,
          "median": 0.9178261756896973,
          "p10": 0.6083167672157288,
          "below90_fraction": 0.4638888888888889,
          "below95_fraction": 0.5861111111111111
        },
        "fixed_self": {
          "n": 2520,
          "mean": 0.7213795503685625,
          "median": 0.8000373542308807,
          "p10": 0.3296941667795181,
          "below90_fraction": 0.6880952380952381,
          "below95_fraction": 0.8051587301587302
        },
        "middle_increment_mean": 0.13179760627625953,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 2520,
          "mean": 0.8622569240097488,
          "median": 0.9249649941921234,
          "p10": 0.631798779964447,
          "below90_fraction": 0.4523809523809524,
          "below95_fraction": 0.5714285714285714
        },
        "oracle_minus_selected_mean": 0.009079767326040874,
        "fixed_already95_fraction": 0.19484126984126984,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2029,
            "mean": 0.8192039010426633,
            "median": 0.8625588417053223,
            "p10": 0.5823068141937255,
            "below90_fraction": 0.5761458846722524,
            "below95_fraction": 0.7279448003942829
          },
          "oracle_coverage": {
            "n": 2029,
            "mean": 0.8304199273874749,
            "median": 0.8713791966438293,
            "p10": 0.604489266872406,
            "below90_fraction": 0.5618531296205027,
            "below95_fraction": 0.7097092163627403
          },
          "oracle_gap_mean": 0.01121602634481164
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 360,
          "mean": 0.853177156683708,
          "median": 0.8841408618858883,
          "p10": 0.6584261647292545,
          "below90_fraction": 0.5333333333333333,
          "below95_fraction": 0.675
        },
        "fixed_self": {
          "n": 360,
          "mean": 0.7213795503685625,
          "median": 0.7712797394820622,
          "p10": 0.43673136650717687,
          "below90_fraction": 0.7916666666666666,
          "below95_fraction": 0.8833333333333333
        },
        "middle_increment_mean": 0.13179760627625953,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 360,
          "mean": 0.8622569240097488,
          "median": 0.8916356308119637,
          "p10": 0.6793188946587698,
          "below90_fraction": 0.5194444444444445,
          "below95_fraction": 0.65
        },
        "oracle_minus_selected_mean": 0.009079767326040874,
        "fixed_already95_fraction": 0.11666666666666667,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 318,
            "mean": 0.8349892449601213,
            "median": 0.8577929053987776,
            "p10": 0.6506018710987909,
            "below90_fraction": 0.6037735849056604,
            "below95_fraction": 0.7641509433962265
          },
          "oracle_coverage": {
            "n": 318,
            "mean": 0.8451086131241657,
            "median": 0.8688401111534664,
            "p10": 0.671221570457731,
            "below90_fraction": 0.5880503144654088,
            "below95_fraction": 0.7358490566037735
          },
          "oracle_gap_mean": 0.010119368164044507
        }
      }
    },
    "cobs": {
      "query_head": {
        "coverage": {
          "n": 2520,
          "mean": 0.8483294241396444,
          "median": 0.914502739906311,
          "p10": 0.5972282767295838,
          "below90_fraction": 0.473015873015873,
          "below95_fraction": 0.5912698412698413
        },
        "fixed_self": {
          "n": 2520,
          "mean": 0.7213795503685625,
          "median": 0.8000373542308807,
          "p10": 0.3296941667795181,
          "below90_fraction": 0.6880952380952381,
          "below95_fraction": 0.8051587301587302
        },
        "middle_increment_mean": 0.12694987383419795,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 2520,
          "mean": 0.8622569240097488,
          "median": 0.9249649941921234,
          "p10": 0.631798779964447,
          "below90_fraction": 0.4523809523809524,
          "below95_fraction": 0.5714285714285714
        },
        "oracle_minus_selected_mean": 0.01392749987010445,
        "fixed_already95_fraction": 0.19484126984126984,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2029,
            "mean": 0.8134911076086327,
            "median": 0.8561824560165405,
            "p10": 0.5718890547752381,
            "below90_fraction": 0.5874815179891573,
            "below95_fraction": 0.7343518974864465
          },
          "oracle_coverage": {
            "n": 2029,
            "mean": 0.8304199273874749,
            "median": 0.8713791966438293,
            "p10": 0.604489266872406,
            "below90_fraction": 0.5618531296205027,
            "below95_fraction": 0.7097092163627403
          },
          "oracle_gap_mean": 0.016928819778842138
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 360,
          "mean": 0.8483294241396444,
          "median": 0.8779041618108749,
          "p10": 0.6576578761850085,
          "below90_fraction": 0.5444444444444444,
          "below95_fraction": 0.6888888888888889
        },
        "fixed_self": {
          "n": 360,
          "mean": 0.7213795503685625,
          "median": 0.7712797394820622,
          "p10": 0.43673136650717687,
          "below90_fraction": 0.7916666666666666,
          "below95_fraction": 0.8833333333333333
        },
        "middle_increment_mean": 0.12694987383419795,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 360,
          "mean": 0.8622569240097488,
          "median": 0.8916356308119637,
          "p10": 0.6793188946587698,
          "below90_fraction": 0.5194444444444445,
          "below95_fraction": 0.65
        },
        "oracle_minus_selected_mean": 0.01392749987010445,
        "fixed_already95_fraction": 0.11666666666666667,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 318,
            "mean": 0.8299689189985274,
            "median": 0.8528936377593449,
            "p10": 0.6475353266511645,
            "below90_fraction": 0.6163522012578616,
            "below95_fraction": 0.779874213836478
          },
          "oracle_coverage": {
            "n": 318,
            "mean": 0.8451086131241657,
            "median": 0.8688401111534664,
            "p10": 0.671221570457731,
            "below90_fraction": 0.5880503144654088,
            "below95_fraction": 0.7358490566037735
          },
          "oracle_gap_mean": 0.01513969412563839
        }
      }
    },
    "quest_aabb": {
      "query_head": {
        "coverage": {
          "n": 2520,
          "mean": 0.810836913287344,
          "median": 0.8960923254489899,
          "p10": 0.5008879065513611,
          "below90_fraction": 0.5075396825396825,
          "below95_fraction": 0.638095238095238
        },
        "fixed_self": {
          "n": 2520,
          "mean": 0.7213795503685625,
          "median": 0.8000373542308807,
          "p10": 0.3296941667795181,
          "below90_fraction": 0.6880952380952381,
          "below95_fraction": 0.8051587301587302
        },
        "middle_increment_mean": 0.08945736292609206,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 2520,
          "mean": 0.8622569240097488,
          "median": 0.9249649941921234,
          "p10": 0.631798779964447,
          "below90_fraction": 0.4523809523809524,
          "below95_fraction": 0.5714285714285714
        },
        "oracle_minus_selected_mean": 0.05142001072240491,
        "fixed_already95_fraction": 0.19484126984126984,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2029,
            "mean": 0.7670897568895965,
            "median": 0.8284732103347778,
            "p10": 0.4705791711807251,
            "below90_fraction": 0.6303597831444061,
            "below95_fraction": 0.7925086249383932
          },
          "oracle_coverage": {
            "n": 2029,
            "mean": 0.8304199273874749,
            "median": 0.8713791966438293,
            "p10": 0.604489266872406,
            "below90_fraction": 0.5618531296205027,
            "below95_fraction": 0.7097092163627403
          },
          "oracle_gap_mean": 0.06333017049787842
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 360,
          "mean": 0.810836913287344,
          "median": 0.8550745589392525,
          "p10": 0.5839627171201365,
          "below90_fraction": 0.5861111111111111,
          "below95_fraction": 0.7472222222222222
        },
        "fixed_self": {
          "n": 360,
          "mean": 0.7213795503685625,
          "median": 0.7712797394820622,
          "p10": 0.43673136650717687,
          "below90_fraction": 0.7916666666666666,
          "below95_fraction": 0.8833333333333333
        },
        "middle_increment_mean": 0.08945736292609206,
        "self_mass_mean": 0.10077871090156507,
        "oracle_gqa_same_budget": {
          "n": 360,
          "mean": 0.8622569240097488,
          "median": 0.8916356308119637,
          "p10": 0.6793188946587698,
          "below90_fraction": 0.5194444444444445,
          "below95_fraction": 0.65
        },
        "oracle_minus_selected_mean": 0.05142001072240491,
        "fixed_already95_fraction": 0.11666666666666667,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 318,
            "mean": 0.7874864232236886,
            "median": 0.8215505821364266,
            "p10": 0.5461252297673906,
            "below90_fraction": 0.6635220125786163,
            "below95_fraction": 0.8459119496855346
          },
          "oracle_coverage": {
            "n": 318,
            "mean": 0.8451086131241657,
            "median": 0.8688401111534664,
            "p10": 0.671221570457731,
            "below90_fraction": 0.5880503144654088,
            "below95_fraction": 0.7358490566037735
          },
          "oracle_gap_mean": 0.05762218990047719
        }
      }
    }
  },
  "untrained_layers": {
    "ranker": {
      "query_head": {
        "coverage": {
          "n": 21000,
          "mean": 0.8310717536642859,
          "median": 0.9086647927761078,
          "p10": 0.5369434356689453,
          "below90_fraction": 0.4785238095238095,
          "below95_fraction": 0.6376190476190476
        },
        "fixed_self": {
          "n": 21000,
          "mean": 0.7350242429578527,
          "median": 0.8367862105369568,
          "p10": 0.316037866473198,
          "below90_fraction": 0.6263809523809524,
          "below95_fraction": 0.7818095238095238
        },
        "middle_increment_mean": 0.09604751071594433,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 21000,
          "mean": 0.8673333887367376,
          "median": 0.9345021843910217,
          "p10": 0.6232833921909332,
          "below90_fraction": 0.39514285714285713,
          "below95_fraction": 0.562952380952381
        },
        "oracle_minus_selected_mean": 0.036261635072451705,
        "fixed_already95_fraction": 0.21819047619047618,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 16418,
            "mean": 0.7875493417705671,
            "median": 0.8578200340270996,
            "p10": 0.48991436064243316,
            "below90_fraction": 0.6120721159702766,
            "below95_fraction": 0.8155682787184797
          },
          "oracle_coverage": {
            "n": 16418,
            "mean": 0.8332973778683941,
            "median": 0.8984709978103638,
            "p10": 0.5801702737808228,
            "below90_fraction": 0.5054208795224754,
            "below95_fraction": 0.7200633451090267
          },
          "oracle_gap_mean": 0.04574803609782695
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3000,
          "mean": 0.8310717536642859,
          "median": 0.8608928280217307,
          "p10": 0.6460941936288562,
          "below90_fraction": 0.6263333333333333,
          "below95_fraction": 0.7996666666666666
        },
        "fixed_self": {
          "n": 3000,
          "mean": 0.7350242429578527,
          "median": 0.7730320862361363,
          "p10": 0.47585375862462187,
          "below90_fraction": 0.8123333333333334,
          "below95_fraction": 0.9416666666666667
        },
        "middle_increment_mean": 0.09604751071594433,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 3000,
          "mean": 0.8673333887367376,
          "median": 0.9002992808818817,
          "p10": 0.6963688667331424,
          "below90_fraction": 0.49866666666666665,
          "below95_fraction": 0.7176666666666667
        },
        "oracle_minus_selected_mean": 0.036261635072451705,
        "fixed_already95_fraction": 0.058333333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2825,
            "mean": 0.8215356275713244,
            "median": 0.8512734770774841,
            "p10": 0.6393821107489722,
            "below90_fraction": 0.6651327433628319,
            "below95_fraction": 0.8492035398230089
          },
          "oracle_coverage": {
            "n": 2825,
            "mean": 0.8598054295418461,
            "median": 0.8923023513385228,
            "p10": 0.6912635139056615,
            "below90_fraction": 0.5295575221238938,
            "below95_fraction": 0.7621238938053098
          },
          "oracle_gap_mean": 0.03826980197052183
        }
      }
    },
    "cobs": {
      "query_head": {
        "coverage": {
          "n": 21000,
          "mean": 0.8492466178518676,
          "median": 0.9255264103412628,
          "p10": 0.5717849195003509,
          "below90_fraction": 0.4298571428571429,
          "below95_fraction": 0.5890952380952381
        },
        "fixed_self": {
          "n": 21000,
          "mean": 0.7350242429578527,
          "median": 0.8367862105369568,
          "p10": 0.316037866473198,
          "below90_fraction": 0.6263809523809524,
          "below95_fraction": 0.7818095238095238
        },
        "middle_increment_mean": 0.11422237488015421,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 21000,
          "mean": 0.8673333887367376,
          "median": 0.9345021843910217,
          "p10": 0.6232833921909332,
          "below90_fraction": 0.39514285714285713,
          "below95_fraction": 0.562952380952381
        },
        "oracle_minus_selected_mean": 0.018086770884870065,
        "fixed_already95_fraction": 0.21819047619047618,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 16418,
            "mean": 0.8101415058525541,
            "median": 0.8834229409694672,
            "p10": 0.5244653999805451,
            "below90_fraction": 0.5498233645998295,
            "below95_fraction": 0.7535022536240712
          },
          "oracle_coverage": {
            "n": 16418,
            "mean": 0.8332973778683941,
            "median": 0.8984709978103638,
            "p10": 0.5801702737808228,
            "below90_fraction": 0.5054208795224754,
            "below95_fraction": 0.7200633451090267
          },
          "oracle_gap_mean": 0.02315587201584001
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3000,
          "mean": 0.8492466178518676,
          "median": 0.8804126509598322,
          "p10": 0.666441371185439,
          "below90_fraction": 0.566,
          "below95_fraction": 0.7573333333333333
        },
        "fixed_self": {
          "n": 3000,
          "mean": 0.7350242429578527,
          "median": 0.7730320862361363,
          "p10": 0.47585375862462187,
          "below90_fraction": 0.8123333333333334,
          "below95_fraction": 0.9416666666666667
        },
        "middle_increment_mean": 0.11422237488015421,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 3000,
          "mean": 0.8673333887367376,
          "median": 0.9002992808818817,
          "p10": 0.6963688667331424,
          "below90_fraction": 0.49866666666666665,
          "below95_fraction": 0.7176666666666667
        },
        "oracle_minus_selected_mean": 0.018086770884870065,
        "fixed_already95_fraction": 0.058333333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2825,
            "mean": 0.8406696610365595,
            "median": 0.8713863406862531,
            "p10": 0.6602046932492938,
            "below90_fraction": 0.6010619469026549,
            "below95_fraction": 0.8042477876106194
          },
          "oracle_coverage": {
            "n": 2825,
            "mean": 0.8598054295418461,
            "median": 0.8923023513385228,
            "p10": 0.6912635139056615,
            "below90_fraction": 0.5295575221238938,
            "below95_fraction": 0.7621238938053098
          },
          "oracle_gap_mean": 0.019135768505286626
        }
      }
    },
    "quest_aabb": {
      "query_head": {
        "coverage": {
          "n": 21000,
          "mean": 0.8237593486222572,
          "median": 0.9113200902938843,
          "p10": 0.5008469164371491,
          "below90_fraction": 0.4716666666666667,
          "below95_fraction": 0.6298571428571429
        },
        "fixed_self": {
          "n": 21000,
          "mean": 0.7350242429578527,
          "median": 0.8367862105369568,
          "p10": 0.316037866473198,
          "below90_fraction": 0.6263809523809524,
          "below95_fraction": 0.7818095238095238
        },
        "middle_increment_mean": 0.08873510567959221,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 21000,
          "mean": 0.8673333887367376,
          "median": 0.9345021843910217,
          "p10": 0.6232833921909332,
          "below90_fraction": 0.39514285714285713,
          "below95_fraction": 0.562952380952381
        },
        "oracle_minus_selected_mean": 0.04357404011448047,
        "fixed_already95_fraction": 0.21819047619047618,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 16418,
            "mean": 0.7781183074612158,
            "median": 0.858092188835144,
            "p10": 0.45516404807567595,
            "below90_fraction": 0.6033012547204288,
            "below95_fraction": 0.8056401510537216
          },
          "oracle_coverage": {
            "n": 16418,
            "mean": 0.8332973778683941,
            "median": 0.8984709978103638,
            "p10": 0.5801702737808228,
            "below90_fraction": 0.5054208795224754,
            "below95_fraction": 0.7200633451090267
          },
          "oracle_gap_mean": 0.055179070407178225
        }
      },
      "gqa_mean": {
        "coverage": {
          "n": 3000,
          "mean": 0.8237593486222572,
          "median": 0.8586100084441048,
          "p10": 0.6168455147317479,
          "below90_fraction": 0.625,
          "below95_fraction": 0.8046666666666666
        },
        "fixed_self": {
          "n": 3000,
          "mean": 0.7350242429578527,
          "median": 0.7730320862361363,
          "p10": 0.47585375862462187,
          "below90_fraction": 0.8123333333333334,
          "below95_fraction": 0.9416666666666667
        },
        "middle_increment_mean": 0.08873510567959221,
        "self_mass_mean": 0.06265996565216841,
        "oracle_gqa_same_budget": {
          "n": 3000,
          "mean": 0.8673333887367376,
          "median": 0.9002992808818817,
          "p10": 0.6963688667331424,
          "below90_fraction": 0.49866666666666665,
          "below95_fraction": 0.7176666666666667
        },
        "oracle_minus_selected_mean": 0.04357404011448047,
        "fixed_already95_fraction": 0.058333333333333334,
        "nontrivial_fixed_below95": {
          "coverage": {
            "n": 2825,
            "mean": 0.8138034715663109,
            "median": 0.8465634797300611,
            "p10": 0.6093629370842661,
            "below90_fraction": 0.6637168141592921,
            "below95_fraction": 0.8545132743362832
          },
          "oracle_coverage": {
            "n": 2825,
            "mean": 0.8598054295418461,
            "median": 0.8923023513385228,
            "p10": 0.6912635139056615,
            "below90_fraction": 0.5295575221238938,
            "below95_fraction": 0.7621238938053098
          },
          "oracle_gap_mean": 0.04600195797553519
        }
      }
    }
  },
  "wall_seconds": 510.24638194963336
}

独立产物：/root/adablock/results/longbench_all28_selector_coverage8_20261004。coverage_steps.jsonl含每观测三方法覆盖和选块，summary.json含28层、训练层/未训练层、样本与步汇总；audit.json与diagnose.py可复核。旧结果保留。


### 全28层rankloss数据构造（独立目录）

<!-- ALL28_DATA_BUILD_START -->
阶段：completed。全部28层训练及匹配验证已完成；原三层所有特征/标签逐元素一致，finite/质量归一化/无未来候选审计通过。

{
  "train": {
    "documents": 1751,
    "shards": 49028,
    "scenes": 25117792,
    "scenes_per_layer": {
      "0": 897064,
      "1": 897064,
      "2": 897064,
      "3": 897064,
      "4": 897064,
      "5": 897064,
      "6": 897064,
      "7": 897064,
      "8": 897064,
      "9": 897064,
      "10": 897064,
      "11": 897064,
      "12": 897064,
      "13": 897064,
      "14": 897064,
      "15": 897064,
      "16": 897064,
      "17": 897064,
      "18": 897064,
      "19": 897064,
      "20": 897064,
      "21": 897064,
      "22": 897064,
      "23": 897064,
      "24": 897064,
      "25": 897064,
      "26": 897064,
      "27": 897064
    },
    "bytes": 554918498344,
    "max_mass_normalization_error": 5.960464477539062e-07,
    "all_shards_finite_causal_and_old3_exact": true
  },
  "validation": {
    "documents": 16,
    "shards": 448,
    "scenes": 414064,
    "scenes_per_layer": {
      "0": 14788,
      "1": 14788,
      "2": 14788,
      "3": 14788,
      "4": 14788,
      "5": 14788,
      "6": 14788,
      "7": 14788,
      "8": 14788,
      "9": 14788,
      "10": 14788,
      "11": 14788,
      "12": 14788,
      "13": 14788,
      "14": 14788,
      "15": 14788,
      "16": 14788,
      "17": 14788,
      "18": 14788,
      "19": 14788,
      "20": 14788,
      "21": 14788,
      "22": 14788,
      "23": 14788,
      "24": 14788,
      "25": 14788,
      "26": 14788,
      "27": 14788
    },
    "bytes": 9579736880,
    "max_mass_normalization_error": 4.76837158203125e-07,
    "all_shards_finite_causal_and_old3_exact": true
  }
}

输出及审计见 /root/adablock/results/ranker_all28_build_20261004；没有启动训练。
<!-- ALL28_DATA_BUILD_END -->


全层数据收尾独立复核：1751/16序列身份及原划分完全保持；每序列0–27层唯一完整，全部文件尺寸与提交时SHA256审计记录匹配，无partial目录。RankingDataset通过顶层manifest读取首末样本成功；训练/验证/冻结测试组件与文本身份交叉均为0，原始源码及输入哈希未变。未重新读取555GB逐文件计算哈希，逐文件SHA256已在提交时生成。详细结果见results/ranker_all28_build_20261004/final_verification.json。未训练。


### 全28层DDP训练启动前吞吐/正确性检查

新随机初始化，沿用local16/global64、LR2e-4、weight_decay0.01、seed42、原rankloss/结构、epoch目标10；本次仅执行100032场景有界检查，不代表已启动全程。全28层embedding非零优化器动量且参数更新；采样器前缀完全一致且无重复，完整优化器/四rank随机状态/数据offset断点可读。

{
  "stage": "completed",
  "bounded_scenes": 100032,
  "optimizer_steps": 1563,
  "valid_scenes": 77016,
  "active_train_seconds": 138.85283770971,
  "scenes_per_second": 720.4173976561427,
  "train_only_hours_per_epoch": 9.68489165745365,
  "train_only_hours_10epochs": 96.84891657453647,
  "wall_seconds": 154.08795567043126,
  "all28_embedding_optimizer_moments_nonzero": true,
  "all28_embedding_rows_changed": true,
  "embedding_row_abs_deltas": [
    0.37499305605888367,
    0.09092285484075546,
    0.20889301598072052,
    0.2852746248245239,
    0.22904235124588013,
    0.2906956672668457,
    0.2966374158859253,
    0.23375961184501648,
    0.25467246770858765,
    0.3732065260410309,
    0.15103112161159515,
    0.24230578541755676,
    0.19096779823303223,
    0.15836530923843384,
    0.10090416669845581,
    0.45221859216690063,
    0.11545352637767792,
    0.220196932554245,
    0.262734591960907,
    0.2311108261346817,
    0.12374113500118256,
    0.16851507127285004,
    0.18347273766994476,
    0.3011518120765686,
    0.1900157928466797,
    0.22176244854927063,
    0.269395649433136,
    0.2764448821544647
  ],
  "exact_original_seed42_sampler_prefix": true,
  "duplicate_scenes": 0,
  "full_optimizer_and_rank_rng_checkpoint_verified": true,
  "initial_model_sha256": "71d24510fb7bde1c358825608193b97ee80385719ea1fd021bb9bc02a8bfd615",
  "checkpoint": "/root/adablock/checkpoints/ranker_all28_ddp_probe100032_20261004/resume_latest.pt",
  "no_full_run_started": true
}

## 2026-10-05 All28 balanced 5k/20k/50k learning curve

Fresh shared BlockRanker, seed42, AdamW lr2e-4 wd0.01, global64 (4x16), original pairwise scene-normalized loss, clip1. No epoch extension. New files utils/ranking_curve.py, utils/ranking_curve_eval.py, scripts/train_ranker_curve.py and scripts/check_ranker_curve_ddp.py; original trainers untouched. mmap avoids eager whole-shard tensor loading; per-rank shard reused for8 batches. Checkpoint every500 batches plus milestones, full optimizer and per-rank RNG; frozen plan next_batch offset.

{
  "seed": 42,
  "query_rng": "numpy default_rng PCG64; frozen full index plan",
  "batches": 60000,
  "global_batch": 64,
  "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3",
  "validation_sha256": "4a34aa8a03ed5b561469dd4a60014a79bda21b0f430db8c9846ebc29340594ac",
  "weights": {
    "4096": {
      "documents": 934,
      "full_scenes": 5803616,
      "full_scene_share": 0.2310559781687817,
      "sample_share": 0.25,
      "relative_to_full_scene_weight": 1.0819888841715235
    },
    "8192": {
      "documents": 466,
      "full_scenes": 6232128,
      "full_scene_share": 0.2481160764449359,
      "sample_share": 0.25,
      "relative_to_full_scene_weight": 1.0075929120839624
    },
    "16384": {
      "documents": 234,
      "full_scenes": 6484800,
      "full_scene_share": 0.25817555938037867,
      "sample_share": 0.25,
      "relative_to_full_scene_weight": 0.9683333333333333
    },
    "32768": {
      "documents": 117,
      "full_scenes": 6597248,
      "full_scene_share": 0.2626523860059037,
      "sample_share": 0.25,
      "relative_to_full_scene_weight": 0.9518283987505094
    }
  },
  "prefix_checks": {
    "64": {
      "presentations": 4096,
      "unique_scenes": 4096,
      "unique_documents": 32,
      "layer_counts": [
        256,
        128,
        128,
        128,
        128,
        128,
        128,
        256,
        128,
        128,
        128,
        128,
        128,
        128,
        256,
        128,
        128,
        128,
        128,
        128,
        128,
        256,
        128,
        128,
        128,
        128,
        128,
        128
      ],
      "head_counts": [
        1024,
        1024,
        1024,
        1024
      ]
    },
    "5000": {
      "presentations": 320000,
      "unique_scenes": 319244,
      "unique_documents": 1442,
      "layer_counts": [
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392
      ],
      "head_counts": [
        80000,
        80000,
        80000,
        80000
      ]
    },
    "20000": {
      "presentations": 1280000,
      "unique_scenes": 1253288,
      "unique_documents": 1751,
      "layer_counts": [
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696
      ],
      "head_counts": [
        320000,
        320000,
        320000,
        320000
      ]
    },
    "50000": {
      "presentations": 3200000,
      "unique_scenes": 3015360,
      "unique_documents": 1751,
      "layer_counts": [
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176
      ],
      "head_counts": [
        800000,
        800000,
        800000,
        800000
      ]
    }
  },
  "validation_scenes": 28672,
  "sampling": "Each 8-batch chunk: one doc from each length bucket, shuffled without replacement within bucket cycles; layer=(chunk+7*bucket)%28. Each doc-layer picks 8 distinct queries per position quartile, all 4 KV heads. Rank assignment rotates. Queries can repeat across visits; no importance weighting. Uniform bucket then uniform doc then query-quartile distribution, not full-scene distribution.",
  "validation": "Fixed seed20261005: 4 queries per quartile, same 16 positions across all layers per doc; every independent validation doc, all 28 layers and 4 heads. Selection mean coverage across budgets4,8,16,32,64,128, macro over these fixed scenes. Full independent validation on chosen checkpoint after50k.",
  "manifest_hashes": {
    "train": "edcd3e00b76ef72c4c9a36034d69cd78ceb481f14acebb6cbe15c5160de597de",
    "validation": "df094ea7587aaf059f5f61cf054ef95210b2ed8558892408772ff261ba878626"
  }
}

Manager runs original unit tests, DDP gradient normalization, uninterrupted64 vs32+resume64 bitwise equality, mmap identity checks before starting production. Validation uses independent16docs only; no old budget internal val or HotpotQA. Final best uses fixed-subset mean coverage across six budgets, then full414064-scene validation. COBS is repository legacy mean-logit baseline; Quest is quest_aabb.

### Preflight corrections before production

Initial64-step prototype measured31.49 active seconds with one length bucket per rank. Its exact resume gate failed on floating point model/optimizer tensors although all RNG states and validation metrics matched; production did not start. Retained all prototype outputs. Enable deterministic torch algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8. Global scene plan unchanged; distribute query/head groups round-robin so every rank receives one query (four heads) from each of four length buckets per batch. mmap cache4 retains these shards for8 batches. Pair RNG remains43+rank; retest uninterrupted/resume before production.

### Reproduction

{
  "source_sha256": {
    "scripts/train_ranker_curve.py": "a1b84b1fbc0cc66de3cf0fdee20a6c312bead445889be0e55f1a7638279e7a1c",
    "scripts/check_ranker_curve_ddp.py": "510867f8ad8420fc018f2b6dd1d0bec9d71733dc3346f7237e016dda81e3ae69",
    "utils/ranking_curve.py": "da170129a4603f1c63167f3122ee250c5a2870b9d5c296ced5b8327f671a9598",
    "utils/ranking_curve_eval.py": "e1d85b9c207bfb55205bea36980e45310ecb8a218fbab29f4c0c0e88fdf279f9",
    "utils/ranking_ddp.py": "43698a63898b526ac7ffe3fe50273d6d86455d683f0ce0293ca7230e1c72d5f4",
    "utils/ranking_batch.py": "914a375c89ad5ac97616affd93e877ecb7d4cfd9ee16cf1a98cfc6d2a77de4cd",
    "utils/ranking_loss.py": "7852536e86343a541e15b5e866d7f4c5fe35f9f08e05f371a3fdedb8af0d7aa1",
    "utils/ranking_data.py": "abf05e488d843c5679eb1fed2d3b0c5f3dda537dd57d66f2647d35ff991bebdd",
    "utils/ranking_eval.py": "9a655d8ca78ba3f756bc48bf1b399b3ad61f6d84825e754deae9589ef733c1b3",
    "models/block_ranker.py": "0c895cdfac47aadd07081375f21bc21a559d29d2017f21a81fab65d06140e601"
  },
  "seed": 42,
  "lr": 0.0002,
  "weight_decay": 0.01,
  "global_batch": 64,
  "world_size": 4,
  "local_batch": 16,
  "targets": [
    5000,
    20000,
    50000
  ],
  "torch_deterministic_algorithms": true,
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
  "rank_mapping": "plan batch reshape(4,4,4).transpose(1,0,2).reshape(64), then each rank takes16; one query from each length bucket per rank",
  "production_command": "/root/miniconda3/envs/ada/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_ranker_curve.py --work /root/adablock/results/ranker_all28_curve_20261005 --out /root/adablock/checkpoints/ranker_all28_curve_20261005",
  "resume": "same command plus --resume; manager also resumes from resume_latest.pt automatically up to3 retries",
  "no_longbench_or_budget_internal_val": true
}

Validation parallelism: assign entire shards round-robin across four ranks so each rank evaluates all16 docs/length buckets and7 layers; prevents length-ordered validation stragglers. Final trainer SHA256 9b1ab1bad7849d3652cd5491b0e87b551d38d8d0ae6f136174b26a67448abd8e; snapshot train_ranker_curve_final.py.

### Production gate: numerically equivalent resume

{
  "unit_tests": "9 passed",
  "DDP_global_effective_scene_gradient": "passed against serial reference",
  "mmap_scene_identity": "bitwise equal four length buckets",
  "resume_rng_and_data_offset": "exact",
  "resume_parameters_max_abs_excluding_output_bias": 5.960464477539063e-08,
  "resume_output_bias_abs_difference": 4.819035530090332e-05,
  "output_bias_semantics": "common additive score constant; cancels in pairwise loss and ranking",
  "resume_optimizer_max_abs": {
    "exp_avg": 9.313225746154785e-10,
    "exp_avg_sq": 2.2737367544323206e-13
  },
  "resume_validation_groups": "all metrics identical on112 fixed test scenes",
  "all28_embeddings_and_optimizer_moments": "updated in64step tests",
  "validation_partition_sizes": [
    7168,
    7168,
    7168,
    7168
  ],
  "all16_validation_docs_per_rank": true,
  "floating_point_resume_bitwise": false
}

DDP restart preserves all RNG states, data offset, optimizer step/moments within1e-8 and ranking metrics. Parameter bitwise equality is not claimed. Original rank loss is invariant to common output bias; tiny reduction errors can be amplified by Adam epsilon for this bias. Production uses unchanged bias parameter and loss.

### Production launched (manager3735, torchrun3736)

{"stage": "training", "step": 400, "batches": 400, "scenes": 25600, "train_seconds": 88.08735273219645, "scenes_per_second": 290.6206079075759, "loss": 0.5722489577576102, "remaining_train_seconds": 10922.83173879236}

ETA is extrapolated active training only, validation and checkpoint/I/O wall overhead separate. No full10epoch run.

### Unique-query and workload audit

{
  "milestones": {
    "5000": {
      "presentations": 320000,
      "unique_query_positions_doc_token": 73871,
      "unique_scenes": 319244,
      "documents": 1442,
      "unique_doc_layers": 2435,
      "layer_counts": [
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392,
        11520,
        11520,
        11392,
        11392,
        11392,
        11392,
        11392
      ],
      "head_counts": [
        80000,
        80000,
        80000,
        80000
      ],
      "bucket_scene_counts": {
        "4096": 80000,
        "8192": 80000,
        "16384": 80000,
        "32768": 80000
      },
      "bucket_docs": {
        "4096": 625,
        "8192": 466,
        "16384": 234,
        "32768": 117
      },
      "unique_shards": 2435,
      "global_adjacent_shard_changes": 19999,
      "mean_candidates": 463.7491,
      "candidate_quantiles": [
        50.0,
        279.0,
        1213.0
      ],
      "mean_target_position": 7956.322
    },
    "20000": {
      "presentations": 1280000,
      "unique_query_positions_doc_token": 180425,
      "unique_scenes": 1253288,
      "documents": 1751,
      "unique_doc_layers": 8624,
      "layer_counts": [
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696,
        45824,
        45696,
        45696,
        45696,
        45696,
        45696,
        45696
      ],
      "head_counts": [
        320000,
        320000,
        320000,
        320000
      ],
      "bucket_scene_counts": {
        "4096": 320000,
        "8192": 320000,
        "16384": 320000,
        "32768": 320000
      },
      "bucket_docs": {
        "4096": 934,
        "8192": 466,
        "16384": 234,
        "32768": 117
      },
      "unique_shards": 8624,
      "global_adjacent_shard_changes": 79999,
      "mean_candidates": 464.179525,
      "candidate_quantiles": [
        50.0,
        279.0,
        1213.0
      ],
      "mean_target_position": 7963.1659
    },
    "50000": {
      "presentations": 3200000,
      "unique_query_positions_doc_token": 220073,
      "unique_scenes": 3015360,
      "documents": 1751,
      "unique_doc_layers": 17473,
      "layer_counts": [
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176,
        114304,
        114304,
        114304,
        114304,
        114304,
        114304,
        114176
      ],
      "head_counts": [
        800000,
        800000,
        800000,
        800000
      ],
      "bucket_scene_counts": {
        "4096": 800000,
        "8192": 800000,
        "16384": 800000,
        "32768": 800000
      },
      "bucket_docs": {
        "4096": 934,
        "8192": 466,
        "16384": 234,
        "32768": 117
      },
      "unique_shards": 17473,
      "global_adjacent_shard_changes": 199999,
      "mean_candidates": 464.18551,
      "candidate_quantiles": [
        50.0,
        279.0,
        1210.0
      ],
      "mean_target_position": 7963.25756
    }
  },
  "same_100032_presentations": {
    "old_probe": {
      "presentations": 100032,
      "unique_query_positions_doc_token": 23962,
      "unique_scenes": 100032,
      "documents": 184,
      "unique_doc_layers": 197,
      "layer_counts": [
        5260,
        1620,
        1432,
        2460,
        3192,
        2764,
        3444,
        5328,
        5404,
        3668,
        3864,
        4484,
        3468,
        4176,
        2828,
        4144,
        3536,
        5820,
        7920,
        3492,
        1916,
        2136,
        2512,
        5072,
        1580,
        2768,
        1592,
        4152
      ],
      "head_counts": [
        25002,
        25002,
        25023,
        25005
      ],
      "bucket_scene_counts": {
        "4096": 23980,
        "8192": 24844,
        "16384": 21788,
        "32768": 29420
      },
      "bucket_docs": {
        "4096": 101,
        "8192": 47,
        "16384": 21,
        "32768": 15
      },
      "unique_shards": 197,
      "global_adjacent_shard_changes": 196,
      "mean_candidates": 490.22172904670504,
      "candidate_quantiles": [
        51.0,
        294.0,
        1329.0
      ],
      "mean_target_position": 8379.273952335252,
      "per_rank_adjacent_shard_changes": [
        196,
        196,
        196,
        196
      ]
    },
    "new_plan": {
      "presentations": 100032,
      "unique_query_positions_doc_token": 24853,
      "unique_scenes": 100000,
      "documents": 705,
      "unique_doc_layers": 781,
      "layer_counts": [
        3584,
        3584,
        3584,
        3584,
        3584,
        3584,
        3504,
        3584,
        3584,
        3584,
        3584,
        3584,
        3584,
        3504,
        3584,
        3584,
        3584,
        3584,
        3584,
        3584,
        3504,
        3584,
        3584,
        3584,
        3584,
        3584,
        3584,
        3504
      ],
      "head_counts": [
        25008,
        25008,
        25008,
        25008
      ],
      "bucket_scene_counts": {
        "4096": 25008,
        "8192": 25008,
        "16384": 25008,
        "32768": 25008
      },
      "bucket_docs": {
        "4096": 196,
        "8192": 196,
        "16384": 196,
        "32768": 117
      },
      "unique_shards": 781,
      "global_adjacent_shard_changes": 6251,
      "mean_candidates": 463.37236084452974,
      "candidate_quantiles": [
        50.0,
        278.0,
        1209.0
      ],
      "mean_target_position": 7950.346769033909,
      "per_rank_adjacent_shard_changes": [
        6251,
        6251,
        6251,
        6251
      ]
    }
  },
  "validation": {
    "scenes": 28672,
    "unique_doc_query_positions": 256,
    "documents": 16,
    "layers": 28,
    "heads": 4,
    "selection": "16 distinct query positions per doc:4 per position quartile, fixed seed20261005, same positions across28layers/all4heads; equal doc/length weighting; full val micro-scene mean is different weighting",
    "full_validation_scenes": 414064
  }
}

### Passive utilization sample

{
  "duration_seconds": 28.320426385849714,
  "gpu_average": [
    {
      "gpu": 0,
      "utilization_percent": 11.8,
      "memory_utilization_percent": 1.2,
      "memory_used_mib": 942.0
    },
    {
      "gpu": 1,
      "utilization_percent": 14.0,
      "memory_utilization_percent": 0.8,
      "memory_used_mib": 964.0
    },
    {
      "gpu": 2,
      "utilization_percent": 13.8,
      "memory_utilization_percent": 1.15,
      "memory_used_mib": 962.0
    },
    {
      "gpu": 3,
      "utilization_percent": 15.8,
      "memory_utilization_percent": 1.15,
      "memory_used_mib": 940.0
    }
  ],
  "processes": {
    "3740": {
      "cpu_percent_one_core": 56.425704127055084,
      "minor_faults": 30256,
      "major_faults": 4266,
      "read_bytes": 219570176,
      "rchar": 165003844,
      "end_wchan": "0",
      "end_state": "R"
    },
    "3741": {
      "cpu_percent_one_core": 57.485010212043605,
      "minor_faults": 36077,
      "major_faults": 3577,
      "read_bytes": 122843136,
      "rchar": 165003844,
      "end_wchan": "0",
      "end_state": "R"
    },
    "3742": {
      "cpu_percent_one_core": 58.47369589136622,
      "minor_faults": 54643,
      "major_faults": 4395,
      "read_bytes": 125128704,
      "rchar": 165003844,
      "end_wchan": "0",
      "end_state": "R"
    },
    "3743": {
      "cpu_percent_one_core": 57.44970000921065,
      "minor_faults": 70255,
      "major_faults": 3486,
      "read_bytes": 123830272,
      "rchar": 165003844,
      "end_wchan": "0",
      "end_state": "R"
    }
  },
  "scope": "passive20second sample, no profiler/benchmark; counters do not attribute time to forward, CPU sampler, I/O, or NCCL"
}

### Throughput interpretation and first document coverage

{
  "first_full_document_coverage_batch": 7465,
  "same_count_cache_misses": {
    "old_probe": [
      197,
      197,
      197,
      197
    ],
    "new_plan": [
      784,
      784,
      784,
      784
    ]
  },
  "throughput_interpretation": [
    "Candidate count decreases490.22 to463.37, not a larger candidate workload explanation.",
    "New sampler increases document diversity184 to705 and shard diversity197 to781 per100032scenes; mmap cache misses are about4x per rank. Logical shard switches are not equal to physical loads.",
    "Passive28.32seconds: GPUs11.8-15.8percent, CPU56-58percent of one core per rank, 15724 major page faults total and591372288 read bytes; proves page faults/reads exist, does not prove their time share.",
    "Same ranking loss/pair sampler/DDP normalization utilities. New execution also enables deterministic algorithms, checks finite scores and synchronizesCUDA at each batch end; their costs are unprofiled.",
    "No staged timing in current logs: cannot apportion the slowdown among mmap/metadata I/O, CPU sampling, finite/sync overhead, deterministic kernels and NCCL. No running configuration changed."
  ]
}

### All28 balanced curve: step 5000

{"step": 5000, "batches": 5000, "train_seconds": 1154.1156445480883, "mean_coverage": 0.9140659537123618, "validation_seconds": 183.25790057517588, "methods": {"mean_pool": {"required_k": 103.29401506696429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22154017857142858, "coverage@4": 0.8595444538596404, "middle_mass@4": 0.028073574004185402, "mass_gap@4": 0.012494773354520332, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.34120396205357145, "coverage@8": 0.8738016119114117, "middle_mass@8": 0.04233073205595669, "mass_gap@8": 0.014458516573014637, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3854282924107143, "coverage@16": 0.8926807680551645, "middle_mass@16": 0.06120988819970944, "mass_gap@16": 0.0154434184874676, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.44712611607142855, "coverage@32": 0.9154367450683468, "middle_mass@32": 0.08396586521289175, "mass_gap@32": 0.015234900406046645, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5374232700892857, "coverage@64": 0.9404029605994797, "middle_mass@64": 0.10893208074402468, "mass_gap@64": 0.013481082904289923, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.65185546875, "coverage@128": 0.9637980210549971, "middle_mass@128": 0.13232714119954206, "mass_gap@128": 0.010489511703486951, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7784598214285714}, "quest_aabb": {"required_k": 102.18659319196429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22586495535714285, "coverage@4": 0.8552399703118224, "middle_mass@4": 0.023769090456367374, "mass_gap@4": 0.016799256902338362, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.33436802455357145, "coverage@8": 0.8678682969835234, "middle_mass@8": 0.036397417128068356, "mass_gap@8": 0.020391831500902974, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3739536830357143, "coverage@16": 0.8857562991672437, "middle_mass@16": 0.05428541931178873, "mass_gap@16": 0.022367887375388307, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.42958286830357145, "coverage@32": 0.9085181700222079, "middle_mass@32": 0.07704729016675291, "mass_gap@32": 0.02215347545218549, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5169852120535714, "coverage@64": 0.9350250728308281, "middle_mass@64": 0.10355419297537313, "mass_gap@64": 0.018858970672941462, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6405901227678571, "coverage@128": 0.9609602492911608, "middle_mass@128": 0.1294893694357057, "mass_gap@128": 0.013327283467323323, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7741350446428571}, "cobs_legacy_mean_logits": {"required_k": 80.44158063616071, "unreachable_rate": 0.0, "over_max_budget_rate": 0.18338448660714285, "coverage@4": 0.8606201155599199, "middle_mass@4": 0.02914923570446499, "mass_gap@4": 0.01141911165424075, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.35103934151785715, "coverage@8": 0.8763599204054398, "middle_mass@8": 0.044889040549984695, "mass_gap@8": 0.011900208078986634, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.40018136160714285, "coverage@16": 0.8967311604188442, "middle_mass@16": 0.06526028056338903, "mass_gap@16": 0.011393026123788008, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.46836635044642855, "coverage@32": 0.9204387451564037, "middle_mass@32": 0.08896786530094854, "mass_gap@32": 0.010232900317989862, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5680803571428571, "coverage@64": 0.9456652366518407, "middle_mass@64": 0.11419435679638568, "mass_gap@64": 0.008218806851928915, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6925223214285714, "coverage@128": 0.968769246333645, "middle_mass@128": 0.13729836647819, "mass_gap@128": 0.005518286424838996, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8166155133928571}, "oracle": {"required_k": 66.78086635044643, "unreachable_rate": 0.0, "over_max_budget_rate": 0.15342494419642858, "coverage@4": 0.8720392272141607, "middle_mass@4": 0.040568347358705736, "mass_gap@4": 0.0, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.37744140625, "coverage@8": 0.8882601284844264, "middle_mass@8": 0.05678924862897133, "mass_gap@8": 0.0, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.43314034598214285, "coverage@16": 0.908124186542632, "middle_mass@16": 0.07665330668717704, "mass_gap@16": 0.0, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.5052664620535714, "coverage@32": 0.9306716454743934, "middle_mass@32": 0.0992007656189384, "mass_gap@32": 0.0, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.6101771763392857, "coverage@64": 0.9538840435037697, "middle_mass@64": 0.12241316364831459, "mass_gap@64": 0.0, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.7308872767857143, "coverage@128": 0.974287532758484, "middle_mass@128": 0.142816652903029, "mass_gap@128": 0.0, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8465750558035714}, "learned": {"required_k": 77.53358677455357, "unreachable_rate": 0.0, "over_max_budget_rate": 0.17532784598214285, "coverage@4": 0.8639518251845267, "middle_mass@4": 0.03248094532907163, "mass_gap@4": 0.00808740202963411, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.35672433035714285, "coverage@8": 0.8796472823382545, "middle_mass@8": 0.04817640248279947, "mass_gap@8": 0.00861284614617186, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.40771484375, "coverage@16": 0.8997359393258656, "middle_mass@16": 0.06826505947041055, "mass_gap@16": 0.008388247216766485, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.47628348214285715, "coverage@32": 0.9231274844567908, "middle_mass@32": 0.09165660460133575, "mass_gap@32": 0.007544161017602643, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5767996651785714, "coverage@64": 0.9478812030385614, "middle_mass@64": 0.1164103231831063, "mass_gap@64": 0.006002840465208286, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.699951171875, "coverage@128": 0.9700519879301712, "middle_mass@128": 0.13858110807471619, "mass_gap@128": 0.004235544828312803, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8246721540178571}}, "all28_embedding_rows_changed": true, "all28_optimizer_moments_nonzero": true}

### First5k interpretation

{
  "step": 5000,
  "scenes": 28672,
  "validation_seconds": 183.25790057517588,
  "groups": {
    "overall": {
      "scenes": 28672,
      "methods": {
        "learned": {
          "coverage@4": 0.8639518251845267,
          "coverage@8": 0.8796472823382545,
          "coverage@16": 0.8997359393258656,
          "coverage@32": 0.9231274844567908,
          "coverage@64": 0.9478812030385614,
          "coverage@128": 0.9700519879301712,
          "required_k": 77.53358677455357
        },
        "cobs_legacy_mean_logits": {
          "coverage@4": 0.8606201155599199,
          "coverage@8": 0.8763599204054398,
          "coverage@16": 0.8967311604188442,
          "coverage@32": 0.9204387451564037,
          "coverage@64": 0.9456652366518407,
          "coverage@128": 0.968769246333645,
          "required_k": 80.44158063616071
        },
        "quest_aabb": {
          "coverage@4": 0.8552399703118224,
          "coverage@8": 0.8678682969835234,
          "coverage@16": 0.8857562991672437,
          "coverage@32": 0.9085181700222079,
          "coverage@64": 0.9350250728308281,
          "coverage@128": 0.9609602492911608,
          "required_k": 102.18659319196429
        },
        "oracle": {
          "coverage@4": 0.8720392272141607,
          "coverage@8": 0.8882601284844264,
          "coverage@16": 0.908124186542632,
          "coverage@32": 0.9306716454743934,
          "coverage@64": 0.9538840435037697,
          "coverage@128": 0.974287532758484,
          "required_k": 66.78086635044643
        }
      }
    },
    "fixed_below_rho": {
      "scenes": 20801,
      "methods": {
        "learned": {
          "coverage@4": 0.8193821472428174,
          "coverage@8": 0.8398915104122558,
          "coverage@16": 0.8662381409116837,
          "coverage@32": 0.8970752349327747,
          "coverage@64": 0.9298916482473852,
          "coverage@128": 0.9594583367151335,
          "required_k": 106.87192923417143
        },
        "cobs_legacy_mean_logits": {
          "coverage@4": 0.8148781764890621,
          "coverage@8": 0.8354300733062413,
          "coverage@16": 0.862135676929163,
          "coverage@32": 0.8933944579997135,
          "coverage@64": 0.9268441120075872,
          "coverage@128": 0.9576912623366336,
          "required_k": 110.88029421662421
        },
        "quest_aabb": {
          "coverage@4": 0.8080928552198346,
          "coverage@8": 0.8245364279685681,
          "coverage@16": 0.8478994229392929,
          "coverage@32": 0.8778046005549056,
          "coverage@64": 0.9127951209243998,
          "coverage@128": 0.9472530539899114,
          "required_k": 140.853516657853
        },
        "oracle": {
          "coverage@4": 0.8301549124788266,
          "coverage@8": 0.85134610699259,
          "coverage@16": 0.8773834347880008,
          "coverage@32": 0.907109850878037,
          "coverage@64": 0.937895453248742,
          "coverage@128": 0.9651512380410536,
          "required_k": 92.05043026777558
        }
      }
    }
  },
  "layer_c32_wins_vs_cobs": 25,
  "current_progress": {
    "stage": "training",
    "step": 5100,
    "batches": 5100,
    "scenes": 326400,
    "train_seconds": 1176.9381093457341,
    "scenes_per_second": 277.32979109789164,
    "loss": 0.3287089311511471,
    "remaining_train_seconds": 10361.670805808522
  },
  "manager_status": "training_attempt_0",
  "checkpoint": {
    "step": 5000,
    "batch": 5000,
    "rng_ranks": 4,
    "optimizer_states": 16
  }
}

Full per-layer comparison: results/ranker_all28_curve_20261005/first_5k_summary.json. This is the fixed independent validation subset, not final held-out test or full-validation selection.

### Requested final document-equal metrics

Original running trainer and selection criterion unchanged. Separate observer records actual coverage from each milestone checkpoint. After original50k and selected-checkpoint full validation complete and GPU contexts are released, run one additional full validation adding per-document aggregates, report scene and document weighting, and assert original scene metrics match within1e-9. This adds evaluation cost, not training or checkpoint selection changes. Then record GPU release.

### Actual coverage at5000 optimizer steps

{"optimizer_steps": 5000, "actual_batches": 5000, "presentations": 320000, "unique_scenes": 319244, "unique_query_positions": 73871, "query_identity": "document plus aligned record index; all28 layer record positions were previously audited identical", "documents": 1442, "doc_layer_pairs": 2435, "layer_counts": [11520, 11520, 11392, 11392, 11392, 11392, 11392, 11520, 11520, 11392, 11392, 11392, 11392, 11392, 11520, 11520, 11392, 11392, 11392, 11392, 11392, 11520, 11520, 11392, 11392, 11392, 11392, 11392], "head_counts": [80000, 80000, 80000, 80000], "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3", "checkpoint_bytes": 815282, "rank_rng_states": 4}

### Next evaluation cost plan (not launched)

{
  "status": "proposal_only_not_started",
  "checkpoint": "Use preselected curve checkpoint; never select using HotpotQA or frozen test.",
  "hotpotqa": {
    "samples": 200,
    "prior_four_gpu_all28_wall_seconds": 3338.8885905742645,
    "prior_peak_reserved_mib": 18666,
    "new_learned_only_estimate": "approximately1hour on4GPUs, dependent on generated lengths",
    "comparison": "Can reuse verified completed dense200 and previous all28 predictions for quality under identical historical manual-greedy128-token protocol. Do not reuse timings as speed baseline. COBS and Quest historical runs incomplete; do not restart stopped jobs. Fresh full200COBS+Quest would add roughly2hours if each uses4-way sample sharding, not yet measured.",
    "protocol": "Freeze model/tokenizer/prompt/token hashes, fullSDPA prefill, block16,sink32tokens,local32blocks,middle32,all28 learned layers,no fallback. Historical protocol is not claimed identical to official cap32/generation-config defaults; any protocol change requires matched reruns."
  },
  "frozen_test": {
    "documents": 16,
    "existing_layers": [
      12,
      20,
      27
    ],
    "existing_scenes": 44448,
    "proposed_all28_scenes": 414848,
    "estimated_all28_bytes": 9583179312.0,
    "construction_estimate": "about5minutes4GPU extraction from prior independent-val cost model, plus startup and audits; estimate only",
    "evaluation_scene_rate_extrapolation_minutes": 44.19187914911793,
    "evaluation_caveat": "linear scene-rate projection only; full length weighting and fixed I/O costs differ, update from completed full validation",
    "protocol": "Preserve exact frozen16source docs and original component/text isolation. Build new all28 directory, match old12/20/27 tensors, never overwrite oldtest. Evaluate selected checkpoint once with COBS/Quest/Oracle on identical scenes; no test-driven tuning."
  }
}

### Actual coverage at20000 optimizer steps

{"optimizer_steps": 20000, "actual_batches": 20000, "presentations": 1280000, "unique_scenes": 1253288, "unique_query_positions": 180425, "query_identity": "document plus aligned record index; all28 layer record positions were previously audited identical", "documents": 1751, "doc_layer_pairs": 8624, "layer_counts": [45824, 45696, 45696, 45696, 45696, 45696, 45696, 45824, 45696, 45696, 45696, 45696, 45696, 45696, 45824, 45696, 45696, 45696, 45696, 45696, 45696, 45824, 45696, 45696, 45696, 45696, 45696, 45696], "head_counts": [320000, 320000, 320000, 320000], "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3", "checkpoint_bytes": 820786, "rank_rng_states": 4}

### All28 balanced curve: step 20000

{"step": 20000, "batches": 20000, "train_seconds": 4456.983422290534, "mean_coverage": 0.915625809423668, "validation_seconds": 48.49729620851576, "methods": {"mean_pool": {"required_k": 103.29401506696429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22154017857142858, "coverage@4": 0.8595444538596404, "middle_mass@4": 0.028073574004185402, "mass_gap@4": 0.012494773354520332, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.34120396205357145, "coverage@8": 0.8738016119114117, "middle_mass@8": 0.04233073205595669, "mass_gap@8": 0.014458516573014637, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3854282924107143, "coverage@16": 0.8926807680551645, "middle_mass@16": 0.06120988819970944, "mass_gap@16": 0.0154434184874676, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.44712611607142855, "coverage@32": 0.9154367450683468, "middle_mass@32": 0.08396586521289175, "mass_gap@32": 0.015234900406046645, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5374232700892857, "coverage@64": 0.9404029605994797, "middle_mass@64": 0.10893208074402468, "mass_gap@64": 0.013481082904289923, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.65185546875, "coverage@128": 0.9637980210549971, "middle_mass@128": 0.13232714119954206, "mass_gap@128": 0.010489511703486951, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7784598214285714}, "quest_aabb": {"required_k": 102.18659319196429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22586495535714285, "coverage@4": 0.8552399703118224, "middle_mass@4": 0.023769090456367374, "mass_gap@4": 0.016799256902338362, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.33436802455357145, "coverage@8": 0.8678682969835234, "middle_mass@8": 0.036397417128068356, "mass_gap@8": 0.020391831500902974, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3739536830357143, "coverage@16": 0.8857562991672437, "middle_mass@16": 0.05428541931178873, "mass_gap@16": 0.022367887375388307, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.42958286830357145, "coverage@32": 0.9085181700222079, "middle_mass@32": 0.07704729016675291, "mass_gap@32": 0.02215347545218549, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5169852120535714, "coverage@64": 0.9350250728308281, "middle_mass@64": 0.10355419297537313, "mass_gap@64": 0.018858970672941462, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6405901227678571, "coverage@128": 0.9609602492911608, "middle_mass@128": 0.1294893694357057, "mass_gap@128": 0.013327283467323323, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7741350446428571}, "cobs_legacy_mean_logits": {"required_k": 80.44158063616071, "unreachable_rate": 0.0, "over_max_budget_rate": 0.18338448660714285, "coverage@4": 0.8606201155599199, "middle_mass@4": 0.02914923570446499, "mass_gap@4": 0.01141911165424075, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.35103934151785715, "coverage@8": 0.8763599204054398, "middle_mass@8": 0.044889040549984695, "mass_gap@8": 0.011900208078986634, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.40018136160714285, "coverage@16": 0.8967311604188442, "middle_mass@16": 0.06526028056338903, "mass_gap@16": 0.011393026123788008, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.46836635044642855, "coverage@32": 0.9204387451564037, "middle_mass@32": 0.08896786530094854, "mass_gap@32": 0.010232900317989862, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5680803571428571, "coverage@64": 0.9456652366518407, "middle_mass@64": 0.11419435679638568, "mass_gap@64": 0.008218806851928915, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6925223214285714, "coverage@128": 0.968769246333645, "middle_mass@128": 0.13729836647819, "mass_gap@128": 0.005518286424838996, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8166155133928571}, "oracle": {"required_k": 66.78086635044643, "unreachable_rate": 0.0, "over_max_budget_rate": 0.15342494419642858, "coverage@4": 0.8720392272141607, "middle_mass@4": 0.040568347358705736, "mass_gap@4": 0.0, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.37744140625, "coverage@8": 0.8882601284844264, "middle_mass@8": 0.05678924862897133, "mass_gap@8": 0.0, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.43314034598214285, "coverage@16": 0.908124186542632, "middle_mass@16": 0.07665330668717704, "mass_gap@16": 0.0, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.5052664620535714, "coverage@32": 0.9306716454743934, "middle_mass@32": 0.0992007656189384, "mass_gap@32": 0.0, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.6101771763392857, "coverage@64": 0.9538840435037697, "middle_mass@64": 0.12241316364831459, "mass_gap@64": 0.0, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.7308872767857143, "coverage@128": 0.974287532758484, "middle_mass@128": 0.142816652903029, "mass_gap@128": 0.0, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8465750558035714}, "learned": {"required_k": 74.32334681919643, "unreachable_rate": 0.0, "over_max_budget_rate": 0.16922433035714285, "coverage@4": 0.8655114305159363, "middle_mass@4": 0.03404055066048122, "mass_gap@4": 0.006527796698224516, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.36146763392857145, "coverage@8": 0.8813454150545841, "middle_mass@8": 0.04987453519912905, "mass_gap@8": 0.006914713429842274, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.41360909598214285, "coverage@16": 0.9014961001962037, "middle_mass@16": 0.07002522034074866, "mass_gap@16": 0.006628086346428371, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.48311941964285715, "coverage@32": 0.9249103507528295, "middle_mass@32": 0.09343947089737448, "mass_gap@32": 0.0057612947215639155, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5853445870535714, "coverage@64": 0.9493154427012309, "middle_mass@64": 0.11784456284577592, "mass_gap@64": 0.004568600802538673, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.708251953125, "coverage@128": 0.9711761173212238, "middle_mass@128": 0.13970523746576877, "mass_gap@128": 0.0031114154372602343, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8307756696428571}}, "all28_embedding_rows_changed": true, "all28_optimizer_moments_nonzero": true}

### 20k comparison summary

{
  "step": 20000,
  "validation_scenes": 28672,
  "baseline_metrics_identical_to_5k": true,
  "groups": {
    "overall": {
      "learned": {
        "coverage@32": 0.9249103507528295,
        "required_k": 74.32334681919643
      },
      "cobs_legacy_mean_logits": {
        "coverage@32": 0.9204387451564037,
        "required_k": 80.44158063616071
      },
      "quest_aabb": {
        "coverage@32": 0.9085181700222079,
        "required_k": 102.18659319196429
      },
      "oracle": {
        "coverage@32": 0.9306716454743934,
        "required_k": 66.78086635044643
      }
    },
    "fixed_below_rho": {
      "learned": {
        "coverage@32": 0.8994315664953197,
        "required_k": 102.44694966588145
      },
      "cobs_legacy_mean_logits": {
        "coverage@32": 0.8933944579997135,
        "required_k": 110.88029421662421
      },
      "quest_aabb": {
        "coverage@32": 0.8778046005549056,
        "required_k": 140.853516657853
      },
      "oracle": {
        "coverage@32": 0.907109850878037,
        "required_k": 92.05043026777558
      }
    }
  },
  "c32_winning_layers_vs_cobs": 28,
  "c32_change_from_5k_pp": 0.1782866296038632,
  "actual_coverage": {
    "optimizer_steps": 20000,
    "actual_batches": 20000,
    "presentations": 1280000,
    "unique_scenes": 1253288,
    "unique_query_positions": 180425,
    "query_identity": "document plus aligned record index; all28 layer record positions were previously audited identical",
    "documents": 1751,
    "doc_layer_pairs": 8624,
    "layer_counts": [
      45824,
      45696,
      45696,
      45696,
      45696,
      45696,
      45696,
      45824,
      45696,
      45696,
      45696,
      45696,
      45696,
      45696,
      45824,
      45696,
      45696,
      45696,
      45696,
      45696,
      45696,
      45824,
      45696,
      45696,
      45696,
      45696,
      45696,
      45696
    ],
    "head_counts": [
      320000,
      320000,
      320000,
      320000
    ],
    "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3",
    "checkpoint_bytes": 820786,
    "rank_rng_states": 4
  },
  "checkpoint_files": {
    "step_020000_resume.pt": 820786,
    "step_020000.pt": 252406
  }
}

### Actual coverage at50000 optimizer steps

{"optimizer_steps": 50000, "actual_batches": 50000, "presentations": 3200000, "unique_scenes": 3015360, "unique_query_positions": 220073, "query_identity": "document plus aligned record index; all28 layer record positions were previously audited identical", "documents": 1751, "doc_layer_pairs": 17473, "layer_counts": [114304, 114304, 114304, 114304, 114304, 114304, 114176, 114304, 114304, 114304, 114304, 114304, 114304, 114176, 114304, 114304, 114304, 114304, 114304, 114304, 114176, 114304, 114304, 114304, 114304, 114304, 114304, 114176], "head_counts": [800000, 800000, 800000, 800000], "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3", "checkpoint_bytes": 826226, "rank_rng_states": 4}

### All28 balanced curve: step 50000

{"step": 50000, "batches": 50000, "train_seconds": 10447.275184473023, "mean_coverage": 0.9160404053179985, "validation_seconds": 55.3077353592962, "methods": {"mean_pool": {"required_k": 103.29401506696429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22154017857142858, "coverage@4": 0.8595444538596404, "middle_mass@4": 0.028073574004185402, "mass_gap@4": 0.012494773354520332, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.34120396205357145, "coverage@8": 0.8738016119114117, "middle_mass@8": 0.04233073205595669, "mass_gap@8": 0.014458516573014637, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3854282924107143, "coverage@16": 0.8926807680551645, "middle_mass@16": 0.06120988819970944, "mass_gap@16": 0.0154434184874676, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.44712611607142855, "coverage@32": 0.9154367450683468, "middle_mass@32": 0.08396586521289175, "mass_gap@32": 0.015234900406046645, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5374232700892857, "coverage@64": 0.9404029605994797, "middle_mass@64": 0.10893208074402468, "mass_gap@64": 0.013481082904289923, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.65185546875, "coverage@128": 0.9637980210549971, "middle_mass@128": 0.13232714119954206, "mass_gap@128": 0.010489511703486951, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7784598214285714}, "quest_aabb": {"required_k": 102.18659319196429, "unreachable_rate": 0.0, "over_max_budget_rate": 0.22586495535714285, "coverage@4": 0.8552399703118224, "middle_mass@4": 0.023769090456367374, "mass_gap@4": 0.016799256902338362, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.33436802455357145, "coverage@8": 0.8678682969835234, "middle_mass@8": 0.036397417128068356, "mass_gap@8": 0.020391831500902974, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.3739536830357143, "coverage@16": 0.8857562991672437, "middle_mass@16": 0.05428541931178873, "mass_gap@16": 0.022367887375388307, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.42958286830357145, "coverage@32": 0.9085181700222079, "middle_mass@32": 0.07704729016675291, "mass_gap@32": 0.02215347545218549, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5169852120535714, "coverage@64": 0.9350250728308281, "middle_mass@64": 0.10355419297537313, "mass_gap@64": 0.018858970672941462, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6405901227678571, "coverage@128": 0.9609602492911608, "middle_mass@128": 0.1294893694357057, "mass_gap@128": 0.013327283467323323, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.7741350446428571}, "cobs_legacy_mean_logits": {"required_k": 80.44158063616071, "unreachable_rate": 0.0, "over_max_budget_rate": 0.18338448660714285, "coverage@4": 0.8606201155599199, "middle_mass@4": 0.02914923570446499, "mass_gap@4": 0.01141911165424075, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.35103934151785715, "coverage@8": 0.8763599204054398, "middle_mass@8": 0.044889040549984695, "mass_gap@8": 0.011900208078986634, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.40018136160714285, "coverage@16": 0.8967311604188442, "middle_mass@16": 0.06526028056338903, "mass_gap@16": 0.011393026123788008, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.46836635044642855, "coverage@32": 0.9204387451564037, "middle_mass@32": 0.08896786530094854, "mass_gap@32": 0.010232900317989862, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5680803571428571, "coverage@64": 0.9456652366518407, "middle_mass@64": 0.11419435679638568, "mass_gap@64": 0.008218806851928915, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.6925223214285714, "coverage@128": 0.968769246333645, "middle_mass@128": 0.13729836647819, "mass_gap@128": 0.005518286424838996, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8166155133928571}, "oracle": {"required_k": 66.78086635044643, "unreachable_rate": 0.0, "over_max_budget_rate": 0.15342494419642858, "coverage@4": 0.8720392272141607, "middle_mass@4": 0.040568347358705736, "mass_gap@4": 0.0, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.37744140625, "coverage@8": 0.8882601284844264, "middle_mass@8": 0.05678924862897133, "mass_gap@8": 0.0, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.43314034598214285, "coverage@16": 0.908124186542632, "middle_mass@16": 0.07665330668717704, "mass_gap@16": 0.0, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.5052664620535714, "coverage@32": 0.9306716454743934, "middle_mass@32": 0.0992007656189384, "mass_gap@32": 0.0, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.6101771763392857, "coverage@64": 0.9538840435037697, "middle_mass@64": 0.12241316364831459, "mass_gap@64": 0.0, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.7308872767857143, "coverage@128": 0.974287532758484, "middle_mass@128": 0.142816652903029, "mass_gap@128": 0.0, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8465750558035714}, "learned": {"required_k": 73.22324916294643, "unreachable_rate": 0.0, "over_max_budget_rate": 0.1669921875, "coverage@4": 0.8658530602628892, "middle_mass@4": 0.03438218040743418, "mass_gap@4": 0.006186166951271556, "actual_k@4": 3.984375, "coverage_target_hit@4": 0.36300223214285715, "coverage@8": 0.8817968340515999, "middle_mass@8": 0.050325954196144815, "mass_gap@8": 0.006463294432826513, "actual_k@8": 7.9296875, "coverage_target_hit@8": 0.41605050223214285, "coverage@16": 0.901973450918052, "middle_mass@16": 0.0705025710625969, "mass_gap@16": 0.006150735624580144, "actual_k@16": 15.7421875, "coverage_target_hit@16": 0.4861537388392857, "coverage@32": 0.9254228546399299, "middle_mass@32": 0.09395197478447491, "mass_gap@32": 0.005248790834463486, "actual_k@32": 30.76953125, "coverage_target_hit@32": 0.5877859933035714, "coverage@64": 0.9497504763709964, "middle_mass@64": 0.11827959651554133, "mass_gap@64": 0.004133567132773257, "actual_k@64": 59.64453125, "coverage_target_hit@64": 0.7113909040178571, "coverage@128": 0.971445755664523, "middle_mass@128": 0.13997487580906792, "mass_gap@128": 0.0028417770939610826, "actual_k@128": 111.8046875, "coverage_target_hit@128": 0.8330078125}}, "all28_embedding_rows_changed": true, "all28_optimizer_moments_nonzero": true}

### All28 curve complete

Best step 50000; full validation: {"scenes": 414064, "methods": {"mean_pool": {"required_k": 167.86547490243055, "unreachable_rate": 0.0, "over_max_budget_rate": 0.3604756752579311, "coverage@4": 0.8239700565895686, "middle_mass@4": 0.029560655242276555, "mass_gap@4": 0.01755559760592573, "actual_k@4": 3.9972951041384905, "coverage_target_hit@4": 0.23707929209011167, "coverage@8": 0.8394609088789238, "middle_mass@8": 0.04505150753163161, "mass_gap@8": 0.02037550155940917, "actual_k@8": 7.977278874763321, "coverage_target_hit@8": 0.26941487306310136, "coverage@16": 0.8599443791250461, "middle_mass@16": 0.06553497777775398, "mass_gap@16": 0.022178951967690715, "actual_k@16": 15.885312415472004, "coverage_target_hit@16": 0.3205712160438966, "coverage@32": 0.8851856805175236, "middle_mass@32": 0.0907762791702315, "mass_gap@32": 0.02255856886109878, "actual_k@32": 31.493643494725454, "coverage_target_hit@32": 0.39851085822481547, "coverage@64": 0.9135423286174585, "middle_mass@64": 0.11913292727016642, "mass_gap@64": 0.02113698792222523, "actual_k@64": 61.87936164457668, "coverage_target_hit@64": 0.5072766528845782, "coverage@128": 0.9420348688165094, "middle_mass@128": 0.14762546746921731, "mass_gap@128": 0.017796822466618182, "actual_k@128": 119.32702190965648, "coverage_target_hit@128": 0.6395243247420689}, "quest_aabb": {"required_k": 159.4623826268403, "unreachable_rate": 0.0, "over_max_budget_rate": 0.35288989141775184, "coverage@4": 0.8214816629803338, "middle_mass@4": 0.0270722616330417, "mass_gap@4": 0.020043991215160588, "actual_k@4": 3.9972951041384905, "coverage_target_hit@4": 0.2450587348815642, "coverage@8": 0.8347565281606234, "middle_mass@8": 0.04034712681333135, "mass_gap@8": 0.02507988227770943, "actual_k@8": 7.977278874763321, "coverage_target_hit@8": 0.27381274392364463, "coverage@16": 0.8534248785183917, "middle_mass@16": 0.05901547717109941, "mass_gap@16": 0.02869845257434529, "actual_k@16": 15.885312415472004, "coverage_target_hit@16": 0.31917771165810116, "coverage@32": 0.8782511760195182, "middle_mass@32": 0.08384177467222605, "mass_gap@32": 0.02949307335910423, "actual_k@32": 31.493643494725454, "coverage_target_hit@32": 0.39387872406198077, "coverage@64": 0.9079890213713708, "middle_mass@64": 0.11357962002407884, "mass_gap@64": 0.0266902951683128, "actual_k@64": 61.87936164457668, "coverage_target_hit@64": 0.506059449746899, "coverage@128": 0.9391836079709218, "middle_mass@128": 0.14477420662362964, "mass_gap@128": 0.020648083312205856, "actual_k@128": 119.32702190965648, "coverage_target_hit@128": 0.6471101085822482}, "cobs_legacy_mean_logits": {"required_k": 124.82701224931411, "unreachable_rate": 0.0, "over_max_budget_rate": 0.29311652304957686, "coverage@4": 0.8258961488483693, "middle_mass@4": 0.031486747501077494, "mass_gap@4": 0.01562950534712479, "actual_k@4": 3.9972951041384905, "coverage_target_hit@4": 0.25111818462846325, "coverage@8": 0.843505465297186, "middle_mass@8": 0.04909606394989375, "mass_gap@8": 0.01633094514114703, "actual_k@8": 7.977278874763321, "coverage_target_hit@8": 0.29037781598979867, "coverage@16": 0.8662465521276341, "middle_mass@16": 0.0718371507803422, "mass_gap@16": 0.0158767789651025, "actual_k@16": 15.885312415472004, "coverage_target_hit@16": 0.35075254067004136, "coverage@32": 0.8933546930210355, "middle_mass@32": 0.09894529167374333, "mass_gap@32": 0.014389556357586953, "actual_k@32": 31.493643494725454, "coverage_target_hit@32": 0.4418133428648711, "coverage@64": 0.9227663194962956, "middle_mass@64": 0.12835691814900343, "mass_gap@64": 0.0119129970433882, "actual_k@64": 61.87936164457668, "coverage_target_hit@64": 0.5656541017813671, "coverage@128": 0.9511280492326348, "middle_mass@128": 0.15671864788534273, "mass_gap@128": 0.008703642050492767, "actual_k@128": 119.32702190965648, "coverage_target_hit@128": 0.7068834769504231}, "oracle": {"required_k": 102.04973385756792, "unreachable_rate": 0.0, "over_max_budget_rate": 0.2478336682252019, "coverage@4": 0.8415256541954944, "middle_mass@4": 0.047116252848202284, "mass_gap@4": 0.0, "actual_k@4": 3.9972951041384905, "coverage_target_hit@4": 0.2810048688125507, "coverage@8": 0.8598364104383328, "middle_mass@8": 0.06542700909104078, "mass_gap@8": 0.0, "actual_k@8": 7.977278874763321, "coverage_target_hit@8": 0.3270484755979752, "coverage@16": 0.8821233310927369, "middle_mass@16": 0.0877139297454447, "mass_gap@16": 0.0, "actual_k@16": 15.885312415472004, "coverage_target_hit@16": 0.394608079910352, "coverage@32": 0.9077442493786224, "middle_mass@32": 0.11333484803133027, "mass_gap@32": 0.0, "actual_k@32": 31.493643494725454, "coverage_target_hit@32": 0.49178629390625606, "coverage@64": 0.9346793165396837, "middle_mass@64": 0.14026991519239165, "mass_gap@64": 0.0, "actual_k@64": 61.87936164457668, "coverage_target_hit@64": 0.6171340662313073, "coverage@128": 0.9598316912831275, "middle_mass@128": 0.1654222899358355, "mass_gap@128": 0.0, "actual_k@128": 119.32702190965648, "coverage_target_hit@128": 0.7521663317747981}, "learned": {"required_k": 113.13429807952394, "unreachable_rate": 0.0, "over_max_budget_rate": 0.2708735847598439, "coverage@4": 0.8330539627664691, "middle_mass@4": 0.03864456141917709, "mass_gap@4": 0.008471691429025196, "actual_k@4": 3.9972951041384905, "coverage_target_hit@4": 0.26443738166080605, "coverage@8": 0.8509714577085542, "middle_mass@8": 0.056562056361262186, "mass_gap@8": 0.008864952729778592, "actual_k@8": 7.977278874763321, "coverage_target_hit@8": 0.30686802040264305, "coverage@16": 0.8735307810946569, "middle_mass@16": 0.0791213797473648, "mass_gap@16": 0.008592549998079893, "actual_k@16": 15.885312415472004, "coverage_target_hit@16": 0.3705200162293752, "coverage@32": 0.9000911386756967, "middle_mass@32": 0.10568173732840443, "mass_gap@32": 0.00765311070292583, "actual_k@32": 31.493643494725454, "coverage_target_hit@32": 0.46469627883612197, "coverage@64": 0.9284300582463054, "middle_mass@64": 0.13402065689901313, "mass_gap@64": 0.006249258293378519, "actual_k@64": 61.87936164457668, "coverage_target_hit@64": 0.5897397503767533, "coverage@128": 0.9552515466180842, "middle_mass@128": 0.160842145270792, "mass_gap@128": 0.004580144665043495, "actual_k@128": 119.32702190965648, "coverage_target_hit@128": 0.7291264152401561}}}

### Final checkpoint and layer audit

{
  "budgets_middle_blocks": [
    4,
    8,
    16,
    32,
    64,
    128
  ],
  "block_tokens": 16,
  "coverage_definition": "Attention mass of fixed sink+local+self plus top-min(k,candidate_count) middle blocks; denominator is full causal attention mass. Fixed sets deduplicated. k excludes fixed blocks.",
  "required_k_definition": "Smallest ranked middle-block prefix reaching rho0.95; zero when fixed mass already reaches rho; not capped at128; unreachable_rate reported separately.",
  "unit_definitions": {
    "scene": "packed-document query position x layer x KVhead; teacher mass averages the7queryheads sharing thatKVhead",
    "unique_query": "deduplicate by packed document and target token position, collapsing layer and KVhead",
    "document": "packed sequence, not every constituent source document"
  },
  "checkpoint_inventory": {
    "5000": {
      "resume_sha256": "d6ca1ab5d6c8c33a31af4ebe356e0411559641d03b8f1e43e0bef63440264c2c",
      "resume_bytes": 815282,
      "model_matches_resume": true,
      "optimizer_steps": [
        5000
      ],
      "rank_rng_states": 4,
      "finite": true
    },
    "20000": {
      "resume_sha256": "248c79907a2ca2ed4f5b7861f561f2e801f2485468c2fb6fd217ee384f3a9769",
      "resume_bytes": 820786,
      "model_matches_resume": true,
      "optimizer_steps": [
        20000
      ],
      "rank_rng_states": 4,
      "finite": true
    },
    "50000": {
      "resume_sha256": "5a921cc9f7951bda57a77313b232dbfba74d42bc92fa99a9b0211d8bbd9ba9f3",
      "resume_bytes": 826226,
      "model_matches_resume": true,
      "optimizer_steps": [
        50000
      ],
      "rank_rng_states": 4,
      "finite": true
    }
  },
  "best_step": 50000,
  "best_sha256": "d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae",
  "best_matches_50k": true,
  "original_execution_sources_unchanged": true,
  "actual_50k_coverage": {
    "optimizer_steps": 50000,
    "actual_batches": 50000,
    "presentations": 3200000,
    "unique_scenes": 3015360,
    "unique_query_positions": 220073,
    "query_identity": "document plus aligned record index; all28 layer record positions were previously audited identical",
    "documents": 1751,
    "doc_layer_pairs": 17473,
    "layer_counts": [
      114304,
      114304,
      114304,
      114304,
      114304,
      114304,
      114176,
      114304,
      114304,
      114304,
      114304,
      114304,
      114304,
      114176,
      114304,
      114304,
      114304,
      114304,
      114304,
      114304,
      114176,
      114304,
      114304,
      114304,
      114304,
      114304,
      114304,
      114176
    ],
    "head_counts": [
      800000,
      800000,
      800000,
      800000
    ],
    "plan_sha256": "afaa912ee1fc79bdc7c52b0527f9554f37b23c7f90d1a01bd7abd90275838ab3",
    "checkpoint_bytes": 826226,
    "rank_rng_states": 4
  },
  "full_validation_scenes": 414064,
  "full_validation_unreachable_rates": {
    "mean_pool": 0.0,
    "quest_aabb": 0.0,
    "cobs_legacy_mean_logits": 0.0,
    "oracle": 0.0,
    "learned": 0.0
  }
}

Full per-layer tables and all-six-budget win/tie/loss counts: results/ranker_all28_curve_20261005/checkpoint_and_layer_audit.json.

### Final validation: scene and document weighting

{
  "best_step": 50000,
  "selection_unchanged": "fixed-subset macro mean coverage over six budgets",
  "dataset_scenes": 414064,
  "scene_weighted": {
    "scenes": 414064,
    "methods": {
      "mean_pool": {
        "required_k": 167.86547490243055,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.3604756752579311,
        "coverage@4": 0.8239700565895686,
        "middle_mass@4": 0.029560655242276555,
        "mass_gap@4": 0.01755559760592573,
        "actual_k@4": 3.9972951041384905,
        "coverage_target_hit@4": 0.23707929209011167,
        "coverage@8": 0.8394609088789238,
        "middle_mass@8": 0.04505150753163161,
        "mass_gap@8": 0.02037550155940917,
        "actual_k@8": 7.977278874763321,
        "coverage_target_hit@8": 0.26941487306310136,
        "coverage@16": 0.8599443791250461,
        "middle_mass@16": 0.06553497777775398,
        "mass_gap@16": 0.022178951967690715,
        "actual_k@16": 15.885312415472004,
        "coverage_target_hit@16": 0.3205712160438966,
        "coverage@32": 0.8851856805175236,
        "middle_mass@32": 0.0907762791702315,
        "mass_gap@32": 0.02255856886109878,
        "actual_k@32": 31.493643494725454,
        "coverage_target_hit@32": 0.39851085822481547,
        "coverage@64": 0.9135423286174585,
        "middle_mass@64": 0.11913292727016642,
        "mass_gap@64": 0.02113698792222523,
        "actual_k@64": 61.87936164457668,
        "coverage_target_hit@64": 0.5072766528845782,
        "coverage@128": 0.9420348688165094,
        "middle_mass@128": 0.14762546746921731,
        "mass_gap@128": 0.017796822466618182,
        "actual_k@128": 119.32702190965648,
        "coverage_target_hit@128": 0.6395243247420689
      },
      "quest_aabb": {
        "required_k": 159.4623826268403,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.35288989141775184,
        "coverage@4": 0.8214816629803338,
        "middle_mass@4": 0.0270722616330417,
        "mass_gap@4": 0.020043991215160588,
        "actual_k@4": 3.9972951041384905,
        "coverage_target_hit@4": 0.2450587348815642,
        "coverage@8": 0.8347565281606234,
        "middle_mass@8": 0.04034712681333135,
        "mass_gap@8": 0.02507988227770943,
        "actual_k@8": 7.977278874763321,
        "coverage_target_hit@8": 0.27381274392364463,
        "coverage@16": 0.8534248785183917,
        "middle_mass@16": 0.05901547717109941,
        "mass_gap@16": 0.02869845257434529,
        "actual_k@16": 15.885312415472004,
        "coverage_target_hit@16": 0.31917771165810116,
        "coverage@32": 0.8782511760195182,
        "middle_mass@32": 0.08384177467222605,
        "mass_gap@32": 0.02949307335910423,
        "actual_k@32": 31.493643494725454,
        "coverage_target_hit@32": 0.39387872406198077,
        "coverage@64": 0.9079890213713708,
        "middle_mass@64": 0.11357962002407884,
        "mass_gap@64": 0.0266902951683128,
        "actual_k@64": 61.87936164457668,
        "coverage_target_hit@64": 0.506059449746899,
        "coverage@128": 0.9391836079709218,
        "middle_mass@128": 0.14477420662362964,
        "mass_gap@128": 0.020648083312205856,
        "actual_k@128": 119.32702190965648,
        "coverage_target_hit@128": 0.6471101085822482
      },
      "cobs_legacy_mean_logits": {
        "required_k": 124.82701224931411,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.29311652304957686,
        "coverage@4": 0.8258961488483693,
        "middle_mass@4": 0.031486747501077494,
        "mass_gap@4": 0.01562950534712479,
        "actual_k@4": 3.9972951041384905,
        "coverage_target_hit@4": 0.25111818462846325,
        "coverage@8": 0.843505465297186,
        "middle_mass@8": 0.04909606394989375,
        "mass_gap@8": 0.01633094514114703,
        "actual_k@8": 7.977278874763321,
        "coverage_target_hit@8": 0.29037781598979867,
        "coverage@16": 0.8662465521276341,
        "middle_mass@16": 0.0718371507803422,
        "mass_gap@16": 0.0158767789651025,
        "actual_k@16": 15.885312415472004,
        "coverage_target_hit@16": 0.35075254067004136,
        "coverage@32": 0.8933546930210355,
        "middle_mass@32": 0.09894529167374333,
        "mass_gap@32": 0.014389556357586953,
        "actual_k@32": 31.493643494725454,
        "coverage_target_hit@32": 0.4418133428648711,
        "coverage@64": 0.9227663194962956,
        "middle_mass@64": 0.12835691814900343,
        "mass_gap@64": 0.0119129970433882,
        "actual_k@64": 61.87936164457668,
        "coverage_target_hit@64": 0.5656541017813671,
        "coverage@128": 0.9511280492326348,
        "middle_mass@128": 0.15671864788534273,
        "mass_gap@128": 0.008703642050492767,
        "actual_k@128": 119.32702190965648,
        "coverage_target_hit@128": 0.7068834769504231
      },
      "oracle": {
        "required_k": 102.04973385756792,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.2478336682252019,
        "coverage@4": 0.8415256541954944,
        "middle_mass@4": 0.047116252848202284,
        "mass_gap@4": 0.0,
        "actual_k@4": 3.9972951041384905,
        "coverage_target_hit@4": 0.2810048688125507,
        "coverage@8": 0.8598364104383328,
        "middle_mass@8": 0.06542700909104078,
        "mass_gap@8": 0.0,
        "actual_k@8": 7.977278874763321,
        "coverage_target_hit@8": 0.3270484755979752,
        "coverage@16": 0.8821233310927369,
        "middle_mass@16": 0.0877139297454447,
        "mass_gap@16": 0.0,
        "actual_k@16": 15.885312415472004,
        "coverage_target_hit@16": 0.394608079910352,
        "coverage@32": 0.9077442493786224,
        "middle_mass@32": 0.11333484803133027,
        "mass_gap@32": 0.0,
        "actual_k@32": 31.493643494725454,
        "coverage_target_hit@32": 0.49178629390625606,
        "coverage@64": 0.9346793165396837,
        "middle_mass@64": 0.14026991519239165,
        "mass_gap@64": 0.0,
        "actual_k@64": 61.87936164457668,
        "coverage_target_hit@64": 0.6171340662313073,
        "coverage@128": 0.9598316912831275,
        "middle_mass@128": 0.1654222899358355,
        "mass_gap@128": 0.0,
        "actual_k@128": 119.32702190965648,
        "coverage_target_hit@128": 0.7521663317747981
      },
      "learned": {
        "required_k": 113.13429807952394,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.2708735847598439,
        "coverage@4": 0.8330539627664691,
        "middle_mass@4": 0.03864456141917709,
        "mass_gap@4": 0.008471691429025196,
        "actual_k@4": 3.9972951041384905,
        "coverage_target_hit@4": 0.26443738166080605,
        "coverage@8": 0.8509714577085542,
        "middle_mass@8": 0.056562056361262186,
        "mass_gap@8": 0.008864952729778592,
        "actual_k@8": 7.977278874763321,
        "coverage_target_hit@8": 0.30686802040264305,
        "coverage@16": 0.8735307810946569,
        "middle_mass@16": 0.0791213797473648,
        "mass_gap@16": 0.008592549998079893,
        "actual_k@16": 15.885312415472004,
        "coverage_target_hit@16": 0.3705200162293752,
        "coverage@32": 0.9000911386756967,
        "middle_mass@32": 0.10568173732840443,
        "mass_gap@32": 0.00765311070292583,
        "actual_k@32": 31.493643494725454,
        "coverage_target_hit@32": 0.46469627883612197,
        "coverage@64": 0.9284300582463054,
        "middle_mass@64": 0.13402065689901313,
        "mass_gap@64": 0.006249258293378519,
        "actual_k@64": 61.87936164457668,
        "coverage_target_hit@64": 0.5897397503767533,
        "coverage@128": 0.9552515466180842,
        "middle_mass@128": 0.160842145270792,
        "mass_gap@128": 0.004580144665043495,
        "actual_k@128": 119.32702190965648,
        "coverage_target_hit@128": 0.7291264152401561
      }
    }
  },
  "document_equal": {
    "documents": 16,
    "methods": {
      "mean_pool": {
        "required_k": 102.41600285913033,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.22082198657572827,
        "coverage@4": 0.8577221122775047,
        "middle_mass@4": 0.02798023757438043,
        "mass_gap@4": 0.012748018462229392,
        "actual_k@4": 3.9936006980124343,
        "coverage_target_hit@4": 0.32576394313844736,
        "coverage@8": 0.8723711202178743,
        "middle_mass@8": 0.0426292455147501,
        "mass_gap@8": 0.014725895025789453,
        "actual_k@8": 7.954597068934784,
        "coverage_target_hit@8": 0.3673164664850449,
        "coverage@16": 0.8916662634722603,
        "middle_mass@16": 0.06192438876913591,
        "mass_gap@16": 0.015744855098178023,
        "actual_k@16": 15.77877682950924,
        "coverage_target_hit@16": 0.43145061637244514,
        "coverage@32": 0.9151687535735777,
        "middle_mass@32": 0.08542687887045347,
        "mass_gap@32": 0.015427879410087185,
        "actual_k@32": 31.03588442557715,
        "coverage_target_hit@32": 0.5273755975274314,
        "coverage@64": 0.9406425817461831,
        "middle_mass@64": 0.11090070704305877,
        "mass_gap@64": 0.013572450528682348,
        "actual_k@64": 59.985091917389006,
        "coverage_target_hit@64": 0.6503353730379006,
        "coverage@128": 0.9642978414999205,
        "middle_mass@128": 0.13455596679679627,
        "mass_gap@128": 0.01036776562003779,
        "actual_k@128": 111.6234760997168,
        "coverage_target_hit@128": 0.7791780134242718
      },
      "quest_aabb": {
        "required_k": 100.63628196840294,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.22470524238496364,
        "coverage@4": 0.8534160956256819,
        "middle_mass@4": 0.023674220922557446,
        "mass_gap@4": 0.017054035114052384,
        "actual_k@4": 3.9936006980124343,
        "coverage_target_hit@4": 0.32083689112575314,
        "coverage@8": 0.8663520957971761,
        "middle_mass@8": 0.036610221094051654,
        "mass_gap@8": 0.020744919446487896,
        "actual_k@8": 7.954597068934784,
        "coverage_target_hit@8": 0.3567828320571473,
        "coverage@16": 0.8845775939776632,
        "middle_mass@16": 0.05483571927453879,
        "mass_gap@16": 0.02283352459277514,
        "actual_k@16": 15.77877682950924,
        "coverage_target_hit@16": 0.41399589453781105,
        "coverage@32": 0.9083375501762698,
        "middle_mass@32": 0.0785956754731453,
        "mass_gap@32": 0.022259082807395362,
        "actual_k@32": 31.03588442557715,
        "coverage_target_hit@32": 0.5067319090805346,
        "coverage@64": 0.9355278115409797,
        "middle_mass@64": 0.10578593683785532,
        "mass_gap@64": 0.018687220733885776,
        "actual_k@64": 59.985091917389006,
        "coverage_target_hit@64": 0.6366508236547639,
        "coverage@128": 0.9616595860007026,
        "middle_mass@128": 0.13191771129757845,
        "mass_gap@128": 0.013006021119255567,
        "actual_k@128": 111.6234760997168,
        "coverage_target_hit@128": 0.7752947576150363
      },
      "cobs_legacy_mean_logits": {
        "required_k": 80.21154846821075,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.1826118594718459,
        "coverage@4": 0.8591103238895744,
        "middle_mass@4": 0.029368449186450173,
        "mass_gap@4": 0.01135980685015966,
        "actual_k@4": 3.9936006980124343,
        "coverage_target_hit@4": 0.3361037226392279,
        "coverage@8": 0.875202973028267,
        "middle_mass@8": 0.045461098325142724,
        "mass_gap@8": 0.011894042215396822,
        "actual_k@8": 7.954597068934784,
        "coverage_target_hit@8": 0.38249975738738334,
        "coverage@16": 0.8958620628395466,
        "middle_mass@16": 0.06612018813642222,
        "mass_gap@16": 0.011549055730891723,
        "actual_k@16": 15.77877682950924,
        "coverage_target_hit@16": 0.4533193711830946,
        "coverage@32": 0.9202794744889512,
        "middle_mass@32": 0.09053759978582694,
        "mass_gap@32": 0.010317158494713714,
        "actual_k@32": 31.03588442557715,
        "coverage_target_hit@32": 0.5575389914979826,
        "coverage@64": 0.9460184898723972,
        "middle_mass@64": 0.116276615169273,
        "mass_gap@64": 0.008196542402468093,
        "actual_k@64": 59.985091917389006,
        "coverage_target_hit@64": 0.6886371209055135,
        "coverage@128": 0.969155968993642,
        "middle_mass@128": 0.13941409429051765,
        "mass_gap@128": 0.0055096381263163964,
        "actual_k@128": 111.6234760997168,
        "coverage_target_hit@128": 0.8173881405281541
      },
      "oracle": {
        "required_k": 66.42513896480344,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.1526951505483388,
        "coverage@4": 0.8704701307397342,
        "middle_mass@4": 0.04072825603660983,
        "mass_gap@4": 0.0,
        "actual_k@4": 3.9936006980124343,
        "coverage_target_hit@4": 0.36214408957206046,
        "coverage@8": 0.8870970152436637,
        "middle_mass@8": 0.05735514054053956,
        "mass_gap@8": 0.0,
        "actual_k@8": 7.954597068934784,
        "coverage_target_hit@8": 0.41484424734896974,
        "coverage@16": 0.9074111185704383,
        "middle_mass@16": 0.07766924386731396,
        "mass_gap@16": 0.0,
        "actual_k@16": 15.77877682950924,
        "coverage_target_hit@16": 0.4928048831489271,
        "coverage@32": 0.9305966329836649,
        "middle_mass@32": 0.10085475828054065,
        "mass_gap@32": 0.0,
        "actual_k@32": 31.03588442557715,
        "coverage_target_hit@32": 0.6011893958500002,
        "coverage@64": 0.9542150322748655,
        "middle_mass@64": 0.12447315757174111,
        "mass_gap@64": 0.0,
        "actual_k@64": 59.985091917389006,
        "coverage_target_hit@64": 0.7292332902562172,
        "coverage@128": 0.9746656071199583,
        "middle_mass@128": 0.14492373241683404,
        "mass_gap@128": 0.0,
        "actual_k@128": 111.6234760997168,
        "coverage_target_hit@128": 0.8473048494516611
      },
      "learned": {
        "required_k": 72.93283258579056,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.16709317411220861,
        "coverage@4": 0.8642785299291318,
        "middle_mass@4": 0.03453665522600749,
        "mass_gap@4": 0.006191600810602337,
        "actual_k@4": 3.9936006980124343,
        "coverage_target_hit@4": 0.34764664751413055,
        "coverage@8": 0.8806215817906811,
        "middle_mass@8": 0.05087970708755681,
        "mass_gap@8": 0.006475433452982733,
        "actual_k@8": 7.954597068934784,
        "coverage_target_hit@8": 0.39740224828463566,
        "coverage@16": 0.9012075744685002,
        "middle_mass@16": 0.07146569976537587,
        "mass_gap@16": 0.006203544101938073,
        "actual_k@16": 15.77877682950924,
        "coverage_target_hit@16": 0.4716022357614287,
        "coverage@32": 0.9252098723677582,
        "middle_mass@32": 0.09546799766463387,
        "mass_gap@32": 0.00538676061590676,
        "actual_k@32": 31.03588442557715,
        "coverage_target_hit@32": 0.5786672137771426,
        "coverage@64": 0.9500373610910672,
        "middle_mass@64": 0.12029548638794284,
        "mass_gap@64": 0.004177671183798276,
        "actual_k@64": 59.985091917389006,
        "coverage_target_hit@64": 0.7088852689818718,
        "coverage@128": 0.9718533150886257,
        "middle_mass@128": 0.14211144038550133,
        "mass_gap@128": 0.0028122920313326982,
        "actual_k@128": 111.6234760997168,
        "coverage_target_hit@128": 0.8329068258877915
      }
    }
  },
  "scene_weighted_fixed_below_rho": {
    "scenes": 334979,
    "methods": {
      "mean_pool": {
        "required_k": 207.49673860152427,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.4455801706972676,
        "coverage@4": 0.7872518073299748,
        "middle_mass@4": 0.03515880216068897,
        "mass_gap@4": 0.02135056640007732,
        "actual_k@4": 3.999794016938375,
        "coverage_target_hit@4": 0.05696177969365244,
        "coverage@8": 0.8057258989816003,
        "middle_mass@8": 0.053632893812314675,
        "mass_gap@8": 0.024790088232185845,
        "actual_k@8": 7.996146027064383,
        "coverage_target_hit@8": 0.09693144943414363,
        "coverage@16": 0.8302249028789929,
        "middle_mass@16": 0.0781318977097071,
        "mass_gap@16": 0.026999105715939692,
        "actual_k@16": 15.96462464811227,
        "coverage_target_hit@16": 0.16016526409118184,
        "coverage@32": 0.8605114604416524,
        "middle_mass@32": 0.10841845527236668,
        "mass_gap@32": 0.02748859093676312,
        "actual_k@32": 31.77908465903833,
        "coverage_target_hit@32": 0.2565056316963153,
        "coverage@64": 0.8946656349487063,
        "middle_mass@64": 0.1425726297794206,
        "mass_gap@64": 0.025796377989552456,
        "actual_k@64": 62.8605285704477,
        "coverage_target_hit@64": 0.3909498804402664,
        "coverage@128": 0.9291376333036425,
        "middle_mass@128": 0.17704462813435676,
        "mass_gap@128": 0.021770451078370306,
        "actual_k@128": 122.5051033049833,
        "coverage_target_hit@128": 0.5544198293027324
      },
      "quest_aabb": {
        "required_k": 197.1097650897519,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.43620346350069705,
        "coverage@4": 0.7845380876807704,
        "middle_mass@4": 0.032445082511484695,
        "mass_gap@4": 0.024064286049281593,
        "actual_k@4": 3.999794016938375,
        "coverage_target_hit@4": 0.06682508455753942,
        "coverage@8": 0.8003809836199836,
        "middle_mass@8": 0.04828797845069773,
        "mass_gap@8": 0.03013500359380279,
        "actual_k@8": 7.996146027064383,
        "coverage_target_hit@8": 0.10236761110397966,
        "coverage@16": 0.822692552807818,
        "middle_mass@16": 0.07059954763853227,
        "mass_gap@16": 0.034531455787114536,
        "actual_k@16": 15.96462464811227,
        "coverage_target_hit@16": 0.15844276805411683,
        "coverage@32": 0.8524354222922601,
        "middle_mass@32": 0.1003424171229743,
        "mass_gap@32": 0.03556462908615551,
        "actual_k@32": 31.77908465903833,
        "coverage_target_hit@32": 0.2507798996354995,
        "coverage@64": 0.8881700605462868,
        "middle_mass@64": 0.136077055377001,
        "mass_gap@64": 0.03229195239197207,
        "actual_k@64": 62.8605285704477,
        "coverage_target_hit@64": 0.38944530851187686,
        "coverage@128": 0.9258123389606512,
        "middle_mass@128": 0.1737193337913654,
        "mass_gap@128": 0.02509574542136166,
        "actual_k@128": 122.5051033049833,
        "coverage_target_hit@128": 0.563796536499303
      },
      "cobs_legacy_mean_logits": {
        "required_k": 154.29734998313327,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.36231823487442494,
        "coverage@4": 0.7896062855395595,
        "middle_mass@4": 0.03751328037027382,
        "mass_gap@4": 0.018996088190492477,
        "actual_k@4": 3.999794016938375,
        "coverage_target_hit@4": 0.07431510632009768,
        "coverage@8": 0.8106713186075118,
        "middle_mass@8": 0.05857831343822615,
        "mass_gap@8": 0.01984466860627436,
        "actual_k@8": 7.996146027064383,
        "coverage_target_hit@8": 0.12284352153418572,
        "coverage@16": 0.8379375157297982,
        "middle_mass@16": 0.08584451056051244,
        "mass_gap@16": 0.01928649286513436,
        "actual_k@16": 15.96462464811227,
        "coverage_target_hit@16": 0.19747208033936456,
        "coverage@32": 0.870515367616566,
        "middle_mass@32": 0.11842236244728024,
        "mass_gap@32": 0.017484683761849557,
        "actual_k@32": 31.77908465903833,
        "coverage_target_hit@32": 0.3100313751011257,
        "coverage@64": 0.9059698553507458,
        "middle_mass@64": 0.15387685018146002,
        "mass_gap@64": 0.014492157587513032,
        "actual_k@64": 62.8605285704477,
        "coverage_target_hit@64": 0.4631096277677108,
        "coverage@128": 0.9402929779398985,
        "middle_mass@128": 0.18819997277061282,
        "mass_gap@128": 0.010615106442114242,
        "actual_k@128": 122.5051033049833,
        "coverage_target_hit@128": 0.6376817651255751
      },
      "oracle": {
        "required_k": 126.14259699861782,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.30634457682421884,
        "coverage@4": 0.8086023737300521,
        "middle_mass@4": 0.05650936856076629,
        "mass_gap@4": 0.0,
        "actual_k@4": 3.999794016938375,
        "coverage_target_hit@4": 0.11125772063323372,
        "coverage@8": 0.8305159872137863,
        "middle_mass@8": 0.07842298204450052,
        "mass_gap@8": 0.0,
        "actual_k@8": 7.996146027064383,
        "coverage_target_hit@8": 0.16817173613868333,
        "coverage@16": 0.8572240085949325,
        "middle_mass@16": 0.1051310034256468,
        "mass_gap@16": 0.0,
        "actual_k@16": 15.96462464811227,
        "coverage_target_hit@16": 0.2516814486878282,
        "coverage@32": 0.8880000513784155,
        "middle_mass@32": 0.1359070462091298,
        "mass_gap@32": 0.0,
        "actual_k@32": 31.77908465903833,
        "coverage_target_hit@32": 0.37180241149445187,
        "coverage@64": 0.9204620129382588,
        "middle_mass@64": 0.16836900776897307,
        "mass_gap@64": 0.0,
        "actual_k@64": 62.8605285704477,
        "coverage_target_hit@64": 0.5267434675009478,
        "coverage@128": 0.9509080843820128,
        "middle_mass@128": 0.19881507921272706,
        "mass_gap@128": 0.0,
        "actual_k@128": 122.5051033049833,
        "coverage_target_hit@128": 0.6936554231757812
      },
      "learned": {
        "required_k": 139.84410963075297,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.33482397404016373,
        "coverage@4": 0.7983017646686854,
        "middle_mass@4": 0.046208759499399676,
        "mass_gap@4": 0.010300609061366612,
        "actual_k@4": 3.999794016938375,
        "coverage_target_hit@4": 0.09077882494126498,
        "coverage@8": 0.8197505714879778,
        "middle_mass@8": 0.06765756631869206,
        "mass_gap@8": 0.010765415725808454,
        "actual_k@8": 7.996146027064383,
        "coverage_target_hit@8": 0.1432268888497488,
        "coverage@16": 0.8468072159435134,
        "middle_mass@16": 0.09471421077422762,
        "mass_gap@16": 0.010416792651419185,
        "actual_k@16": 15.96462464811227,
        "coverage_target_hit@16": 0.2219064478668812,
        "coverage@32": 0.878745465758867,
        "middle_mass@32": 0.1266524605895813,
        "mass_gap@32": 0.009254585619548499,
        "actual_k@32": 31.77908465903833,
        "coverage_target_hit@32": 0.3383167303024966,
        "coverage@64": 0.9129255210926157,
        "middle_mass@64": 0.16083251592332998,
        "mass_gap@64": 0.007536491845643076,
        "actual_k@64": 62.8605285704477,
        "coverage_target_hit@64": 0.4928816433268951,
        "coverage@128": 0.9453984842372719,
        "middle_mass@128": 0.19330547906798617,
        "mass_gap@128": 0.0055096001447408815,
        "actual_k@128": 122.5051033049833,
        "coverage_target_hit@128": 0.6651760259598363
      }
    }
  },
  "document_equal_fixed_below_rho": {
    "documents": 16,
    "methods": {
      "mean_pool": {
        "required_k": 126.20243197431913,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.2712851093787029,
        "coverage@4": 0.8186519849913769,
        "middle_mass@4": 0.035954975090609555,
        "mass_gap@4": 0.015946614539125607,
        "actual_k@4": 3.998966426760848,
        "coverage_target_hit@4": 0.08962913403790121,
        "coverage@8": 0.8375776006227038,
        "middle_mass@8": 0.05488059072193658,
        "mass_gap@8": 0.018418285640768337,
        "actual_k@8": 7.99007024542164,
        "coverage_target_hit@8": 0.14975901653228416,
        "coverage@16": 0.862617965837433,
        "middle_mass@16": 0.07992095593666587,
        "mass_gap@16": 0.019622357556339908,
        "actual_k@16": 15.920871317765874,
        "coverage_target_hit@16": 0.24207196409396925,
        "coverage@32": 0.8932137067149516,
        "middle_mass@32": 0.1105166968141845,
        "mass_gap@32": 0.019080836507960844,
        "actual_k@32": 31.52284764180139,
        "coverage_target_hit@32": 0.37943896967860835,
        "coverage@64": 0.9263276765104611,
        "middle_mass@64": 0.14363066660969387,
        "mass_gap@64": 0.01655613720006784,
        "actual_k@64": 61.49215548991088,
        "coverage_target_hit@64": 0.5528591200967105,
        "coverage@128": 0.956735471184617,
        "middle_mass@128": 0.17403846128384953,
        "mass_gap@128": 0.012365653684068634,
        "actual_k@128": 115.52820783747889,
        "coverage_target_hit@128": 0.7287148906212969
      },
      "quest_aabb": {
        "required_k": 125.13931865761269,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.2790388576946585,
        "coverage@4": 0.8128421038054001,
        "middle_mass@4": 0.030145093904632932,
        "mass_gap@4": 0.02175649572510223,
        "actual_k@4": 3.998966426760848,
        "coverage_target_hit@4": 0.07922410089471427,
        "coverage@8": 0.829672967743901,
        "middle_mass@8": 0.04697595784313383,
        "mass_gap@8": 0.02632291851957108,
        "actual_k@8": 7.99007024542164,
        "coverage_target_hit@8": 0.1310056346420828,
        "coverage@16": 0.8534622220155035,
        "middle_mass@16": 0.07076521211473615,
        "mass_gap@16": 0.028778101378269614,
        "actual_k@16": 15.920871317765874,
        "coverage_target_hit@16": 0.2134478240766917,
        "coverage@32": 0.8844981993018801,
        "middle_mass@32": 0.10180118940111274,
        "mass_gap@32": 0.027796343921032612,
        "actual_k@32": 31.52284764180139,
        "coverage_target_hit@32": 0.3464846783794739,
        "coverage@64": 0.9198698886002886,
        "middle_mass@64": 0.1371728786995214,
        "mass_gap@64": 0.023013925110240283,
        "actual_k@64": 61.49215548991088,
        "coverage_target_hit@64": 0.5308382685663859,
        "coverage@128": 0.9534132013344969,
        "middle_mass@128": 0.17071619143372982,
        "mass_gap@128": 0.015687923534188367,
        "actual_k@128": 115.52820783747889,
        "coverage_target_hit@128": 0.7209611423053416
      },
      "cobs_legacy_mean_logits": {
        "required_k": 100.1972130608081,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.22544026154414593,
        "coverage@4": 0.8203970918561552,
        "middle_mass@4": 0.037700081955387744,
        "mass_gap@4": 0.01420150767434742,
        "actual_k@4": 3.998966426760848,
        "coverage_target_hit@4": 0.10276904753652615,
        "coverage@8": 0.8411146709634894,
        "middle_mass@8": 0.058417661062722226,
        "mass_gap@8": 0.01488121529998268,
        "actual_k@8": 7.99007024542164,
        "coverage_target_hit@8": 0.16911585432809959,
        "coverage@16": 0.8677918613862502,
        "middle_mass@16": 0.08509485148548292,
        "mass_gap@16": 0.01444846200752283,
        "actual_k@16": 15.920871317765874,
        "coverage_target_hit@16": 0.26996262905072527,
        "coverage@32": 0.8994168942778755,
        "middle_mass@32": 0.11671988437710817,
        "mass_gap@32": 0.01287764894503716,
        "actual_k@32": 31.52284764180139,
        "coverage_target_hit@32": 0.41770873685605975,
        "coverage@64": 0.9327319325338208,
        "middle_mass@64": 0.15003492263305343,
        "mass_gap@64": 0.010151881176708264,
        "actual_k@64": 61.49215548991088,
        "coverage_target_hit@64": 0.6006352614619437,
        "coverage@128": 0.9623951865452104,
        "middle_mass@128": 0.179698176644443,
        "mass_gap@128": 0.006705938323475184,
        "actual_k@128": 115.52820783747889,
        "coverage_target_hit@128": 0.7745597384558544
      },
      "oracle": {
        "required_k": 83.26515579403387,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.18791590105085598,
        "coverage@4": 0.8345985995305024,
        "middle_mass@4": 0.051901589629735165,
        "mass_gap@4": 0.0,
        "actual_k@4": 3.998966426760848,
        "coverage_target_hit@4": 0.13796934465007443,
        "coverage@8": 0.8559958862634721,
        "middle_mass@8": 0.0732988763627049,
        "mass_gap@8": 0.0,
        "actual_k@8": 7.99007024542164,
        "coverage_target_hit@8": 0.21297758991880683,
        "coverage@16": 0.882240323393773,
        "middle_mass@16": 0.09954331349300578,
        "mass_gap@16": 0.0,
        "actual_k@16": 15.920871317765874,
        "coverage_target_hit@16": 0.32365960874429,
        "coverage@32": 0.9122945432229126,
        "middle_mass@32": 0.12959753332214533,
        "mass_gap@32": 0.0,
        "actual_k@32": 31.52284764180139,
        "coverage_target_hit@32": 0.4766188309427272,
        "coverage@64": 0.9428838137105288,
        "middle_mass@64": 0.16018680380976175,
        "mass_gap@64": 0.0,
        "actual_k@64": 61.49215548991088,
        "coverage_target_hit@64": 0.6539581898389181,
        "coverage@128": 0.9691011248686854,
        "middle_mass@128": 0.18640411496791817,
        "mass_gap@128": 0.0,
        "actual_k@128": 115.52820783747889,
        "coverage_target_hit@128": 0.8120840989491439
      },
      "learned": {
        "required_k": 91.19403033613844,
        "unreachable_rate": 0.0,
        "over_max_budget_rate": 0.20578134701093054,
        "coverage@4": 0.8268558377866327,
        "middle_mass@4": 0.044158827885865504,
        "mass_gap@4": 0.007742761743869657,
        "actual_k@4": 3.998966426760848,
        "coverage_target_hit@4": 0.11835852487973643,
        "coverage@8": 0.8479085911122841,
        "middle_mass@8": 0.06521158121151695,
        "mass_gap@8": 0.008087295151187964,
        "actual_k@8": 7.99007024542164,
        "coverage_target_hit@8": 0.18946095784574118,
        "coverage@16": 0.8745326271193045,
        "middle_mass@16": 0.09183561721853742,
        "mass_gap@16": 0.00770769627446835,
        "actual_k@16": 15.920871317765874,
        "coverage_target_hit@16": 0.2950027616262608,
        "coverage@32": 0.9056667873978945,
        "middle_mass@32": 0.12296977749712713,
        "mass_gap@32": 0.006627755825018204,
        "actual_k@32": 31.52284764180139,
        "coverage_target_hit@32": 0.44653378041187597,
        "coverage@64": 0.9378275112280838,
        "middle_mass@64": 0.15513050132731643,
        "mass_gap@64": 0.0050563024824453,
        "actual_k@64": 61.49215548991088,
        "coverage_target_hit@64": 0.6275802376266674,
        "coverage@128": 0.965779383151896,
        "middle_mass@128": 0.18308237325112894,
        "mass_gap@128": 0.0033217417167892105,
        "actual_k@128": 115.52820783747889,
        "coverage_target_hit@128": 0.7942186529890695
      }
    }
  },
  "full_validation_parity_max_abs": 0.0,
  "evaluation_seconds": 386.2700106482953,
  "document_scene_counts": [
    6160,
    6160,
    6160,
    6160,
    13328,
    13328,
    13440,
    13328,
    27664,
    27664,
    27664,
    27664,
    56336,
    56336,
    56336,
    56336
  ],
  "semantics": "Scene weighted averages all414064 scenes equally. Document equal first averages all scenes within each of16 documents, then averages16 document means equally. Same checkpoint, candidate blocks, labels, methods and budgets. Fixed-below-rho macro averages conditional document means."
}

### GPU release after all curve work

{"compute_apps_after_completion": "", "gpu_snapshot": "0, 0 MiB, 0 %\n1, 0 MiB, 0 %\n2, 0 MiB, 0 %\n3, 0 MiB, 1 %\n", "our_train_and_eval_processes_exited": true, "all_gpu_compute_contexts_released": true}

### Updated next-evaluation proposal after full validation

{
  "status": "proposal_only_not_started",
  "checkpoint": "Use preselected curve checkpoint; never select using HotpotQA or frozen test.",
  "hotpotqa": {
    "samples": 200,
    "prior_four_gpu_all28_wall_seconds": 3338.8885905742645,
    "prior_peak_reserved_mib": 18666,
    "new_learned_only_estimate": "approximately1hour on4GPUs, dependent on generated lengths",
    "comparison": "Can reuse verified completed dense200 and previous all28 predictions for quality under identical historical manual-greedy128-token protocol. Do not reuse timings as speed baseline. COBS and Quest historical runs incomplete; do not restart stopped jobs. Fresh full200COBS+Quest would add roughly2hours if each uses4-way sample sharding, not yet measured.",
    "protocol": "Freeze model/tokenizer/prompt/token hashes, fullSDPA prefill, block16,sink32tokens,local32blocks,middle32,all28 learned layers,no fallback. Historical protocol is not claimed identical to official cap32/generation-config defaults; any protocol change requires matched reruns.",
    "checkpoint": {
      "path": "/root/adablock/checkpoints/ranker_all28_curve_20261005/best.pt",
      "step": 50000,
      "sha256": "d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae"
    },
    "comparable_configuration": {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "model_snapshot": "a09a35458c702b33eeacc393d103063234e8bc28",
      "dtype": "bfloat16",
      "dataset": "cached official LongBench v1 hotpotqa, all200 source-order rows",
      "dataset_sha256": "a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f",
      "sample_assignment": "index modulo4,50samples per existing3090",
      "context_positions": 32768,
      "max_input_tokens": 32640,
      "max_new_tokens": 128,
      "prompt": "same historical QA prompt and Qwen chat wrapping; verify input_ids_sha256 against existing input manifest",
      "prefill": "fullSDPA",
      "decode": "same manual greedy raw-logit loop, no sampling or repetition penalty, same EOS handling as archived runner",
      "block_size": 16,
      "sink_tokens": 32,
      "local_blocks": 32,
      "middle_blocks_per_kv_head": 32,
      "learned_layers": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
      ],
      "fallback": "forbidden; verify all28layers actually call ranker",
      "seed": 42,
      "metrics": [
        "normalized QA tokenF1 maximum over references, macro across200",
        "normalized exact match",
        "truncation/EOS checks"
      ],
      "selection": "checkpoint locked before evaluating HotpotQA"
    },
    "cost_units": "Prior observed55.65minutes four-GPU wall, approximately3.71 allocated GPU-hours; estimate~1hour for new learned run, not a quoted currency cost. GPU price unknown."
  },
  "frozen_test": {
    "documents": 16,
    "existing_layers": [
      12,
      20,
      27
    ],
    "existing_scenes": 44448,
    "proposed_all28_scenes": 414848,
    "estimated_all28_bytes": 9583179312.0,
    "construction_estimate": "about5minutes4GPU extraction from prior independent-val cost model, plus startup and audits; estimate only",
    "evaluation_scene_rate_extrapolation_minutes": 6.252281555265105,
    "evaluation_caveat": "Updated from completed full independent validation374.43seconds on414064 scenes; frozen-test projection~6.25minutes, still an estimate.",
    "protocol": "Preserve exact frozen16source docs and original component/text isolation. Build new all28 directory, match old12/20/27 tensors, never overwrite oldtest. Evaluate selected checkpoint once with COBS/Quest/Oracle on identical scenes; no test-driven tuning."
  }
}

## All28 learning curve task completed

50k exact boundary; best50k by locked six-budget subset metric. Training and both complete-validation passes finished without retries. All four GPU contexts released. Final audit: results/ranker_all28_curve_20261005/FINAL_AUDIT.json. HotpotQA and all28 frozen-test proposals remain unexecuted. Server wall clock is about10m22s ahead of client UTC; durations use perf_counter.

{
  "both_weight_summary": {
    "scene_weighted": {
      "learned": {
        "coverage@4": 0.8330539627664691,
        "coverage@8": 0.8509714577085542,
        "coverage@16": 0.8735307810946569,
        "coverage@32": 0.9000911386756967,
        "coverage@64": 0.9284300582463054,
        "coverage@128": 0.9552515466180842,
        "required_k": 113.13429807952394
      },
      "cobs_legacy_mean_logits": {
        "coverage@4": 0.8258961488483693,
        "coverage@8": 0.843505465297186,
        "coverage@16": 0.8662465521276341,
        "coverage@32": 0.8933546930210355,
        "coverage@64": 0.9227663194962956,
        "coverage@128": 0.9511280492326348,
        "required_k": 124.82701224931411
      },
      "quest_aabb": {
        "coverage@4": 0.8214816629803338,
        "coverage@8": 0.8347565281606234,
        "coverage@16": 0.8534248785183917,
        "coverage@32": 0.8782511760195182,
        "coverage@64": 0.9079890213713708,
        "coverage@128": 0.9391836079709218,
        "required_k": 159.4623826268403
      },
      "oracle": {
        "coverage@4": 0.8415256541954944,
        "coverage@8": 0.8598364104383328,
        "coverage@16": 0.8821233310927369,
        "coverage@32": 0.9077442493786224,
        "coverage@64": 0.9346793165396837,
        "coverage@128": 0.9598316912831275,
        "required_k": 102.04973385756792
      }
    },
    "document_equal": {
      "learned": {
        "coverage@4": 0.8642785299291318,
        "coverage@8": 0.8806215817906811,
        "coverage@16": 0.9012075744685002,
        "coverage@32": 0.9252098723677582,
        "coverage@64": 0.9500373610910672,
        "coverage@128": 0.9718533150886257,
        "required_k": 72.93283258579056
      },
      "cobs_legacy_mean_logits": {
        "coverage@4": 0.8591103238895744,
        "coverage@8": 0.875202973028267,
        "coverage@16": 0.8958620628395466,
        "coverage@32": 0.9202794744889512,
        "coverage@64": 0.9460184898723972,
        "coverage@128": 0.969155968993642,
        "required_k": 80.21154846821075
      },
      "quest_aabb": {
        "coverage@4": 0.8534160956256819,
        "coverage@8": 0.8663520957971761,
        "coverage@16": 0.8845775939776632,
        "coverage@32": 0.9083375501762698,
        "coverage@64": 0.9355278115409797,
        "coverage@128": 0.9616595860007026,
        "required_k": 100.63628196840294
      },
      "oracle": {
        "coverage@4": 0.8704701307397342,
        "coverage@8": 0.8870970152436637,
        "coverage@16": 0.9074111185704383,
        "coverage@32": 0.9305966329836649,
        "coverage@64": 0.9542150322748655,
        "coverage@128": 0.9746656071199583,
        "required_k": 66.42513896480344
      }
    }
  },
  "full_layer_loss_ids": {
    "cobs_legacy_mean_logits": {
      "4": [],
      "8": [],
      "16": [
        20
      ],
      "32": [
        20
      ],
      "64": [
        20
      ],
      "128": []
    },
    "quest_aabb": {
      "4": [
        20
      ],
      "8": [],
      "16": [],
      "32": [],
      "64": [],
      "128": []
    }
  }
}


## Authorized frozen50k HotpotQA three-way full200 run
Independent directory: /root/adablock/results/hotpotqa_curve50k_threeway_20261005. Learned then COBS then Quest,4-way sample shards, cached Qwen revision a09a35458c702b33eeacc393d103063234e8bc28,32768 context/32640 input/128 greedy output,block16,sink32tokens,local32blocks,middle32,full SDPA prefill,all28 layers. Frozen best SHA d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae. Per-sample fsync journals and input-hash equality gates; no checkpoint tuning. Historical Full is quality-only baseline, not a speed comparison.


Completed learned full200: {"samples": 200, "F1": 57.65032410905455, "EM": 43.0, "input_truncation_rate": 0.0, "output_truncation_rate": 0.0, "output_limit_rate": 0.0, "sample_seconds": 11432.61634655483, "generated_tokens": 1517, "peak_reserved_mib": 18666.0, "wall_seconds": 3126.473019523546}


Completed cobs full200: {"samples": 200, "F1": 56.897860314860566, "EM": 42.0, "input_truncation_rate": 0.0, "output_truncation_rate": 0.005, "output_limit_rate": 0.005, "sample_seconds": 10878.892240364105, "generated_tokens": 1556, "peak_reserved_mib": 18666.0, "wall_seconds": 2865.8912391196936}


Completed quest full200: {"samples": 200, "F1": 56.44081815626111, "EM": 42.5, "input_truncation_rate": 0.0, "output_truncation_rate": 0.0, "output_limit_rate": 0.0, "sample_seconds": 11032.426402656361, "generated_tokens": 1668, "peak_reserved_mib": 18666.0, "wall_seconds": 2966.089666880667}


Completed all3 full200; paired comparisons: {"learned_minus_full": {"mean_F1_difference": -0.49570135618137384, "F1_wins": 4, "F1_ties": 186, "F1_losses": 10, "mean_EM_difference": 0.0}, "cobs_minus_full": {"mean_F1_difference": -1.2481651503753581, "F1_wins": 3, "F1_ties": 186, "F1_losses": 11, "mean_EM_difference": -1.0}, "quest_minus_full": {"mean_F1_difference": -1.7052073089748025, "F1_wins": 5, "F1_ties": 178, "F1_losses": 17, "mean_EM_difference": -0.5}, "learned_minus_cobs": {"mean_F1_difference": 0.7524637941939847, "F1_wins": 7, "F1_ties": 190, "F1_losses": 3, "mean_EM_difference": 1.0}, "learned_minus_quest": {"mean_F1_difference": 1.209505952793429, "F1_wins": 12, "F1_ties": 185, "F1_losses": 3, "mean_EM_difference": 0.5}}
GPU receipt: 0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 1 %

Answer F1/EM measure QA quality; attention coverage is a separate earlier evaluation. No paper-exact or speedup claim.


## Offline HotpotQA prediction diagnosis completed (2026-10-05)

Analysis directory: /root/adablock/results/hotpotqa_prediction_analysis_20261005. Report: [REPORT.md](../results/hotpotqa_prediction_analysis_20261005/REPORT.md); full26-case evidence review: [REVIEW_ALL_DIFFERENCES.md](../results/hotpotqa_prediction_analysis_20261005/REVIEW_ALL_DIFFERENCES.md). No new inference/training or scoring changes. All original source hashes and frozen50k checkpoint verified unchanged. Optional device query returned nvidia-smi exit6 (no devices found); this did not affect CPU-only analysis and no GPU/network policy was changed.

# HotpotQA 现有预测诊断（冻结50k模型）

本分析只读取已完成的五组各200题预测：新 Learned、COBS、Quest、历史 Full、旧三层训练但全28层推理模型。未新增推理、训练、预算实验或调参。原始主评分口径不变。

## 对齐与复算

200个唯一源ID、题序、输入token SHA256、参考答案均一致。旧记录没有独立sample_id字段时，使用完整selected_rows一致性、题序、输入哈希和参考答案共同确认映射。共享prompt、context上限32768、input上限32640、输出128、greedy、block16、sink32tokens、local32blocks、middle32、dtype和相关源码哈希均匹配。新旧Learned均全28层推理，训练checkpoint不同。逐题重算F1与归档分数的最大误差只在浮点舍入量级；EM按同一normalize_answer精确匹配。主结果保持：

| 方法 | F1 | EM |
|---|---:|---:|
| Full |58.146025|43.0|
| 新Learned |57.650324|43.0|
| COBS |56.897860|42.0|
| Quest |56.440818|42.5|
| 旧三层训练/全28层推理 |57.670305|45.0|

这些是固定200题上的回答指标，不是attention coverage，也不是主观事实正确率。

## 配对统计

按题有放回抽样，每次抽200题，同次抽样对所有方法和两指标使用相同索引；NumPy default_rng，seed=20261005，20,000次。区间为bootstrap差值均值的2.5%/97.5% percentile（linear quantile），单位均为百分点。

| 新Learned减去 | F1差值 [95%区间] | EM差值 [95%区间] |
|---|---|---|
| Full |−0.4957 [−1.5817, +0.5890]|0.0000 [−2.0000, +2.0000]|
| COBS |+0.7525 [+0.1233, +1.6571]|+1.0000 [0.0000, +2.5000]|
| Quest |+1.2095 [+0.0669, +2.5495]|+0.5000 [0.0000, +1.5000]|
| 旧模型 |−0.0200 [−1.2350, +1.1567]|−2.0000 [−4.0000, −0.5000]|

这是未做多重比较校正的逐项区间，依赖按题可交换抽样近似；不能据此断言推广到其他任务仍优于基线。F1对Full/COBS/Quest/旧模型的双侧精确符号检验p分别为0.179565、0.343750、0.035156、1.000000；对应EM为1、0.5、1、0.125。8项辅助检验做Holm校正后没有p<0.05的结果。符号检验只看方向，bootstrap均值还看幅度，两者问题不同。特别是旧模型EM的percentile区间虽不跨0，但只有4个不一致题且全部为新模型失分，符号检验p=0.125，不能夸大离散小样本证据。两种方法均不校正参考答案局限。

## 差异集合与重叠

以下编号全部从0开始；完整ID、原题、参考答案、五组完整预测和证据位置见逐题文件。

- 新Learned vs COBS，不同F1的10题：14,48,50,57,78,95,141,180,188,194。
- 新Learned vs Quest，不同F1的15题：14,41,48,54,57,95,101,110,117,122,141,144,171,188,194。
- 新Learned对Full失分10题：14,28,48,57,72,95,117,122,148,180。
- 前两集合交集7题：14,48,57,95,141,188,194。
- COBS集合与Full失分集合交集5题：14,48,57,95,180。
- Quest集合与Full失分集合交集6题：14,48,57,95,117,122。
- 三者交集4题：14,48,57,95；三个集合去重共21题，不是35题。
- Full全部不同F1共14题（含4个新Learned得分更高的题）；新旧不同F1共17题。纳入这四个比较后去重共26题。
- 新旧17题与上述重点21题重合12题：14,28,41,48,57,95,122,148,171,180,188,194。
- 字面预测不同题数分别为Full23、COBS18、Quest31、旧模型25，不能等同于F1差异数。全部200题五组完整输出另存all_200_review.json；26题重点证据审阅见REVIEW_ALL_DIFFERENCES.md。

## 可以支持的解释

1. **COBS差距有很强的答案边界成分。** 第50题双方都答Pope John X，但COBS多写整句；第78题coaxial与coaxial cable语义相同。这两题对200题均值差贡献0.333333+0.166667=0.500000分，占总优势0.752464的约66.45%。再加第57题双方均包含Floyd Casey Stadium但冗余词不同，以及第180题Qin/Qin Empire的名称完整度，四题合计0.633333分（约84.17%的净优势）。这是分数代数贡献，不能称为84%的错误因果，也不能把重叠标签视为独立事件。

2. **确有内容差异，但F1不可靠地排序事实质量。** 第117题：Learned/COBS/旧模型答Susanne Bier；原文Passage3列出The Great Mouse Detective的四位导演，Susanne Bier属于另一段The Night Manager。Full给齐四人，Quest给两人。第110题Quest答原先的Representative，Learned给历史语境中的Secretary of the Interior。第141题Learned给O，COBS/Quest给Mystère；原文O自1998年驻演，后者起始更早，属于时间约束/候选混淆的可支持解释，而非已经证实的选块原因。

3. **较高分可能伴随错误结论。** 第48题Learned说Tongshanjiabu更高，同时列7207与7893米；Full也作相同矛盾结论。旧模型开头错误但最终改判Himalchuli更高，F1却比新模型低。第54题所有答案都未给Australian Supercars Championship，Quest仅凭Championship拿到部分分；题干还把原文“Pagenaud替代de Silvestro”的关系反过来。不能把此处F1升降解释为事实能力升降。

4. **参考表达和上下文存在局限。** 第28题Band-e-Amir与Band-e Amir是同一公园；当前normalizer删除连字符而不补空格，导致token合并、F1从100降至57.14。第31题省略LLC，第101题省略头衔/地域，第144题35与35 people，均影响词面评分。第194题原文明确over100million，参考140million未在上下文找到；前者是与140million相容的下界，不能定性为事实错误，但不满足参考精度。第14题参考的Falkland列表在提供上下文中缺失，且所有输出都未按问题给地区；第188题问题含混，参考是Senate任命，而原文另有该报民调排名，所有模型否认提及也不符合原文。第176题问film但材料谈TV series，各模型未给参考中的具体副标题。上述不改主分，仅做诊断。

5. **新模型没有比旧模型改善QA均值的具体表现。** 183/200题F1相同，9题提高、8题下降。新模型在第18题丢失正确专辑Talk That Talk，第20题把节目块答成Cartoon Cartoons；在第70题修正1706为1939但漏月日。第148题新旧都未答出地区，增加Italy只改善字面分；第48题新分更高却保留矛盾结论。旧模型EM胜出的四题为20、28、31、180：一题明显答错类型，三题主要是连字符/法律后缀/名称完整度。净F1差−0.01998及宽区间不支持新模型QA优于旧模型。

覆盖指标的提高不保证QA提高：训练/验证关注teacher attention mass和块排序，QA是自回归生成、实体与关系链、格式和参考词面评分；题分大多未变化，少数边界和事实得失相互抵消。可以提出目标不完全一致、序列分布及生成轨迹差异等候选解释，但本分析没有逐步候选块ID、相同轨迹反事实或逐token证据attention对照。现有token trace和选块哈希不足以确定“某证据块被漏选”这一因果，更不能证明哪一训练设计导致上述答案。

## 唯一截断题敏感性诊断

第95题COBS在128-token上限截断；其余方法未截断。原文Passage1开头明确43,000 demonstrators，但所有模型都未给43,000并声称数字未提供。Learned F1=3.921569，COBS=2.150538；这不是任何一方正确回答人数。

该题贡献Learned−COBS总均值差0.008855分，约占0.752464的1.18%，不是优势主要来源。无法从保存结果证明延长生成会补出正确数字。

以下仅作另列诊断：对所有方法配对删除同一题95，样本统一变199，未修改主结果：
- Learned−COBS F1 +0.747345，EM +1.005025。
- Learned−Quest F1 +1.214489，EM +0.502513。
- Learned−Full F1 −0.492129，EM 0。
- Learned−旧模型 F1 −0.015859，EM −2.010050。

## 证据与复现

- analyze_predictions.py：读取原始文件、对齐/评分断言、固定seed20,000次bootstrap、符号检验、集合与敏感性分析及逐题证据导出。
- annotations.json：人工审慎审阅结论与证据锚点；分类可重叠，不是盲审金标准或新的正确率评分。
- REVIEW_ALL_DIFFERENCES.md / review_cases.json：26个不同分题的原问题、参考答案、五组完整预测、判断及证据。
- contexts/NNN.txt：对应题完整原始上下文；证据使用Passage标题、1起始行号及0起始Unicode字符区间[start,end)。
- all_200_review.json：所有200题五组完整预测和重算分数，包括同分异文；原始context路径/行索引可回溯。
- per_question_scores.json / per_question_contributions.csv：逐题分数、每个比较对总均值的贡献。
- paired_statistics.json / difference_sets.json / truncation_sensitivity.json / alignment_audit.json / source_manifest.json：完整统计与来源校验。
- Reproduce: /root/miniconda3/envs/ada/bin/python /root/adablock/results/hotpotqa_prediction_analysis_20261005/analyze_predictions.py
- 然后运行同目录finalize_analysis.py重建本报告及全量审阅索引。脚本仅写本分析目录，不运行模型、不改原始预测。

这200题现已用于结果诊断，后续据此修改系统会使它们成为开发数据。新的确认性结论需要未使用任务/数据和预先固定协议；本次不据这些诊断调参或启动新实验。


## Full baseline protocol audit completed, no inference (2026-10-05)

Detailed report: [REPORT.md](../results/full_baseline_protocol_audit_20261005/REPORT.md). New independent audit directory: /root/adablock/results/full_baseline_protocol_audit_20261005.

# Full HotpotQA baseline protocol audit — no new inference

## Conclusion

The saved Full58.146025 result is a valid internal baseline for the project's locked protocol. No misalignment, incorrect F1 calculation, unintended sparse selection, or input truncation was found. It is **not yet a reproduction of the paper's55.84 Full baseline**: the paper's exact implementation settings and per-question outputs are unavailable, and the local run has known generation-protocol differences. No cause for the remaining2.140650-point gap has been established.

Paper-side findings below were supplied by the parent research task, not independently inferred from local code: final ICLR2026 PDF p16/Table2 has Full55.84 and LouisKV51252.14; accuracy-specific prompt/chat/max-new/revision/generation/truncation/scorer details are not disclosed. FA2 in the efficiency section and FP16 in memory analysis must not be treated as proof of the accuracy configuration. No verified official implementation repository was found. Changes in wording between preprint and final are not evidence of code changes.
Paper: https://proceedings.iclr.cc/paper_files/paper/2026/file/6b241c515433caae3051266668d808b7-Paper-Conference.pdf

## Frozen local evidence

- Model and tokenizer cache: Qwen/Qwen2.5-7B-Instruct snapshot a09a35458c702b33eeacc393d103063234e8bc28.
- Current environment: transformers4.55.4, torch2.5.1+cu121. Historical config/source hashes match current audited implementation; original Full did not record a comprehensive environment/weight-byte manifest, so do not claim retroactive proof of every weight byte or CUDA kernel.
- Cached model config:28layers,28queryheads,4KVheads,BF16,32768positions,RoPE theta1e6,no rope_scaling,no sliding window. Tokenizer metadata says131072 model_max_length, but actual inputs max17684, below32768. No long-context RoPE extension was active.
- Cached generation_config:do_sample=true,temperature0.7,top_p0.8,top_k20,repetition_penalty1.05,EOS[151645,151643],pad151643. These defaults are NOT applied by the manual decoder.
- Model weights inventory and resolved cache blob names are saved, but this audit did not reread/hash roughly15GB of weights. Config/tokenizer JSON hashes and current revision are captured. Historical cap32 audit already records this tokenizer revision and hashes.

## Input and data

Local data/longbench/hotpotqa.jsonl SHA256 a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f exactly equals the member data/hotpotqa.jsonl in cached zai-org/LongBench data.zip snapshot5e628be450b7e67fb7ae6e201bd6d8f7056f7672. Exactly200 unique IDs, same source order, contexts/questions/references as archived selected_rows. This confirms our source version; paper-side IDs/data hash remain unknown.

Historical runner hardcodes the official LongBench HotpotQA prompt exactly. The current generic utils/longbench_eval.py prompt is modified, but this generic prompt was **not used** by this historical run. Accidentally reusing the generic helper in calibration would change the protocol.

Qwen2TokenizerFast apply_chat_template receives one user message, tokenize=False,add_generation_prompt=True. The template inserts its default system message: You are Qwen, created by Alibaba Cloud. You are a helpful assistant. The user prompt ends in Answer:, followed by im_end/newline and im_start assistant/newline. No extra hand-written system message; no tools; add_bos_token=false; no double BOS. The final answer/reference fields are not interpolated into the input template.

Every rebuilt token hash matches archived Full input_manifest. Input lengths1927–17684,mean13475.455. All200 are untruncated, so32640 vs nearby input caps and pre-chat vs post-chat truncation order do not change this run (provided cap remains above all actual inputs). They could matter for an unknown lower paper cap; do not assume the paper used ours.

Official LongBench v1 upstream commit2e00731f8d0bff23dc4325161044d0ed8af94c1e was saved. Its original model list/chat adapter does NOT include Qwen2.5. Therefore a Qwen adaptation with official task prompt/scorer can be carefully specified, but cannot be called an exact upstream Qwen recipe or paper reproduction.

## Generation: proven differences and observed effects

| Aspect | Historical Full | HF generate / official-task reference | Assessment |
|---|---|---|---|
| Next token | raw logits argmax | HF generate(do_sample=False,num_beams=1) | Must test numerical/path parity under explicitly identical settings |
| Repetition penalty | effectively1.0 | cached config1.05 survives do_sample=False unless overridden | Confirmed config difference; direction/score effect unknown |
| EOS | only151645 | cached[151645,151643] | No151643 in any saved Full output; no effect on existing raw-greedy trajectory |
| First token EOS check | missing before step1 | generate normally checks after each token | Defect exists, but zero affected cases among200 |
| Maximum new tokens |128 | official HotpotQA32 | Exact saved-prefix rescore57.980650 |
| Sampling temperature/top_p/top_k | raw argmax at temperature0 | official generate temperature1 with do_sample=False | Warpers inactive; these values alone do not introduce sampling |
| Padding | batch1/no padding; mask all ones | explicit pad151643 in proposal | No padding-related truncation found |

HF repetition penalty acts on tokens occurring in the **prompt and prior generated tokens**, including the first generated step; it is not merely a late-output repeat suppressor. The installed processor divides positive logits and multiplies negative logits by1.05 for seen IDs. Saved traces have only top1 logits: absent competitor logits prevent offline reconstruction of a changed winner and its future trajectory.

Official source: fixed upstream pred.py uses generate(max_new_tokens=task_limit,num_beams=1,do_sample=False,temperature=1.0); it does not explicitly reset repetition_penalty or EOS for HotpotQA. With this Qwen cache, directly transplanting that call retains penalty1.05 and dualEOS. This is a concrete candidate configuration for calibration, **not proof the paper used it**.

Saved token-prefix cap32:58.146025 ->57.980650, delta−0.165376. Eight predictions change text, six change F1. The original gap to55.84 is2.306025; cap32 leaves2.140650, so generation cap alone does not explain the gap. This offline result is exact for the existing raw-greedy trajectory; it does not substitute for HF or penalty-changed inference.

All200 token traces exactly decode back to saved predictions; EOS151645 appears only at the last token, no151643, no first-token EOS. Whitespace strip does not affect English normalized F1. No hidden answer extraction/first-line shortcut was used.

## Dense attention, cache, positions, dtype

Production Full records all have zero selector/learned calls. Preflight temporarily installs sparse attention for parity checks, but run_layer_head_qk_decode restores native attention in finally via controller.uninstall; the production Full method invokes only run_full_decode_trace. Loading an unrelated ranker into args for common preflight does not feed it into the Full branch.

The manual decoder uses model.model for all28 native layers, then the same lm_head as Qwen2ForCausalLM.forward. Prefill is full causal SDPA. Decode passes one token with the entire unpruned KV cache, all-ones mask length history+1, explicit position_id=history. Legacy-cache conversion preserves all layer K/V tensors and is converted back into DynamicCache. Installed Qwen infers cache_position from cached length, applies RoPE to Q/K, and appends K/V at native layer indices. Static review found no future-label access, mask/position offset mismatch or sparse selection in Full.

Actual model loading explicitly sets BF16 and sdpa; generic parse_args defaultfloat16 is overridden by the explicit loader and does not describe this run. HF's SDPA dispatcher may select different CUDA kernels; exact dispatched kernel/TF32/determinism settings were not archived. No claim of FA2 identity or FP16 identity with paper is justified. Static review and small saved all-candidate parity probes do not prove full200 numerical equivalence to native generate. An A test is needed.

## Official scoring

Executed only reviewed pure normalize_answer/f1_score/qa_f1_score functions from the pinned upstream metrics.py on all200 predictions. Every per-question F1 agrees with saved scores (floating rounding only); macro mean58.146025, upstream rounded output58.15. Reference maximum, punctuation/articles/case/whitespace normalization match. Official eval.py does not apply first-line-only processing to HotpotQA. Dataset-length buckets apply to LongBench-E mode, not this normal200 macro run. Scoring alone does not explain55.84.

## Minimal calibration proposal — prepared, NOT run

Do not choose settings by closeness to55.84. Freeze the official prompt, exact existing200input hashes, model/tokenizer revision, BF16/SDPA, batch1, seed42,context32768,output32 and scorer first.

A. Native HF generate, explicit penalty1.0,EOS151645,pad151643,greedy,num_beams1,no other penalties. First8 fixed source-order questions as a technical pilot, then resume remaining192 without regenerating the pilot. Compare every generated token with archived Full trace[:32]. Stop on first mismatch and diagnose cache/position/numerics before making accuracy claims. If all match, A score must be57.980650.

B. Same200 inputs and settings, change ONLY repetition_penalty to1.05; compare paired F1/EM/tokens against A. Do not jointly change chat, dtype,backend,EOS or prompt.

C. DualEOS[151645,151643] is a separate stopping-rule check. Existing A trajectory has no151643. Once B token IDs exist, clipping each B trace at the first dualEOS can diagnose that stopping rule offline, because no length-dependent/forced-EOS processors are configured. This does not require an additional model run for the fixed B trajectory. If no151643 appears, the EOS difference again has no effect. A future protocol claiming defaults must record the full resolved generation_config.

prepare_hf_full_calibration.py is saved, syntax checked only, and defaults to print a plan without loading model weights. --execute is required for inference. It uses only cached model files; no downloads. A/B JSON configs differ only in repetition_penalty. Sampling-only top_k is inactive; runner sets it to HF's conventional50 to avoid an unused-setting warning. The script uses fresh output directories, fsync per-question records, resume identity checks and input-hash gates. No run was launched.

Measured historical Full200 dense sample time746.52seconds (~12m27s on one3090), peak reserved18628MiB; loading/startup and differing HF execution affect walltime. Proposed A200+B200 is roughly20–40minutes on one available3090, about0.33–0.67 GPU-hours, **estimate only**, unlike ~3h sparse CPU-digest evaluation. Pilot results should update runtime estimate. Native generate backend differences or limited CPU/GPU availability can change it. Current device query may report no devices after released allocation; do not provision/start a machine without separate authorization.

Any further paper matching of template/system,lower input cap,data revision,FP16,FA2 orlibraryversion requires author evidence or a predeclared sensitivity study, not a score-target search. Even A/B results would measure local configuration effects, not establish paper settings or explain its exact55.84 without paper predictions/protocol.

## Deliverables and evidence

- AUDIT.json: source/data/config/token/hash and static dense-path audit.
- per_sample_audit.json: all200 IDs/input hashes, generation length/EOS/call counts and official scores.
- cap32_recheck.json: exact32-token rescore and changed cases.
- template_audit.json and cached_*_config.json: actual system/prefix/tokenizer/default settings.
- implementation_snapshot/: exact reviewed local runner, scorer/controller and installed Transformers implementations.
- upstream_longbench/: five small official files, fixed commit and saved UTF8 hashes.
- proposed_hf_A_generation_config.json / proposed_hf_B_generation_config.json and prepare_hf_full_calibration.py: unexecuted minimal A/B plan.
- audit_full.py: CPU-only reproducible tokenizer/scoring/config audit; never instantiates model weights.

Upstream sources:
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json


## Authorized A-only native HF Full calibration started
Directory: /root/adablock/results/full_hf_A_calibration_20261005. GPU0 only; no B/16-block/training. Same200token hashes, cached Qwen revision a09a35458c702b33eeacc393d103063234e8bc28,BF16/SDPA,greedy,penalty1.0,EOS151645,maxnew32. First8 fixed questions gated by token parity; stop at any mismatch. Per-sample fsync resume journal; no source result overwrite.


A-only calibration ended: {"stage": "failed", "returncode": 1, "worker_pid": 402, "wall_seconds": 25.029928071424365, "pilot8_passed": false, "source_hashes_unchanged": true}. GPU receipt: 0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 1 %



## Authorized A-only native HF Full calibration started
Directory: /root/adablock/results/full_hf_A_calibration_20261005_v2. GPU0 only; no B/16-block/training. Same200token hashes, cached Qwen revision a09a35458c702b33eeacc393d103063234e8bc28,BF16/SDPA,greedy,penalty1.0,EOS151645,maxnew32. First8 fixed questions gated by token parity; stop at any mismatch. Per-sample fsync resume journal; no source result overwrite.


A-only pilot8 token parity passed; continuing existing process to200.


A-only calibration ended: {"stage": "completed", "returncode": 0, "worker_pid": 726, "wall_seconds": 720.9321448560804, "pilot8_passed": true, "summary": {"all_token_parity": true, "input_truncated": 0, "output_truncated": 8, "eos": 192, "contains_endoftext": 0, "sample_seconds": 693.102415971458, "peak_reserved_mib": 22962.0, "samples": 200, "F1": 57.9806499087555, "EM": 43.0, "generation_config": {"max_length": 20, "max_new_tokens": 32, "min_length": 0, "min_new_tokens": null, "early_stopping": false, "max_time": null, "stop_strings": null, "do_sample": false, "num_beams": 1, "num_beam_groups": 1, "penalty_alpha": null, "dola_layers": null, "use_cache": true, "cache_implementation": null, "cache_config": null, "return_legacy_cache": null, "prefill_chunk_size": null, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "min_p": null, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "diversity_penalty": 0.0, "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "bad_words_ids": null, "force_words_ids": null, "renormalize_logits": false, "constraints": null, "forced_bos_token_id": null, "forced_eos_token_id": null, "remove_invalid_values": false, "exponential_decay_length_penalty": null, "suppress_tokens": null, "begin_suppress_tokens": null, "sequence_bias": null, "token_healing": false, "guidance_scale": null, "low_memory": null, "watermarking_config": null, "num_return_sequences": 1, "output_attentions": false, "output_hidden_states": false, "output_scores": false, "output_logits": null, "return_dict_in_generate": false, "pad_token_id": 151643, "bos_token_id": 151643, "eos_token_id": 151645, "encoder_no_repeat_ngram_size": 0, "decoder_start_token_id": null, "is_assistant": false, "num_assistant_tokens": 20, "num_assistant_tokens_schedule": "constant", "assistant_confidence_threshold": 0.4, "prompt_lookup_num_tokens": null, "max_matching_ngram_size": null, "assistant_early_exit": null, "assistant_lookbehind": 10, "target_lookbehind": 10, "disable_compile": true, "_from_model_config": false, "transformers_version": "4.55.4"}}, "source_hashes_unchanged": true}. GPU receipt: 0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 1 %



## Authorized B-only Full calibration
Directory: /root/adablock/results/full_hf_B_calibration_20261005. Exactly A generation settings and inputs except repetition_penalty1.05; GPU0 only; no other experiments. Actual processor must be RepetitionPenaltyLogitsProcessor1.05 only. Config/source hashes and per-question fsync journals saved.


B-only calibration ended: {"stage": "completed", "returncode": 0, "wall_seconds": 725.8984802439809, "worker_pid": 1301, "analysis_returncode": 0, "comparison": {"samples": 200, "A_F1": 57.9806499087555, "B_F1": 57.65029276589836, "A_EM": 43.0, "B_EM": 43.0, "B_minus_A_F1": -0.330357142857143, "B_minus_A_EM": 0.0, "F1_paired": {"wins": 2, "ties": 195, "losses": 3}, "EM_paired": {"wins": 1, "ties": 198, "losses": 1}, "token_changed": 11, "prediction_changed": 11, "F1_changed": 5, "EM_changed": 2, "first_token_divergences": 9, "eos_count": 192, "output_truncated_indices": [48, 59, 63, 72, 95, 117, 188, 189], "endoftext_indices": [], "no_input_truncation": true, "official_scores_recomputed": true, "only_expected_processor": true, "paper_full": 55.84, "B_minus_paper": 1.8102927658983532, "A_minus_paper": 2.140649908755499, "dual_eos_diagnostic": {"new_inference": false, "F1": 57.65029276589836, "changed_indices": []}}, "run_summary": {"input_truncated": 0, "output_truncated": 8, "eos": 192, "contains_endoftext": 0, "sample_seconds": 692.2911993917078, "peak_reserved_mib": 22962.0, "samples": 200, "F1": 57.65029276589836, "EM": 43.0, "generation_config": {"max_length": 20, "max_new_tokens": 32, "min_length": 0, "min_new_tokens": null, "early_stopping": false, "max_time": null, "stop_strings": null, "do_sample": false, "num_beams": 1, "num_beam_groups": 1, "penalty_alpha": null, "dola_layers": null, "use_cache": true, "cache_implementation": null, "cache_config": null, "return_legacy_cache": null, "prefill_chunk_size": null, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "min_p": null, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "diversity_penalty": 0.0, "repetition_penalty": 1.05, "encoder_repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "bad_words_ids": null, "force_words_ids": null, "renormalize_logits": false, "constraints": null, "forced_bos_token_id": null, "forced_eos_token_id": null, "remove_invalid_values": false, "exponential_decay_length_penalty": null, "suppress_tokens": null, "begin_suppress_tokens": null, "sequence_bias": null, "token_healing": false, "guidance_scale": null, "low_memory": null, "watermarking_config": null, "num_return_sequences": 1, "output_attentions": false, "output_hidden_states": false, "output_scores": false, "output_logits": null, "return_dict_in_generate": false, "pad_token_id": 151643, "bos_token_id": 151643, "eos_token_id": 151645, "encoder_no_repeat_ngram_size": 0, "decoder_start_token_id": null, "is_assistant": false, "num_assistant_tokens": 20, "num_assistant_tokens_schedule": "constant", "assistant_confidence_threshold": 0.4, "prompt_lookup_num_tokens": null, "max_matching_ngram_size": null, "assistant_early_exit": null, "assistant_lookbehind": 10, "target_lookbehind": 10, "disable_compile": true, "_from_model_config": false, "transformers_version": "4.55.4"}}, "sources_unchanged": true}. GPU receipt: 0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 1 %

A/B isolates local penalty effect; no claim that authors used this setting and no score-target configuration selection.


## FASA official Full reproduction preflight

Report: [PREFLIGHT_REPORT.md](../results/fasa_full_reproduction_20261005/PREFLIGHT_REPORT.md).

# FASA official Full reproduction preflight — blocked, no inference

Pinned official commit:96c00c46db12ccee56104175470aa29579d66c58.
Eight reviewed text files and hashes are preserved in official_source/ and preflight_audit.json. No third-party code was executed; Python AST parsing only. No dependencies were installed and ada was not changed.

SSH recovered after user reported power-on. Repeated nvidia-smi exit6: No devices were found. Device nodes only nvidiactl/nvidia-uvm/nvidia-uvm-tools; no numbered GPU nodes. GPU allocation is not visible to this session, so no smoke or200run can start. Do not change driver/network/shutdown policy.

Current ada:Python3.10.20,Torch2.5.1+cu121,CXX11_ABI=False,Transformers4.55.4,tokenizers0.21.4,accelerate1.13.0,datasets5.0.0. Missing flash-attn,jieba,rouge,fuzzywuzzy; no cached flash wheel found in checked project/pip/autodl-tmp paths. Only ada env exists.

Official requirements specify Torch2.5.1,Transformers==4.43 (=4.43.0),datasets3.6.0,flash-attn>=2.7.4 plus metrics. README's suggested FA2 wheel is2.7.4.post1 cu12torch2.5cxx11abiFALSE cp310 linux_x86_64. Official release asset metadata reports187,797,312bytes (~179.1MiB), inspected only, not downloaded:
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

Minimal proposed environment: separate project venv with system-site-packages reusing existing Torch, install isolated transformers4.43.0 and compatible tokenizers plus only Full-required metrics and matching FA2 wheel; pin datasets3.6.0 if reproducing declared stack, though local JSONL Full entry does not call datasets. Do not bulk-install unrelated reasoning packages or downgrade ada. Review ABI/import compatibility before any forward. Do not silently fall back to SDPA. Any large wheel should be fetched directly on server with approved network access, not through user's local proxy. If no binary wheel works, stop and report build/toolchain cost instead of compiling blindly. Dependency preparation time/availability unknown; no model download is needed.

Audited official Full mode is compression_methods=base. It skips selector activation but imports sparse modules at module top-level, so an isolated Full-only import guard is likely needed to avoid irrelevant incompatible dependencies; this is an explicit compatibility patch, not a parameter change.
- Loader BF16 + flash_attention_2; trust_remote_code=True. Cached standard Qwen config has no custom auto_map and model revision can be pinned to existing a09a35458c702b33eeacc393d103063234e8bc28.
- HotpotQA prompt exactly equals A/B and original Full. QA normalize_answer,f1_score,qa_f1_score ASTs exactly equal pinned upstream official functions.
- Same one-user-message Qwen chat template with generation prefix; actual input hashes still need runtime tokenizer validation in target env.
- Qwen max input127500 before chat; our existing data max17684 after chat, so under either cap. No long-input/RoPE equivalence claim beyond these200 inputs.
- maxnew32,greedy,num_beams1,min_length=context_length+1. This adds first-step EOS suppression unlike A/B.
- Inherits checkpoint generation_config: candidate penalty1.05 and EOS[151645,151643] must be captured at runtime, not guessed from paper.
- tokenizer.pad_token_id is changed to tokenizer.eos_token_id. This does NOT by itself change model.generation_config.pad_token_id; record both.
- CUDNN deterministic=True,benchmark=False in official seed function; preserve in reproduction.
- Main passes evaluate_num=-1 regardless of CLI, so --evaluate_num8 is ineffective; patch or reviewed wrapper needed for8-case smoke.
- eval_longbench unconditionally removes its output file; replace with new-directory-only and resume-safe logic, preserving all old results.
- Official entry expects ../longbench/hotpotqa.jsonl, not bundled200-ID data; reuse our independently verified cached data snapshot and disclose that author data hash is unavailable.

Prepared patch plan, not applied/executed:
1. Keep immutable official_source, create separate patched source + unified diff.
2. Restrict mode tobase; avoid unused sparse module imports without modifying Full loader/generate/scorer.
3. Bind local model-cache/data paths, block downloads and unreviewed remote custom code.
4. Honor first8 smoke and continue same200 without replay; add fsync journal, input/token traces, generation processor and native-attention audits.
5. Preserve official BF16FA2,chat,min_length and inherited resolved generation settings. No score-target tuning.
6. Compare200 IDs/references/input hashes, F1/EM/EOS/truncation and paired A/B differences; record exact environment deviations if any.

Once GPU/FA2 are available, first8 smoke likely1–2minutes including warm-up and Full200 roughly10–20minutes on one3090, based only on prior denseA/B12-minute runs; FA2 timing is unmeasured. Installation/download work is additional. No paid resources, sparse evaluations, ranker integration or training authorized here.


# FASA official Full attempt: blocked by single-GPU memory

The pinned official Full code at commit 96c00c46db12ccee56104175470aa29579d66c58 was launched after all dependencies and GPU checks passed. The first example failed before producing a prediction: Transformers 4.43.0 Qwen2ForCausalLM computes logits for every input position and converts all logits to FP32; allocation of 6.68 GiB failed with 4.46 GiB free on a 24 GiB RTX3090. This is an actual memory failure, not missing GPU or missing FA2.

Independent environment: /root/adablock/.venvs/fasa_full_20261005. Torch 2.5.1+cu121, Transformers 4.43.0, FA2 2.7.4.post1, tokenizers0.19.1, datasets3.6.0. Official FA2 wheel was downloaded directly from GitHub on the server, resumed after the GPU-recovery restart. Original ada package lists are unchanged.

All 200 ordered IDs and input hashes match previous A/B; max input17684 tokens, no truncation. Model audit confirms 28 native Qwen2FlashAttention2 layers. Official prediction, prompt, scoring, and seed functions have identical ASTs. Only unused sparse imports and local-cache/native-model loading options differ in the saved patch. Original single-GPU loader placement was retained for this attempt.

No sample predictions, no smoke8 pass, no Full200 score. Existing A/B and prior experiments untouched. Run logs, generation/native-layer audit, source patch, dependency provenance, and failed status are retained here.

Recommended next action: a separate, explicitly documented model-placement-only attempt using the already available GPUs to distribute weights and the large logits allocation. Preserve official 4.43.0 forward/generation semantics, model revision, input tokens and scoring. Do not present last-token-only logits modifications or upgraded Transformers as an exact official-code run. Multi-GPU execution would require a distinct run directory and should not be used for direct speed claims against single-GPU A/B.

Artifacts: /root/adablock/results/fasa_full_reproduction_20261005


# FASA official Full reproduction

{
  "summary": {
    "samples": 200,
    "F1": 58.18362609923169,
    "official_rounded_F1": 58.18,
    "EM": 43.5,
    "input_truncated": 0,
    "output_truncated": 8,
    "ended_eos": 192,
    "sample_seconds": 797.8417475968599,
    "loop_wall_seconds": 804.9895437713712,
    "peak_reserved_mib": {
      "0": 22496.0,
      "1": 11622.0,
      "2": 12566.0,
      "3": 12566.0
    }
  },
  "comparison": {
    "A": {
      "old_F1": 57.9806499087555,
      "delta_F1": 0.20297619047618554,
      "token_changed": 15,
      "F1_wins": 4,
      "F1_ties": 194,
      "F1_losses": 2
    },
    "B": {
      "old_F1": 57.65029276589836,
      "delta_F1": 0.5333333333333314,
      "token_changed": 5,
      "F1_wins": 2,
      "F1_ties": 198,
      "F1_losses": 0
    },
    "paper": {
      "reported_F1": 55.9,
      "delta_F1": 2.2836260992316895,
      "limitation": "Paper data ordering/hash and exact model revision are not published in inspected repository; this is pinned official-code Full reproduction on verified cached LongBench 200."
    }
  }
}

Official commit 96c00c46db12ccee56104175470aa29579d66c58. Qwen2.5-7B-Instruct cached revision a09a35458c702b33eeacc393d103063234e8bc28. Native FA2 Full, BF16, transformers 4.43.0, torch 2.5.1+cu121, flash-attn 2.7.4.post1; seed42, greedy, max_new_tokens32, inherited repetition_penalty1.05, dualEOS, min_length=input_length+1. All 200 input hashes match A/B; no input truncation. Official prediction function AST unchanged. Adaptations: remove unused sparse imports; local cache only and native trust_remote_code=False; per-row journal/resume and smoke8 checkpoint; fixed four-GPU device map with full prefill logits retained, placement_plan.json records budgets. No FASA sparse or training executed. Timings include four-GPU placement and are not comparable to single-GPU A/B as a speed result. Paper exact dataset/model provenance unavailable; score difference cannot identify one cause.

Artifacts: /root/adablock/results/fasa_full_reproduction_20261005_multigpu


## Completed-result audit supplement

200/200 complete. F1 58.18362609923169 (official scorer rounded58.18), EM43.5. Versus paper Full55.9: +2.28362609923169 percentage points. A: F157.9806499087555/EM43; deltaF1+0.20297619047618554, 15 token/prediction changes, 11 first-token changes, F1 4wins/194ties/2losses, EM3 changed cases (net+1 correct). B: F157.65029276589836/EM43; deltaF1+0.5333333333333314, 5 token/prediction changes, 4 first-token changes, F1 2wins/198ties/0losses, EM only index0 improved.

Zero-based changed indices versus B: 0,50,67,149,189. Index0 changes Gates v. Collier to Miller v. California (F1 33.3333 to100); index50 changes King Hugh of Italy to a longer Pope John X answer (F1 0 to40). Remaining three remain F1 zero. The two improved cases account for the entire meanF1 delta versus B. These comparisons do not isolate backend, Transformers version, minimum-length/EOS settings, or device placement as a single causal factor.

Actual effective generation: greedy do_sample=False, num_beams1, max_new_tokens32, min_length=input_length+1; inherited repetition_penalty1.05, EOS[151645,151643], generation pad151643, use_cacheTrue. The saved base generation config has do_sampleTrue and sampling defaults temperature0.7/top_p0.8/top_k20, but official generate explicitly overrides do_sampleFalse, so those sampling defaults are inactive. Tokenizer pad151645 differs from generation pad151643 exactly as in the official loader. Per-sample observed processors are RepetitionPenaltyLogitsProcessor and MinLengthLogitsProcessor.

200 input hashes match A/B, no input truncation. 192 EOS completions; 8 max32 output truncations at indices48,59,63,72,95,117,188,189, identical index set to A/B. No generated token151643. Total generated tokens1373 (including EOS), input tokens2695091. Unmodified official prediction rows and canonical audit rows match on all200 predictions/answers.

Runtime: sum of sample durations797.8417475968599s, loop wall804.9895437713712s, manager wall838.5552978161722s. Peak reserved GPU0/1/2/3:22496/11622/12566/12566MiB. max_memory in placement_plan is a planning input for a fixed map, not a hard CUDA allocation cap; activation/cache allocator usage can exceed that input. Native FA2, BF16 and unmodified full-position logits retained; embeddings and LMhead onGPU0, layers0-7 onGPU1,8-17 onGPU2,18-27 plus norm onGPU3. These are four-GPU timings and do not establish speed against single-GPU A/B.

Completed worker exit0; final and subsequent checks found no worker/manager, all four GPUs0MiB. The earlier single-GPU OOM remains preserved in its original directory. Exact paper data hash/order and model revision cannot be established from inspected official source, so this remains a pinned official-code reproduction with documented cached data/model and multigpu loader adaptation, not proof of the paper's exact environment.

Evidence: extended_result_audit.json, changed_vs_A.json, changed_vs_B.json, comparison.json, final_audit.json, run/native_full_audit.json, run/predictions.jsonl, source_ast_audit.json, official_full_only.patch, placement_plan.json and REPRODUCE.md.

Artifacts: /root/adablock/results/fasa_full_reproduction_20261005_multigpu


FASA gap audit finalized: /root/adablock/results/fasa_full_gap_audit_20261005/REPORT.md

# FASA Full 与论文 55.9 的差异审计

审计日期：2026-10-05。范围：仅检查源码、文件内容、已完成的输出与运行记录，并进行 CPU 重新评分；未执行新模型推理或训练。

## 结论

当前结果应称为“固定公开源码、已验证公开数据及权重、带多卡加载适配的 Full 复现”，不能称为“论文原环境逐位复现”。未发现会改变本次 HotpotQA 提示词、标签、评分或注意力范围的 wrapper 错误。已排除本地权重损坏、误用 LongBench-E、输入截断、中文源码传输损坏影响英文评分等问题。

尚不能解释论文与本次之间全部 2.283626 个百分点的差距。最值得隔离的技术因素是 HF 版本、注意力实现、prefill LM-head 矩阵形状及多卡执行；作者实际外部数据文件、检查点目录和完整环境没有公开，仍是来源层面的未知项。固定 greedy 不应被笼统解释为“随机误差”。这里存在的是浮点执行路径敏感性的候选解释，尚未取得同一隐藏状态/同一前缀的 logits 对照来证实。

## 1. 比较对象与证据边界

[论文 v3 Table 2](https://arxiv.org/html/2602.03152v3#S5.T2) 的 Qwen2.5-7B、Hqa、FKV 行确为 55.9；附录图 15 使用 Qwen2.5-7B-Instruct。256-token 预算针对稀疏方法，不限制完整 KV。论文规定 Qwen 系列最大提示长度 127.5k，QA 使用 F1。本次比较未错读模型或指标。55.9 仅公布到一位小数，真实差距不能精确到六位小数：按通常四舍五入，差距约为 2.2336–2.3336 个百分点。

论文中的 AMAP-ML 仓库地址会重定向到 wangyifei0047/FASA-ICLR2026。固定源码 commit 为 96c00c46db12ccee56104175470aa29579d66c58。该入口的公开历史只有 [2026-08-18 发布提交](https://github.com/wangyifei0047/FASA-ICLR2026/commit/215dbe20377129fe480efe9f90c2e010f871b436)；提交说明称未公开实验输出、捆绑数据和内部作业脚本。公开入口晚于论文，不能倒推出作者当时使用的所有参数。

依赖表的精确含义：

| 项目 | 公开文件 | 本次实际值 | 能否据此确认论文实际环境 |
|---|---|---|---|
| Torch | requirements 固定 2.5.1，并声明核心版本用于论文实验 | 2.5.1+cu121 | 公共版本符合；CUDA 构建仍未确认 |
| Transformers | requirements 固定 4.43，即 4.43.0 | 4.43.0 | 符合声明的版本 |
| FA2 | requirements >=2.7.4；举例 wheel 2.7.4.post1 | 2.7.4.post1 | 满足要求，但作者确切二进制未知 |
| CUDA | README 安装示例使用 PyTorch cu124 源 | Torch cu121；实际 CUDA 扩展记录在审计中 | **示例不是作者运行证据** |
| Accelerate | >=0.26.0 | 复用隔离环境可见的安装，用于多卡 dispatch | 不是论文锁定版本；多卡为本地适配 |
| 内部启动文件 | requirements 注释指向 jobs_r1/start.py | 发布树中未提供 | 无法检查其额外参数 |

## 2. 从 Git 原字节到实际执行的完整改动矩阵

原始文件已按 Git tree 的 blob ID 验证，见 verified_git_blobs.json。canonical_to_executed.patch 是原字节入口到实际执行入口的完整差异；wrapper 本身保留在多卡结果目录的 run_official_full.py。不能只看旧 official_full_only.patch：此前下载传输时部分中文字符已损坏。

| 环节 | 官方公开入口 | 本次执行/差异 | 影响与核验 |
|---|---|---|---|
| 源码文本 | 原始 UTF-8 文件 | 旧保存文件中中文任务提示、中文标点/检索正则损坏 | 已纠正“全部源码原字节一致”的说法；本次英文路径 AST 相同，原字节评分重算逐题一致。未覆盖运行证据 |
| 顶层导入 | 无条件导入多种 sparse 模块 | 删除这些导入 | 静态检查未见 import 即替换模型 forward；但 frequency/utils.py 会创建一个小 CUDA 常量，删除改变初始化/分配状态，不能说完全无副作用 |
| base 分支 | 仅加载模型；不调用 replace/enable sparse 函数 | 直接调用同一加载函数 | 没有 Oracle、Quest、FASA 等 patch；28 层原生 Qwen2FlashAttention2 的运行审计已保存 |
| 模型目录 | FASA_MODEL_DIR/model_name，目录内容未固定 | 明确固定 Qwen2.5-7B-Instruct 快照 | 本地内容已核验；作者同名目录是否相同未知 |
| trust/local 标志 | trust_remote_code=True | False、local_files_only=True、离线环境 | 缓存为原生 qwen2，无 auto_map 自定义入口；模型/tokenizer 内容固定后不改变该原生实现 |
| dtype/backend | BF16、flash_attention_2 | 相同 | 运行原生类证据和实际函数绑定支持；不是仅凭安装版本推断 |
| 模型放置 | CPU 加载后整体 .to(cuda) | 固定四卡 device_map 和规划 max_memory | 真正的执行适配。保留所有层和完整 prefill logits；数学意图相同，未证明位级一致 |
| 数据读取 | ../longbench/hotpotqa.jsonl；main 传 evaluate_num=-1 | 固定已验证文件，读取全部 200 | 无随机子集；作者外部文件未公开，不能认定作者同样使用这 200 条 |
| model alias | Qwen2.5-7B-Instruct 是目录名和查表键 | 相同键，固定快照内容 | 未自动切换 Base、1M 或其他模型 |
| prompt/maxlen | 使用入口硬编码字典，而非旁边 JSON 配置 | 直接使用该字典 | Hotpot prompt 完全一致；127500 raw-token 限制一致 |
| chat/tokenization | 单条 user，apply_chat_template，加 assistant generation prefix，然后 tokenize | 同一原函数，另增加 CPU 哈希检查 | 200 条输入哈希均与旧 Full、A/B 一致；1927–17684 token；未发生截断 |
| batch/顺序 | for-loop，每次一条；新 generate，不传跨题 cache | 包装成逐行调用同一函数 | 有效 batch 都是 1；同一顺序；没有复用前题 KV |
| seed/线程 | seed42，torch/CUDA/numpy/random，cuDNN deterministic | 同一 seed 函数；另设 torch threads=4 | 无采样、eval/dropout0；线程及放置不构成跨库位级保证 |
| generate | greedy、beam1、max_new32、min_length=N+1，继承模型 generation_config | 原函数不变，包装器只记录参数并调用原 generate | 逐题核验 repetition penalty1.05、MinLength、双EOS；不改 logits |
| postprocess | Qwen 不额外改文本；decode skip special tokens | 相同；B 曾额外 strip | FASA 200 条没有首尾空白，strip 不能解释差异 |
| scorer | English QA normalize/F1，逐题最大参考分数，再平均 | 原 scorer + sidecar F1/EM | 使用 Git 原字节纯函数重新评分，所有 200 条一致，仍为 58.18362609923169 |
| 输出恢复 | 入口会删除旧输出，单次循环 | 新目录、fsync 日志、8条检查、续跑验证 | 本次完整一次成功，没有 resume 混入不同配置；官方输出与 sidecar 的 200 条预测一致 |
| 增加的统计 | 官方主要给 F1 | EM、token、输入哈希、耗时、显存审计 | 只读统计，不影响解码；四卡耗时不作为单卡速度对比 |

模型四卡分配：GPU0 为 embedding 和 LM head；层0–7在GPU1、8–17在GPU2、18–27及norm在GPU3。max_memory 为 20/12/12/12 GiB 的规划参数，不是显存硬上限。实测预留峰值为 22496/11622/12566/12566 MiB；原单卡 OOM 证据独立保留。

## 3. 模型与数据内容验证

完整重算四个 safetensors 分片 SHA256，并与 [HF 固定 revision 元数据](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct/revision/a09a35458c702b33eeacc393d103063234e8bc28?blobs=true) 的 LFS SHA256 一一匹配。tokenizer.json、tokenizer_config、vocab、merges、model config、generation config 和索引同时通过实际内容和官方 Git blob 验证。详见 model_content_hashes.json、model_remote_hash_verification.json。没有下载权重。

模型内容：28层、28 Q heads、4 KV heads、hidden3584、vocab152064、BF16、RoPE theta=1e6、无 rope_scaling、use_sliding_window=False、max_position_embeddings32768。config 内的 transformers_version=4.43.1 是保存元数据，不代表实际运行版本；本次运行确为4.43.0。当前完整内容验证不能追溯证明作者目录内容，也不能替代未曾记录的历史运行时权重字节。

本地 data.zip 完整 SHA256 为 cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64，与官方 LFS 一致。公开历史表明此 ZIP 与2023-08-29提交相同，2024-12-18最新快照只更新了README；不存在“近期官方数据更新”证据。2023-08-01曾有另一个ZIP，但未下载或推断其 Hotpot 内容。

ZIP 内 data/hotpotqa.jsonl 与本地实际文件逐字节相等，SHA256 a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f，200个唯一ID、顺序一致。hotpotqa_e.jsonl 是300条、哈希不同、ID集合不同。本次没有误用E变体或相邻片段抽样。此前 Full、cap32、A/B 与本次对同一200输入的比较成立；**这个证据不等于作者仓库外数据文件已被验证**。

## 4. 实际运行库证据与仍缺的观测

运行日志记录了28层原生 Qwen2FlashAttention2、BF16、device_map、逐题生成处理器。审计又解析了同一未修改隔离环境里的绑定链：

Qwen2FlashAttention2.forward → transformers 的 _flash_attention_forward → flash_attn.flash_attn_interface.flash_attn_func → FlashAttnFunc / 已加载的 flash_attn_2_cuda 扩展。

源码文件和扩展 .so 的 SHA256 已保存。源码中无 padding 的 mask 返回 None，本组 batch1无padding，应走普通 flash_attn_func；没有 sparse 替代函数。该结论由运行类证据和代码路径支持，**没有当时的 kernel profiler trace，不能声称记录了每个 CUDA kernel 名称**。B 的 SDPA 内部究竟选择哪个 Torch CUDA kernel 也没有历史 trace。

B 与本次同时变化的数值路径不止 attention backend：

- B 使用4.55.4，generate 自动设置 logits_to_keep=1，LM head只投影末位置隐藏状态。
- 4.43.0 使用 lm_head(hidden_states)，对全部 prefill 位置投影，再把全部 logits 转float。
- 因此 prefill GEMM 的M维从1变成输入长度；即使同一末位置隐藏状态，不同矩阵乘法形状也可能导致浮点舍入不同。目前没有保存两种投影在同一隐藏状态下的对照，不能断言它导致了已见分歧。
- HF版本还影响 RoPE/cache/generation 实现；四卡引入 Accelerate hooks 和跨卡搬运。BF16、greedy和固定seed不能保证这些不同路径逐位相同。
- requirements 固定的是部分公共版本；未固定 FA2精确二进制、CUDA构建、全部依赖和作者硬件。README cu124 示例与本次 cu121 不同，仅构成待确认项，不是已证明的论文环境偏差。

## 5. B→本次的五题：能解释多少，不能解释什么

全部索引为零起始；完整问题、token和原文片段见 changed_case_evidence.json。

| 题号 | B → 本次 | 首个分歧 token 位置 | F1变化 | 原文核验及解释 |
|---|---|---:|---:|---|
| 0 | Gates v. Collier → Miller v. California | 0 | 33.3333→100 | 同一比较案件问题和相同上下文；参考是Miller。改变的是答案选择，不是EOS或尾部处理 |
| 50 | King Hugh of Italy → 含Pope John X的长句 | 0 | 0→40 | 原文明确授予Marozia头衔的是Pope John X；本次长句又混淆Marozia与其母Theodora。F1部分提升不代表整句事实正确 |
| 67 | 9543 → 796 | 0 | 0→0 | 问2010人口，参考810；原文出现Goshen的2020人口796。两组均未答中参考，不能据此说本次更准 |
| 149 | Middlesbrough → 1904 | 0 | 0→0 | 原文区分Leeds City成立1904与Leeds United成立1919；参考1919，本次仍不正确 |
| 189 | “does not provide…” → “does not explicitly…” | 4 | 0→0 | 前4个token相同；均在32token上限截断，未答中Jodie Prenger。变动发生在EOS或截断之前 |

两题提高贡献 (66.6667+40)/200 = 0.533333 个百分点，恰好解释本次相对B的全部平均F1增益，**不能解释与论文55.9之间全部差距**。A、B和本次的作者逐题输出都未知，不能把本地改善当成定位作者误差。

逐项排除条件：

- 双EOS：B与本次均未生成151643，单纯增删此停止token不会改变已观察到的序列。
- min_length=N+1：只在首步抑制EOS，不改变两个非EOS候选的相对排序。它不能单独解释4个首步非EOS分歧，也不能解释第189题第5个token的分歧。没有原始logits，不能反向保证在另一数值路径取消该抑制一定不出现EOS。
- 重复惩罚：两组都是1.05；实际二维处理分支均为gather、正负分数分别除/乘、scatter。它不是本地B→本次的独立变量。
- temperature/top_p/top_k：greedy模式下不参与选token。assistant-token默认值不同但未用assistant model。return_legacy_cache主要是返回格式差异，不能把它当已证实的质量原因；库内部cache实现变化仍需独立核验。
- 输入预算：两组长度均低于实际截断阈值；127500与旧阈值差异未触发。
- 输出预算：同为32，8个截断索引相同。此前128→32降低旧Full约0.1654点，已由保存序列前缀核实；这不是尚未执行的补救。
- 后处理/评分：200条没有strip差异；原字节英文F1重算一致。
- 随机采样：未开启，eval且attention dropout0；不应称为“抽样随机误差”。

## 6. 剩余解释按证据强弱排序

1. **有观测支持的本地路径敏感性候选：** 4/5分歧始于首token，多项浮点路径同时改变，尤其 attention实现和LM-head矩阵形状。这证明结果不完全相同，未证明哪个因素是原因，更不能外推它解释论文差距。
2. **有直接来源缺口的作者运行设置：** 公开发布晚于论文，模型路径和外部数据不固定，内部启动脚本未公开；CUDA/FA2二进制及硬件未锁定。这是无法声称严格复现的明确理由，但不是作者必然改过参数的证据。
3. **作者外部数据或本地模型文件不同：** 保留为可能性。我们已核验标准发布数据和权重，且数据ZIP自2023-08-29稳定，因此“我们缓存损坏/近期数据漂移”的解释已明显弱化。
4. **本地wrapper改变任务语义、误用E、评分错误、截断/双EOS：** 对已见B→本次差异已有强反证，优先级低。删除导入的小CUDA分配和多卡放置仍不具位级等价证明。

若假设作者也是同样200题、同样逐题F1，则每题最多贡献0.5个平均百分点；约2.23–2.33点差距至少涉及5题评分变化。这只是条件数学下界；没有作者 predictions/IDs，无法知道是哪几题、差异方向或成因。Full也是一个模型输出基线，不是逐题必然优于任何稀疏方法的数学质量上界。

## 7. 最小可区分的后续方案（未执行）

先索取作者 Table2 对应的 predictions、200个ID或输入哈希、模型/tokenizer/generation文件哈希、实际命令、pip freeze/CUDA与GPU信息；不需要新GPU实验。本轮没有向作者发送消息。

本地诊断先固定8题：变化题0/50/67/149/189，加稳定且较长的1/2/3。固定所有权重、输入、输出上限和处理器；记录pre-processor logits、top候选差距、首次分歧前的共同token前缀及实际 kernel。不要只比较最终F1，也不要按是否更接近55.9选择配置。

| 顺序 | 单变量对照 | 固定项/判别意义 | 粗估成本 |
|---|---|---|---|
| 1 | **LM-head形状**：同一次prefill得到的同一隐藏状态，分别全序列投影与末位置投影，再比较末位置raw logits | 相同HF4.43/FA2/四卡、相同H和W；直接隔离GEMM形状。诊断性改动，不冒称官方分数 | 8题约1–2分钟含加载；不必完整生成200题 |
| 2 | **attention backend**：HF4.43 FA2 vs HF4.43 SDPA | 四卡、完整LM-head、所有生成参数相同；比较同前缀raw logits/首差异 | 每新增8题分支约1–2分钟，SDPA可能更慢 |
| 3 | **placement**：重现B的HF4.55/SDPA/末位置logits，单卡 vs同一四卡map | 使用B完全相同generation配置；单卡形状已知能装下。两分支仅改变放置/dispatch | 两个8题分支约2–4分钟；加载时间占较大部分 |
| 条件追加 | HF4.43 SDPA vs HF4.55 SDPA，后者明确logits_to_keep=0 | 保持四卡、完整投影与生成参数，进一步隔离HF版本；只在前面不能解释时做 | 新8题分支约1–2分钟 |

单卡官方4.43全logits已OOM，不再重复该配置。先做第1步，依据logits证据决定是否需要第2/3步，而非盲目执行整个矩阵。估算基于本次约4秒/题、约34秒启动开销；诊断hook、完整投影、SDPA和加载波动均可能增加耗时，不是资源报价。若需完整200验证，单个分支暂按约14–20分钟估算；必须先检查8题显存与实际速度，再决定扩展。全程不需要训练、新机器或付费API。

## 8. 交付与审计文件

审计目录：/root/adablock/results/fasa_full_gap_audit_20261005

- active_path_canonical_audit.json、canonical_to_executed.patch：原字节源码与实际执行路径。
- verified_source/、verified_git_blobs.json、transport_diff_summary.json：Git blob原始证据和编码差异。
- import_side_effects_static.json：导入副作用静态清单。
- model_content_hashes.json、model_remote_hash_verification.json、qwen_official_metadata.json：实际文件内容与官方元数据。
- data_content_audit.json、dataset_zip_content_verified.json、longbench_zip_metadata.json、longbench_release_history.json：原始200/E区分及发布历史。
- runtime_source_resolution.json、qwen2_lm_forward.py、fa2_helper.py等：函数绑定、源码及扩展二进制指纹。
- effective_generation_differences.json、inactive_parameter_evidence.json：有效参数和未触发因素证据。
- changed_case_evidence.json：5题问题、原文片段、参考、两组token和输出。

原单卡失败、多卡200结果、A/B、权重、数据及环境均未覆盖。没有新增推理或训练。




# Official FASA sparse256 / FC16 HotpotQA200

{
  "worker_exit": 0,
  "manager_wall_seconds": 867.372917464003,
  "summary": {
    "samples": 200,
    "F1": 57.76682400445557,
    "official_rounded_F1": 57.77,
    "EM": 42.5,
    "input_truncated": 0,
    "output_truncated": 8,
    "ended_eos": 192,
    "sample_seconds": 826.1011357754469,
    "loop_wall_seconds": 832.9114499073476,
    "peak_reserved_mib": {
      "0": 22496.0,
      "1": 11622.0,
      "2": 12566.0,
      "3": 12566.0
    }
  },
  "comparison": {
    "Full_F1": 58.18362609923169,
    "Full_EM": 43.5,
    "FASA_F1": 57.76682400445557,
    "FASA_EM": 42.5,
    "delta_F1": -0.4168020947761164,
    "delta_EM": -1.0,
    "F1_wins": 11,
    "F1_ties": 177,
    "F1_losses": 12,
    "EM_wins": 1,
    "EM_ties": 196,
    "EM_losses": 3,
    "token_changed": 36,
    "first_token_changed": 0
  },
  "budget": {
    "query_heads": 28,
    "kv_heads": 4,
    "head_token_budget": 256,
    "frequency_chunks": 16,
    "feature_dimensions": 32,
    "sparse_layers": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27
    ],
    "prefill_full": true,
    "layer0_dense": true,
    "fixed_sink_local": false,
    "full_kv_cache_retained": true,
    "selector_calls": 29646,
    "expected_selector_calls": 29646,
    "per_head_count_min": 256,
    "per_head_count_max": 256,
    "per_kv_group_union_min": 302,
    "per_kv_group_union_max": 1557,
    "per_kv_group_union_mean": 791.8131788436889,
    "all_query_heads_union_min": 634,
    "all_query_heads_union_max": 3800,
    "all_query_heads_union_mean": 2014.789381366795,
    "self_token_head_fraction": 0.8112248339935043,
    "token0_head_fraction": 0.009525496091980609
  },
  "gpu_after_exit": "index, memory.used [MiB], utilization.gpu [%]\n0, 0 MiB, 0 %\n1, 0 MiB, 0 %\n2, 0 MiB, 0 %\n3, 0 MiB, 1 %\n",
  "calibration": {
    "stage": "completed",
    "samples": 1,
    "decode_queries": 3,
    "counts_per_layer": {
      "0": 3,
      "1": 3,
      "2": 3,
      "3": 3,
      "4": 3,
      "5": 3,
      "6": 3,
      "7": 3,
      "8": 3,
      "9": 3,
      "10": 3,
      "11": 3,
      "12": 3,
      "13": 3,
      "14": 3,
      "15": 3,
      "16": 3,
      "17": 3,
      "18": 3,
      "19": 3,
      "20": 3,
      "21": 3,
      "22": 3,
      "23": 3,
      "24": 3,
      "25": 3,
      "26": 3,
      "27": 3
    },
    "calibration_seconds": 4.538837386295199,
    "agreement_shape": [
      6,
      28,
      64,
      28
    ],
    "computed_budget_index": 1,
    "computed_budget": 256,
    "uncomputed_slots_nan": true,
    "agreement_sha256": "262c6458b817734c345f7b219c652dc873e630bcbc4a2072a18dffd1dc0f56f4",
    "peak_reserved_mib": {
      "0": 5704.0,
      "1": 4522.0,
      "2": 5466.0,
      "3": 5466.0
    }
  }
}

Same cached model, inputs, BF16 evaluation, generation/scoring and four-GPU placement as the completed Full. Calibration uses first1 independent Qasper example per paper (official CLI default8), FP16 per official profiler, only budget256 overlaps; arithmetic mean over decode queries creates the unpublished agreement aggregation format. Other budget slots are NaN and must not be reused. No Hotpot answer tuning.

Official Qwen selector operates after repeating four KV heads to28 query heads: each query head selects256 tokens using16 frequency chunks/32 feature dimensions. Layer0 and all prefills are dense. No mandatory sink/local/self region. Persistent KV cache remains full; union across query heads can exceed256. Finite raw logits, actual per-layer FA2/selector call counts and selected-index uniqueness/range are checked on every sample. Original selector results and FA2 returns are not modified by observers. Timing includes audit overhead and is not an optimized speedup claim. Do not compare this256-token-per-query-head budget as a fair equal-budget result against our1056 setting.

Artifacts: /root/adablock/results/fasa_sparse_native256_20261005


## Same-FASA-protocol RankLoss smoke complete; full200 on hold (2026-10-05)


Both RankLoss variants completed exactly the same first eight official HotpotQA examples. Both passed input/checkpoint/config hash checks, Full-prefill first-token equality, native position/full-cache checks, finite outputs, intended learned-layer call counts, no COBS fallback, and the original 16-token block / 2 sink blocks / 32 local blocks / 32 middle blocks budget.

| Variant | Samples | Seconds | Seconds/sample | Selector calls | Ranker calls |
|---|---:|---:|---:|---:|---:|
| layer0_full | 8 | 446.40 | 55.80 | 756 | 3024 |
| all_layers | 8 | 460.72 | 57.59 | 784 | 3136 |

Same-eight Full time: 34.10 seconds; FASA: 34.82 seconds. RankLoss variant token sequences are identical on these eight examples. These smoke results are not a 200-example quality comparison.

Both full200 stages remain unstarted: there are exactly eight canonical rows per variant, no limit200 logs, and no COMPLETE200 markers. The smoke-only guard PID 2449 exited. Original manager PID 920 remains SIGSTOP, holding about 9.8 MiB host RAM and no GPU resources. Do not resume it directly because its in-memory loop includes the full200 stages. Finalizer PID 2016 is sleeping while awaiting full completion, holding about 14.3 MiB host RAM and no GPU resources.

All four GPUs report 0 MiB allocated and no compute processes. Existing predictions and checkpoints are preserved. No profiling, optimization or new inference was launched during this verification.

Evidence: smoke_completion_audit.json, smoke_timing.json, each variant's run/smoke8_passed.json and run/predictions.jsonl. Runtime analysis and proposed validation costs: SMOKE_RUNTIME_REVIEW.md.

Experiment directory: /root/adablock/results/rankloss_fasa_same_protocol_20261005.


## Same-protocol RankLoss profile: initial SVD bottleneck (2026-10-05)


The bottleneck is the first-decode all-history covariance SVD, not ranker inference, KV transfer or the measured budget audit. The long trace spends 63.76 of 74.13 seconds in initial SVD (86.0%); the short trace spends 8.29 of 12.55 seconds (66.0%). The two complete traces preserve the retained smoke tokens, first-decode selection hashes and saved budget audits exactly.

## Bounded runs and integrity

Three diagnostic inference attempts were executed: layer0_full long index 1 (71.36 seconds; trace lost before save due to integer-versus-string JSON layer keys), layer0_full short index 7, and all_layers long index 1. The failed-save long example was not rerun. Two complete segmented traces remain. No new training, selector change, budget change or full200 run occurred. The JSON comparison was normalized and trace saving moved before assertions for the remaining two runs.

| Component | Short: 2,317 input / 6 output tokens, 27 learned layers | Long: 17,322 input / 5 output tokens, 28 learned layers |
|---|---:|---:|
| Whole sample | 12.5463 s (100.00%) | 74.1299 s (100.00%) |
| Native model prefill | 1.4103 s (11.24%) | 5.4896 s (7.41%) |
| Initial all-history digest, including SVD | 8.4296 s (67.19%) | 63.9871 s (86.32%) |
| Initial SVD, included above | 8.2861 s (66.04%) | 63.7590 s (86.01%) |
| Selection total, including ranker and top-k | 1.0968 s (8.74%) | 3.0101 s (4.06%) |
| Ranker forward, included in selection | 0.7680 s (6.12%) | 2.6592 s (3.59%) |
| COBS feature terms, included in ranker | 0.1018 s (0.81%) | 0.4076 s (0.55%) |
| Position features, included in ranker | 0.1200 s (0.96%) | 0.4827 s (0.65%) |
| Top-k dispatch, included in selection | 0.0762 s (0.61%) | 0.0897 s (0.12%) |
| Incremental digest update, including SVD/clones | 0.4900 s (3.91%) | 0.3691 s (0.50%) |
| GPU KV gather plus index H2D | 0.0592 s (0.47%) | 0.0494 s (0.07%) |
| Decode FA2 dispatch | 0.0497 s (0.40%) | 0.0444 s (0.06%) |
| Budget audit, CPU | 0.1361 s (1.08%) | 0.1595 s (0.22%) |
| Finite-logit audit | 0.0052 s (0.04%) | 0.0051 s (0.01%) |
| Cross-GPU transfer call wall | 0.0324 s (0.26%) | 0.0996 s (0.13%) |

Rows are inclusive and several are nested: do not add this table. Host dispatch time understates asynchronous FA2 computation; CUDA event prefill-FA2 totals are 0.0258 / 0.8596 seconds (short/long), decode-FA2 totals 0.0466 / 0.0418 seconds, and cross-GPU intervals 0.0747 / 0.5279 seconds. CUDA event intervals include stream waits/host gaps and are not isolated kernel times.

Initial SVD CUDA-event totals are 8.2814 / 63.7199 seconds, close to the host-call totals 8.2861 / 63.7590 seconds. This directly measures the SVD API's dominant time, rather than inferring it from GPU utilization. Initial digest work outside SVD totals only about 0.144 / 0.228 seconds; deleting unused min/max/sphere summaries alone cannot address the main cost.

## What actually runs

- Initial digests: exactly 27 calls for the short variant and 28 for the long variant, once per learned layer. There is no all-history rebuild on every decoded token.
- Short: 15,660 initial 16-by-128 matrices in 135 SVD API calls, five calls per learned layer. Long: 121,296 such matrices in 952 calls, 34 per learned layer. Batch chunks are 128 matrices plus a tail. These are many small matrix problems, not one SVD per API call.
- Incremental digests: 135 / 112 calls for short/long, exactly one per learned-layer decode step; each SVD call contains four 16-by-128 matrices. Only the current block is recomputed.
- All observed SVD inputs are CUDA tensors on layer devices 1/2/3, with the original torch default driver and full_matrices=False. No tensor-level CPU fallback was observed. These timers do not name or isolate internal cuSOLVER kernels.
- Cross-device payload observed by Accelerate hooks is about 67.6 / 504.5 MB (short/long). Full historical KV stays on the owning GPU; gather/index transfer totals are only about 0.059 / 0.049 seconds.
- Ranker feature functions and neural submodules are separately recorded in the trace. The ranker runs four KV-head calls per learned layer/decode step and retains seven query-head nonlinear processing before pooling.

## Timing and audit limitations

Model/tokenizer loader wall time, outside sample timing: 6.835 / 6.722 seconds. This does not include interpreter imports or the earlier 200-input CPU manifest construction.
Observed sample overhead versus retained smoke: short +2.218 seconds (+21.48%); long +3.216 seconds (+4.53%). These differences include event/hook instrumentation, changed cold/warm state (the original short example followed seven samples), and normal variation. They are not a calibrated measurement of instrumentation alone.
Component timers do not synchronize at every component; they record CUDA events and host wall around unchanged calls. The model forward wrapper explicitly synchronizes all four GPUs at its boundaries. Event reduction and diagnostic JSON/journal writes occur after the recorded sample total and have no separate timing. Saved complete traces are about 110 KB and 268 KB. Official prediction writing within get_pred is included in sample wall but not isolated, so no precise logging-time claim is warranted.
Budget CPU audit is 0.136 / 0.159 seconds; finite-logit checks are about 0.005 seconds each. Position, cache and other small checks are retained but not all individually isolated. These measured audits are too small to explain the 50–70-second runtime.
All generated token IDs and first-decode layer selection hashes match the original smoke. Complete saved budget audits also match. Later per-step selection hashes were not retained in the original smoke, so exact equality of every later selection cannot be claimed; the diagnostic now retains those hashes.

## Minimal next options — proposed only

1. Focus any optimization experiment on the initial SVD call. Keep FP32 centered blocks, exact thin decomposition, rank four, the same covariance normalization and the same ranker/budgets. A bounded microbenchmark of the equivalent transposed thin-SVD layout and batching is the smallest numerical path to test before touching the rest of the runtime. Transposition preserves the mathematical rank-four target, but floating-point vectors/ties can differ; speed and token equality are not guaranteed.
2. Validation gate for that proposal: one-layer representative blocks with a small fixed set of layout/batch choices (budget at most 1–2 minutes), compare reconstructed rank-four covariance and COBS features, ranker scores and selected indices; then the existing synthetic/GQA/cache suite (previous body time 4.6 seconds), followed by one retained long and one short inference (about 1.5–2 minutes plus loading). Stop on a selection/token difference and report it; do not silently change the ranking objective. No such tests or implementation have been run.
3. If exact reuse is preferable to numerical changes, the two variants share native Full-prefill K/V. A future paired-example schedule could reuse identical initial digest tensors across variants after verifying matching keys, avoiding much of the second initialization. This requires scheduling/cache-lifetime work and separate accounting of amortized time; it is not an independent-method speed comparison. It has not been implemented.
4. Removing the measured budget/finite audits or unused non-SVD digest fields offers only subsecond headroom on this long sample. It is not the priority.

## Final state and evidence

Both diagnostic managers and inference workers exited. All four GPUs show 0 MiB allocated and no compute processes. Original manager PID 920 remains SIGSTOP; the original two journals remain exactly eight examples each, with no full200 logs. No manager was resumed.
Artifacts: profile_summary.json; instrument.py; serialization_fix.json; per-variant run/profile_sample_*.json, model_load_timing.json, predictions.jsonl, config.json and diagnostic_complete.json; worker logs; preserved layer0_full/failed_run_before_json_key_fix/.

Profile directory: /root/adablock/results/rankloss_fasa_same_protocol_20261005/profile_smoke_20261005.


## Same-protocol RankLoss full200 resumed before midnight (2026-10-05)

The user authorized proceeding with the original frozen implementation to start the long run before 2026-10-05 16:00 UTC (Beijing midnight). Launch reference: 2026-10-05 15:13:35 UTC, about 23:13 Beijing time.

A new detached resume manager PID 4624 runs layer0_full first, then all_layers, each to 200. The first eight audited predictions in each variant are reused without rerunning them. Worker PID 4625 was verified alive, loading all four local checkpoint shards and entering sample index 8 (13,269 chat-input tokens); its worker_limit200.log grew to 1,893 bytes. The manager and worker have session ID 4624, independent of SSH, with stdin=/dev/null and file-backed output. HF offline mode, local model/tokenizer files, the frozen 50k checkpoint, local dataset, original budgets, four-GPU placement and unchanged inference source hashes are enforced.

The stopped obsolete manager PID 920 was retired without resuming its loop. The smoke guard had already exited. Existing finalizer PID 2016 remains waiting for the new manager's completion and will generate paired comparisons, the report, documentation and GPU-release evidence.

Per-sample JSONL writes flush and fsync; restart validation checks a complete sequential prefix, identity hashes and matching configuration. Failures preserve the journal and stop subsequent stages. Do not launch a second manager while resume_manager.py holds manager.lock. Expected remaining wall time is approximately six hours, uncertain with output lengths and host/GPU conditions.

No SVD or other optimization was implemented. The profile's initial-SVD bottleneck is accepted for this quality run. Network disconnection from the Windows client will not stop these detached local-data jobs; interruption or shutdown of the remote instance remains outside this mechanism.

State/evidence: manager_status.json, resume_authorization.json, resume_manager.log, layer0_full/worker_limit200.log and all_layers/worker_limit200.log when that stage starts. All paths are under /root/adablock/results/rankloss_fasa_same_protocol_20261005.



## RankLoss versus FASA under the same generation protocol (2026-10-05)


Frozen 50k checkpoint; official 200 HotpotQA inputs; Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28. Transformers 4.43.0, Torch 2.5.1+cu121, FlashAttention2 2.7.4.post1, BF16, same four-GPU placement. Greedy generation, repetition penalty 1.05, max_new_tokens 32, dual EOS 151645/151643, min_length=input_length+1. Full and FASA results reused after protocol/hash verification.

| Configuration | F1 | EM | Output capped | Sample seconds |
|---|---:|---:|---:|---:|
| Full | 58.1836 | 43.50 | 8 | 797.8 |
| FASA | 57.7668 | 42.50 | 8 | 826.1 |
| RankLoss_layer0_full | 58.0165 | 44.00 | 8 | 10890.9 |
| RankLoss_all_layers | 58.1285 | 43.50 | 9 | 11247.2 |

All 200 first tokens match Full in every configuration. Prefill is native dense attention. Every generated decode step was audited for expected layer calls, no COBS fallback, position/cache consistency, finite outputs and native budgets.

## Paired differences (A minus B)

- Full minus FASA: F1 +0.4168 pp (wins/ties/losses 12/177/11; bootstrap95% [-1.1740172863816372, 2.02433601399251]), EM +1.00 pp; 36 changed token sequences.
- Full minus RankLoss_layer0_full: F1 +0.1671 pp (wins/ties/losses 8/187/5; bootstrap95% [-0.9368889421957162, 1.2446131351345462]), EM -0.50 pp; 21 changed token sequences.
- Full minus RankLoss_all_layers: F1 +0.0551 pp (wins/ties/losses 8/186/6; bootstrap95% [-1.0910670072516715, 1.1388684012209713]), EM +0.00 pp; 22 changed token sequences.
- FASA minus RankLoss_layer0_full: F1 -0.2497 pp (wins/ties/losses 10/179/11; bootstrap95% [-1.6238195542921954, 1.1191599468297286]), EM -1.50 pp; 34 changed token sequences.
- FASA minus RankLoss_all_layers: F1 -0.3617 pp (wins/ties/losses 9/180/11; bootstrap95% [-1.6816459309739111, 0.9215300633758112]), EM -1.00 pp; 32 changed token sequences.
- RankLoss_layer0_full minus RankLoss_all_layers: F1 -0.1121 pp (wins/ties/losses 1/197/2; bootstrap95% [-0.6349206349206349, 0.3800000000000001]), EM +0.50 pp; 9 changed token sequences.

Layer-0 intervention is isolated by the two RankLoss variants; generation and the remaining 27 layers use identical code and weights. Interpret the paired score difference on this finite set rather than a universal effect.

## Validation and limitations

Synthetic tests compare exact selected indices and incremental digests against the frozen original controller at short, boundary and partial-block lengths on GPUs 1, 2 and 3. GQA mapping and full cache are checked exactly; selected FA2 attention agrees with FP32 softmax within max absolute 0.002 and relative L2 0.24%. Both variants pass separate first-eight-example smoke gates before 200-example continuation.

- FASA256 per query head and RankLoss66 historical blocks per KV head plus current self are native unequal budgets.
- All methods preserve full KV cache. Adapter uses GPU cache gathering and FA2; prototype used CPU history and SDPA.
- Elapsed time includes digest construction, Python ranking, synchronizations and audits; not fused-kernel throughput.
- Only 200 official HotpotQA examples and one checkpoint; no training or budget tuning in this experiment.

Artifacts: adapter_tests.json, fasa_reuse_audit.json, source_hashes.json, rankloss_adapter.py, manager.py, each variant's run/config.json, run/rankloss_runtime_audit.json, run/smoke8_passed.json, run/predictions.jsonl, run/summary.json, and comparison.json.


Experiment directory: /root/adablock/results/rankloss_fasa_same_protocol_20261005.


## Full versus paper: follow-up source audit and bounded diagnostic plan (2026-10-06)


Scope: source/metadata/retained-output analysis only. No new model inference, model construction, package installation, environment changes, or large weight/data rehashing. Existing 200-example Full/FASA/RankLoss results remain unchanged.

## New actionable findings

1. **The old single-GPU OOM does not rule out a same-HF4.43/FA2 placement test.** Of the five B-to-FASA divergent cases, index 50 has only 5,279 chat-input tokens. It diverges at the first token: retained B token 33555 versus FASA Full token 47. Index 129 (1,927 tokens) is a stable control with identical three output tokens. The other divergent cases are substantially longer: 0=11,787; 67=12,840; 149=16,912; 189=15,963. Prior advice to avoid repeating the full-length single-card OOM must not be interpreted as ruling out these bounded short inputs.

   Read-only lower-component memory estimate for #50: stored BF16 model weights 14.185 GiB, simultaneous BF16+FP32 prefill logits 4.486 GiB, compact full KV 0.282 GiB. This leaves plausible space on a 24-GiB card, but excludes RoPE buffers, activations, CUDA workspace and allocator overhead; it is not a guarantee. Start with #129 and inspect actual reserved memory before #50. Do not run the 11k–17k cases on this configuration and do not silently switch to last-only logits to make it fit.

2. **This four-GPU path is layer sharding, not tensor parallelism.** The saved map assigns whole decoder layers to GPUs 1/2/3 and embedding/LM head to GPU0. Accelerate 1.13.0 dispatch uses send_to_device with Tensor.to(device), without a dtype conversion argument; no tensor-parallel collective reduction is introduced. Qwen2 skips past_key_values in generic placement. Thus “multi-GPU changes all-reduce order” is not supported here. Placement/dispatch, buffer initialization and CUDA library selection still require a direct diagnostic; lossless copies alone do not prove the complete execution bitwise identical.

3. **There are now concrete HF-version candidates beyond a generic version label.**
   - HF4.43 Qwen2RotaryEmbedding creates cached sin/cos tables using outer(position, inv_freq), and indexes/casts that cache during forward. Its attention constructor does not pass a device. HF4.55 computes float32 position products and trig inside forward on x.device. Buffer construction/movement history was not recorded in the old runs; do not assert a proven CPU-versus-GPU discrepancy. Record actual buffer dtypes, initialization/movement devices and the selected cos/sin values in a diagnostic.
   - HF4.43 FA2 and SDPA explicitly repeat 4 KV heads to 28 query heads. HF4.55 SDPA can pass enable_gqa=True with compact KV heads when Torch>=2.5 and the attention mask is None. The installed source satisfies the version predicate, but the historical B mask/kernel branch was not traced. Record actual shapes, mask and enable_gqa rather than treating this conditional source path as observed runtime.
   - HF4.43 has **no logits_to_keep argument** in its Qwen2 forward. Passing that argument is not a valid clean control. Its whole-sequence LM-head projection and FP32 cast differ from modern last-position projection. Isolate GEMM shape by applying the same LM head to the same captured hidden tensor and its last-position slice; do not simultaneously switch Transformers versions.

4. **The saved base generation_config is not the effective invocation.** native_full_audit.json contains inherited do_sample=True, max_new_tokens=None and min_length=0. The unchanged official get_pred_single_gpu explicitly overrides these to do_sample=False, max_new_tokens=32 and min_length=input_length+1. All retained per-example processor/stopping audits confirm the effective values. For #50: min_length=5280, repetition_penalty=1.05, dual EOS 151645/151643 and max-length criterion 5311. A comparison based only on the inherited config would incorrectly diagnose sampling or a missing min-length guard.

## Exact public-release provenance and entry behavior

Fixed author commit: 96c00c46db12ccee56104175470aa29579d66c58. The verified requirements text literally specifies torch==2.5.1, transformers==4.43, flash-attn>=2.7.4, accelerate>=0.26.0 and datasets==3.6.0. It names jobs_r1/start.py as the source of paper core-version pins, but that file is absent from the public tree. It does not lock a Torch CUDA build, exact FA2 version/binary or Accelerate version.

Actual local environment: Torch 2.5.1+cu121, Transformers 4.43.0, FA2 2.7.4.post1, Accelerate 1.13.0. The retained wheel-provenance record identifies the official Dao-AILab cu12/torch2.5/Python3.10 wheel and its hash; this establishes our binary, not the author's. README's cu124 install command and wheel example are not a paper runtime manifest.

The verified bash.sh sets CUDA_VISIBLE_DEVICES=0 and demonstrates Meta-Llama-3.1-8B-Instruct / narrativeqa / dynamic_frequency. It is not a Qwen/Hotpot/base experiment record. No alternate public LongBench generation entry or Qwen/Hotpot job configuration was found in the cached complete file tree. Perplexity/reasoning launchers concern other tasks. The available history explicitly excludes internal job scripts, data and experiment dumps. This does not prove the authors used hidden settings; it limits what can be reconstructed.

The public parse_args default evaluate_num=8 is misleading for the main evaluation: __main__ passes literal -1 to eval_longbench, so CLI evaluate_num does not limit that path. full_longeval_datasets contains qasper but is also bypassed because main passes [args.dataset_name]. The JSON prompt/max-length files are commented out; active dictionaries are in longbench_pred.py. The base branch does not read sparse compression parameters, budget or frequency-calibration files. The external dataset path is relative to process CWD (../longbench/hotpotqa.jsonl), and the model directory is FASA_MODEL_DIR/model_name. The content of the author's actual external paths remains unavailable.

Actual call graph:
run_official_full.py -> imported patched_source/longbench/longbench_pred.py -> seed_everything + patched load_model_and_tokenizer -> AutoModelForCausalLM.from_pretrained with fixed device_map -> Accelerate dispatch -> per-row get_pred_single_gpu -> tokenizer/build_chat -> model.generate wrapper -> original GenerationMixin.generate -> native Qwen2ForCausalLM/Qwen2Model -> native Qwen2FlashAttention2 -> Transformers FA2 helper -> installed flash_attn interface/extension -> whole-sequence LM head -> repetition/minlength processors -> greedy argmax -> EOS/max32 -> official decode/post_process -> original English scorer.

The wrapper calls get_pred_single_gpu directly once per example; it does not execute the author's CLI __main__/eval_longbench as a complete process. Its seed, input order, effective batch1 and generation function are aligned, but lifecycle, import allocations, threads=4, output journaling and dispatch are adaptations.

**Not all executed upstream files were original bytes.** The executed longbench_pred.py had an explicit loader/import patch plus earlier UTF-8 transport damage in unrelated Chinese strings. metrics.py also had Chinese-only transport changes; eval.py was byte-identical. The preserved canonical audit shows AST equivalence for get_pred_single_gpu, build_chat, post_process, seed_everything, read_jsonl and active English metric functions; Hotpot prompt/maxlen dictionaries match. This supports this task's semantic path, not a claim that the entire author source tree ran unmodified. Use the canonical_to_executed.patch and source_ast_audit/active_path_canonical_audit evidence rather than only the smaller loading patch.

## Additional public version mapping

The author-hosted OpenReview rebuttal PDF, page 8 Table 4, explicitly labels Qwen2.5-7B-Instruct and reports HotpotQA FKV 55.90 and FASA 58.29:
https://cseweb.ucsd.edu/~jmcauley/reviews/iclr26b.pdf

The previously audited final paper table retains Full 55.9 but reports FASA 57.4. These are distinct public experimental tables; do not mix their sparse numbers or assume they came from one unchanged job. The rebuttal evidence strengthens the Instruct-model identification and shows Full 55.9 predates the public code release. It does not identify the source of our Full discrepancy or prove a bug, changed data or an undocumented parameter.

## Minimal staged experiment matrix — approval required before running

All stages freeze the same local snapshot/tokenizer, official input hashes, BF16, Torch 2.5.1+cu121, seed42, no dropout, full cache, prompt/chat, greedy/beam1, max_new32, repetition penalty1.05, dual EOS and min_length=N+1. No installation is needed.

| Stage | Branches / fixed sample IDs | Sole intended variable | Diagnostic and decision | Bounded cost estimate |
|---|---|---|---|---|
| P1, first | HF4.43/FA2/full logits; four-card original map vs single-card GPU0; each #129 then #50 | placement/dispatch | Use device_map in both branches, with identical low_cpu_mem_usage, to avoid also changing loading strategy. Single map is {"":0}. Save first raw and processed logits, top2 margins, B/FASA candidate ranks, every generated token, actual buffer/copy metadata and per-layer last-position hidden fingerprints. Stop on memory pressure; no long inputs. | 2 model loads, 4 short-example generations; plan 2–3 min, hard operational budget 5 min. No 200 expansion. |
| P1b, conditional only | Single-card device_map={"":0} vs original from_pretrained(...).to("cuda") on #50 | loading/dispatch path | Needed only if P1 implicates placement/buffers or exact author loading remains material. This is separate from P1, not silently combined. | One additional load/example, roughly 1–2 min. |
| P2, after P1 | Captured identical prefill hidden tensor, identical lm_head weight/device; full H projection vs H[:,-1:] | GEMM M dimension | Keep official raw logits and compare both projections after the original forward has produced its output. No new full-model forward is required if H is retained. Compare raw and processed top2 and whether candidate order flips. HF4.43 does not support a logits_to_keep switch. | At most two extra LM-head projections per short case; seconds of compute, included in a <=1 min allowance; no full evaluation. |
| P3, only if needed | HF4.43 FA2 baseline vs HF4.43 SDPA, same four-card map/full logits/#129,#50 | attention backend | Keep output_attentions=False to avoid SDPA eager fallback. Record actual mask, repeated KV shapes and selected Torch kernel. If a margin/token changes, local backend sensitivity is established for this case only. | One added load, two examples, roughly 1–2 min. |
| P4, conditional investigation | Compare old/new RoPE tensors and actual GQA layouts at fixed positions/shapes; only then a controlled HF version branch | separately isolated RoPE/GQA or version | Scalar metadata alone is insufficient. Never switch HF version, logits shape, backend and placement together and label it one-factor evidence. | Tensor-only inspection seconds; any added inference branch needs a separate bounded decision. |

P1 is the immediate requested test scope: four short generations total, not the former eight-long-case grid. P2/P3 are staged choices, not automatic permission to run the whole matrix. Costs are engineering estimates based on recent 6–13 second model loads and prior Full sample timings, not measured costs for this instrumentation.

For every branch, save traces **before assertions**, with JSON-normalized keys. Verify the four-card baseline reproduces the retained token stream before interpreting changes. Compare logits on an identical prefix: a full-generation comparison is causal only through the first divergent token. Capture first-step margins for #50 (already a first-token disagreement); do not compare later logits from different generated histories. Store compact top candidates/last-position hidden tensors and bounded diagnostic logits; avoid complete all-layer caches. Record both raw top2 and post-processor top2, because repetition penalty can change the chosen ordering.

A null single-versus-multi result constrains placement for the tested cases, not all 200 or the author's hardware. A head-shape/backend flip can explain some local B-to-public-harness differences; it still does not prove why the author's Full is 55.9. The local B->FASA Full gain is only +0.533333 points, from two scored cases. The paper gap remains approximately +2.28 points with no author per-example output to align. Do not optimize a configuration merely toward 55.9.

## Prepared evidence and remaining blocker

evidence.json fixes IDs, input hashes, retained B/Full tokens, memory-component estimates, literal requirements and source pointers/line numbers. Numbered source excerpts preserve the exact local implementation examined. No experiment script has been launched or environment altered.

The strongest missing author artifact remains the Table-2/55.90 job's launch/config plus prediction IDs/tokens or hashes, and exact model/generation/environment metadata; jobs_r1/start.py is not public. We have not contacted the authors. Locally, permission to run P1 is the only next operational gate.

Follow-up directory: /root/adablock/results/fasa_full_gap_audit_20261005/followup_20261006.


## MuSiQue and MultiFieldQA-en comparison launched (2026-10-06)

# Two-task evaluation launched 2026-10-06
Tasks: musique (200, official max_new_tokens32) and multifieldqa_en (150, official64).
Methods: original FASA256/FC16 with layer0 dense; frozen 50k RankLoss all28, block16/sink2/local32/middle32 per KV head, no COBS fallback. These are unequal-budget comparisons.
Same Qwen snapshot, HF4.43/FA2/BF16, original four-GPU map, prompt/chat/greedy/repetition1.05/dual-EOS/min_length=N+1 protocol. No truncation needed for audited inputs.
Historical Full exists in results/full_musique_32k_gen128 and results/full_multifieldqa_32k_gen128, with F1 28.005944 and52.221464; each used128-token cap and lacks equivalent per-example protocol hashes, so neither is a same-protocol Full comparator. No Full long run launched; paper gap remains unresolved.
Manager runs two-example smoke for each task/method, checks paired input hashes and first tokens (both dense prefill), then proceeds to full counts. No extrapolation of smoke quality. All inference jobs run sequentially on four GPUs to preserve verified placement and avoid concurrent memory pressure.
Initial estimated duration5-7h, to be refined from observed smoke. Existing calibrated FASA frequency selection is reused unchanged; no new calibration.
Outputs: each task/method/run holds input_manifest, config, per-sample prediction/token/audit journal and eventual summary. Root holds manager_status, smoke_gate, source_hashes, comparison, REPORT and GPU completion evidence. Manager is detached from SSH; no shutdown policy changes.

Experiment: /root/adablock/results/fasa_rankloss_musique_multifieldqa_20261006


Clock correction (2026-10-06): server clock measured about623.95 seconds fast (network interval uncertainty +/-0.50 s). Recorded full launch06:16:28 server UTC corresponds to real UTC approximately06:06:04. Use local clock and perf_counter sample durations for ETA. See clock_audit.json. No job restarted.


## MuSiQue completed paired comparison (2026-10-06)

# MuSiQue complete: FASA versus all-layer RankLoss

200 paired examples, official generation cap32. Same inputs and dense-prefill first tokens; zero COBS fallback.

|Method|F1|EM|Output capped|Sample seconds|
|---|---:|---:|---:|---:|
|fasa|29.8146|20.50|25|1104.5|
|all_layers|30.2983|20.50|29|14053.2|

RankLoss minus FASA: F1 +0.4837 pp, paired bootstrap95 [-1.0637704433866182, 2.0575313082965296]; EM +0.00 pp. F1 wins/ties/losses [17, 166, 17]; changed token sequences 74.

Budgets remain unequal: FASA256 per query head (layer0 dense), RankLoss block16/sink2/local32/middle32 per KV head (all28 learned). Full KV retained. Historical Full protocol differs; no same-protocol Full claim and no resolution of paper discrepancy. MultiFieldQA-en remains in progress.

Artifacts: /root/adablock/results/fasa_rankloss_musique_multifieldqa_20261006/musique


## Two additional LongBench tasks (2026-10-06)

# FASA versus all-layer RankLoss: MuSiQue and MultiFieldQA-en

Qwen2.5-7B-Instruct; frozen 50k ranker; existing verified HF4.43/FA2 environment and four-GPU map.
FASA: native256 per query head, FC16, layer0 dense. RankLoss: learned all28, block16/sink2/local32/middle32 per KV head, no COBS fallback. Budgets are unequal; full cache retained. Timing includes auditing and digest preprocessing.
Official task generation limits: MuSiQue32, MultiFieldQA-en64. Historical Full used128 and is not a same-protocol baseline; no new Full run. This experiment does not resolve the paper Full discrepancy.

|Task|Method|N|F1|EM|Output capped|Seconds|
|---|---|---:|---:|---:|---:|---:|
|musique|fasa|200|29.8146|20.50|25|1104.5|
|musique|all_layers|200|30.2983|20.50|29|14053.2|

musique: RankLoss-FASA F1 +0.4837 pp; paired bootstrap95 [-1.0637704433866182, 2.0575313082965296].

|multifieldqa_en|fasa|150|51.5271|21.33|20|547.9|
|multifieldqa_en|all_layers|150|53.3090|22.00|17|5697.2|

multifieldqa_en: RankLoss-FASA F1 +1.7820 pp; paired bootstrap95 [-0.19947112255933605, 3.8487650057333633].


Artifacts: /root/adablock/results/fasa_rankloss_musique_multifieldqa_20261006


## Final independent two-task verification (2026-10-06)

# Final verified results: MuSiQue and MultiFieldQA-en

All four stages completed. Final resource verification: 2026-10-06 12:02:24 true UTC. Manager619 and worker5266 exited; four GPUs0MiB and no compute processes. No job restarted and no new Full run.

Frozen Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28, frozen50k RankLoss checkpoint, HF4.43.0/Torch2.5.1cu121/FA2 2.7.4.post1, BF16 and existing four-GPU placement. Generation caps32 for MuSiQue and64 for MultiFieldQA-en.

|Task|Method|N|F1|EM|Output capped|Sample minutes|
|---|---|---:|---:|---:|---:|---:|
|musique|fasa|200|29.8146|20.5000|25|18.41|
|musique|all_layers|200|30.2983|20.5000|29|234.22|
|multifieldqa_en|fasa|150|51.5271|21.3333|20|9.13|
|multifieldqa_en|all_layers|150|53.3090|22.0000|17|94.95|

|Task|Metric|RankLoss-FASA pp|Paired bootstrap95 pp|RankLoss wins/ties/losses|
|---|---|---:|---|---|
|musique|F1|+0.4837|[-1.0637704433866182, 2.0575313082965296]|[17, 166, 17]|
|musique|EM|+0.0000|[-2.0, 2.0]|[2, 196, 2]|
|multifieldqa_en|F1|+1.7820|[-0.19947112255933605, 3.8487650057333633]|[37, 85, 28]|
|multifieldqa_en|EM|+0.6667|[-2.6666666666666665, 4.0]|[4, 143, 3]|

Both F1 confidence intervals include zero. These runs do not establish a statistically clear advantage. Intervals use10000 paired example resamples, seed42; no adjustment for multiple comparisons. All700 predictions independently rescored with the frozen official English QA metric and EM normalization; exact agreement. All350 paired input IDs/hashes and dense-prefill first tokens match. No input truncation.

## Actual budget and implementation limits

FASA uses256 selected tokens per query head, FC16, sparse layers1-27 and dense layer0. Current self/sink/local tokens are not mandatory fixed regions. RankLoss uses all28 learned layers, block16,2sink blocks,32local blocks,32middle blocks per KV head. It selects1041-1056 historical tokens plus self, total1042-1057 per KV head, shared by its7 query heads. No COBS fallback. Thus these are unequal-budget native-configuration comparisons, not equal-budget quality claims.

|Task|FASA union within a7-Q-head KV group min/max(mean)|FASA all-Q union min/max|RankLoss historical-token union across4KV heads min/max (add self once)|
|---|---|---|---|
|musique|[324, 1573] (823.49)|[737, 3904]|[1138, 2544]|
|multifieldqa_en|[291, 1534] (773.92)|[510, 3539]|[1076, 2541]|

Union columns have explicitly different scopes; do not treat per-query-head256 as physical256-token KV residency. Both methods retain full KV cache. Sample timings include instrumentation and RankLoss SVD/digest preprocessing and describe the current unoptimized implementations, not a kernel speedup claim. This is end-to-end QA evaluation, not a direct measurement of block attention-mass ranking coverage.

Historical Full runs use a different generation/protocol configuration and are not reused as matched baselines. No conclusion is drawn about the unresolved paper Full-score discrepancy.

## Reproducibility and evidence

Root: /root/adablock/results/fasa_rankloss_musique_multifieldqa_20261006

- experiment_plan.json; source_hashes.json; runner_changes.patch under each method.
- Per task/method/run: config, input_manifest, prediction and token journals, per-layer budget audit, scoring summary and completion marker.
- comparison.json; final_paired_audit.json; final_integrity_budget_audit.json; gpu_after_completion.json; manager_status.json.
- Frozen ranker SHA256: d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae.
- clock_audit.json records initial server offset. Later observation showed roughly620s fast, so server wall timestamps must not be read as true UTC.
