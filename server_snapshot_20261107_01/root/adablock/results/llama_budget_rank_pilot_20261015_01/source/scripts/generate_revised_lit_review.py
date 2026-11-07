from pathlib import Path
from html import escape
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import date


OUT = Path("研究生学位（毕业）论文文献综述_修改稿.docx")
MD_OUT = Path("研究生学位（毕业）论文文献综述_修改稿.md")


def x(text: str) -> str:
    return escape(str(text), quote=False)


def run(text: str, *, bold=False, italic=False, size=24, color=None, font="宋体") -> str:
    rpr = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:hint="eastAsia"/>',
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
    ]
    if bold:
        rpr.append('<w:b/>')
    if italic:
        rpr.append('<w:i/>')
    if color:
        rpr.append(f'<w:color w:val="{color}"/>')
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ''
    return f'<w:r><w:rPr>{"".join(rpr)}</w:rPr><w:t{preserve}>{x(text)}</w:t></w:r>'


def para(text="", kind="body", runs=None):
    if kind == "title":
        ppr = '<w:jc w:val="center"/><w:spacing w:before="80" w:after="240"/>'
        default = dict(size=36, bold=True)
    elif kind == "meta":
        ppr = '<w:jc w:val="center"/><w:spacing w:after="30"/>'
        default = dict(size=28)
    elif kind == "label":
        ppr = '<w:spacing w:before="80" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
        default = dict(size=24, bold=True)
    elif kind == "h1":
        ppr = '<w:spacing w:before="220" w:after="100"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
        default = dict(size=28, bold=True)
    elif kind == "h2":
        ppr = '<w:spacing w:before="140" w:after="70"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
        default = dict(size=24, bold=True)
    elif kind == "caption":
        ppr = '<w:spacing w:before="80" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="center"/>'
        default = dict(size=21, bold=False)
    elif kind == "ref":
        ppr = '<w:spacing w:line="300" w:lineRule="auto" w:after="20"/><w:ind w:left="420" w:hanging="420"/><w:jc w:val="left"/>'
        default = dict(size=21)
    elif kind == "note":
        ppr = '<w:spacing w:before="60" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
        default = dict(size=21)
    else:
        ppr = '<w:spacing w:line="360" w:lineRule="auto" w:after="30"/><w:ind w:firstLine="480"/><w:jc w:val="both"/>'
        default = dict(size=24)
    if runs is None:
        runs = [(text, {})]
    rr = []
    for item in runs:
        if isinstance(item, str):
            txt, attrs = item, {}
        else:
            txt, attrs = item
        opts = dict(default)
        opts.update(attrs)
        rr.append(run(txt, **opts))
    return f'<w:p><w:pPr>{ppr}</w:pPr>{"".join(rr)}</w:p>'


def table(rows):
    widths = [1950, 3000, 3838]
    tblpr = (
        '<w:tblPr><w:tblW w:w="8788" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="808080"/>'
        '<w:left w:val="single" w:sz="6" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="808080"/>'
        '<w:right w:val="single" w:sz="6" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="B0B0B0"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="B0B0B0"/></w:tblBorders></w:tblPr>'
    )
    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>'
    body = []
    for ridx, row in enumerate(rows):
        cells = []
        for cidx, text in enumerate(row):
            shade = '<w:shd w:fill="D9EAF7"/>' if ridx == 0 else ''
            tcpr = f'<w:tcPr><w:tcW w:w="{widths[cidx]}" w:type="dxa"/>{shade}<w:vAlign w:val="center"/></w:tcPr>'
            cells.append(tcpr + para(text, kind="note", runs=[(text, {"bold": ridx == 0, "size": 20})]))
        body.append('<w:tr>' + ''.join(f'<w:tc>{c}</w:tc>' for c in cells) + '</w:tr>')
    return '<w:tbl>' + tblpr + grid + ''.join(body) + '</w:tbl>'


TITLE = "长上下文大语言模型 Block 稀疏检索中的动态访问预算预测与在线校准研究综述"
ABSTRACT = (
    "长上下文大语言模型在文档理解和复杂推理中的应用使 KV Cache 的存储与访问开销日益突出。"
    "现有研究已形成 Token 淘汰、量化压缩、Query-aware 稀疏检索以及 Layer/Head 级预算分配等路线，"
    "并开始探索任务感知和逐解码步动态预算。本文围绕可回取历史 KV Block 的稀疏检索场景，梳理不同方法在预算粒度、"
    "决策时点、运行时状态利用和在线反馈方面的差异，重点分析逐步预算预测与预算不足校准的研究进展。"
    "在此基础上，结合固定 Sink/Local 约束和无 Full-Attention 在线条件，讨论初始 Block 访问预算预测、预算低估风险检测及增量校准的研究空间，"
    "并总结质量、访问开销和端到端延迟等评价要点，为后续方法设计提供依据。"
)
KEYWORDS = "长上下文大语言模型；KV Cache；Block 稀疏检索；动态访问预算；在线校准"


SECTIONS = [
    ("h1", "一、前言"),
    ("body", "近年来，大语言模型在自然语言理解、代码生成、智能问答、检索增强生成和复杂推理等任务中取得了显著进展。随着模型应用从短文本交互扩展到长文档、代码仓库和多轮历史，上下文长度不断增加，部分模型和评测基准已经覆盖数十万乃至更长的输入。长上下文提升了模型整合远距离证据的能力，同时也使推理阶段的显存容量、显存带宽和端到端延迟成为实际部署的主要瓶颈。"),
    ("body", "Transformer 通过自注意力建立不同位置之间的依赖关系。自回归解码时，为避免重复计算历史 Token 的 Key 和 Value 表示，系统通常将其保存为 KV Cache；其规模不仅随序列长度线性增长，还受到网络层数、KV Head 数量和并发请求数影响。Attention Is All You Need 奠定了该计算范式，PagedAttention 则表明 KV 状态的物理分页与逻辑访问选择可以分离，为后续缓存管理和稀疏检索提供了系统基础[1-2]。"),
    ("body", "需要特别区分两类问题：缓存存储状态与当前解码步的访问状态。Token 淘汰或量化主要改变缓存中保存的数据；Block 稀疏检索则可以保留完整或较完整的历史 KV，仅在某一步加载、读取其中一部分。因而，本文所说的“访问预算”是当前解码步允许参与注意力计算的历史 KV Block 数，而不是简单的缓存容量或永久删除比例。对于可回取的历史 Block，未被当前步选中的 Block 可能仍保存在 CPU、分层内存或压缩存储中，不能将“未访问”直接表述为“已删除”。"),
    ("body", "为统一后文讨论，设可回取的历史 KV Block 总数为 M。在解码步 t，预算预测器先根据当前可因果获得的运行时状态输出初始访问预算 B_t^(0)，检索器据此选择候选 Block；若初始集合存在预算不足风险，再输出增量预算 ΔB_t，并令校准后预算为 B_t=B_t^(0)+ΔB_t。前者发生在初始检索执行前，目标是降低预算预测误差和访问开销；后者发生在初始检索边界形成后，目标是在质量风险约束下进行最小增量访问。离线 Full-Attention 仅用于构造覆盖率参考或训练标签，部署阶段不调用在线 Full-Attention Oracle。"),
    ("body", "本文以 2026 年 8 月 31 日为文献检索截止日，重点梳理 KV Cache 压缩、Query-aware/Block 稀疏检索、Layer/Head 异质性、动态预算预测和在线风险控制五条相互关联但不等价的研究路线，并据此界定研究内容 1 与研究内容 2 的边界。"),

    ("h1", "二、相关研究进展"),
    ("h2", "（一）KV Cache 压缩、淘汰与 Block 稀疏检索"),
    ("body", "早期 KV Cache 优化主要在给定容量约束下选择性保留历史 Token。H2O 观察到少量 Heavy-Hitter Token 在注意力分布中长期占据较大权重，结合 Heavy-Hitter 与近期 Token 进行淘汰；Scissorhands 利用重要性持续性假设，以历史注意力统计估计后续保留价值；StreamingLLM 则揭示序列起始位置的 Attention Sink 现象，通过固定保留 Sink 与近期窗口维持流式生成稳定性[3-5]。Oren 等在 Transformers are Multi-State RNNs 中提出 TOVA，将注意力驱动的 Token 保留纳入在线缓存更新；SnapKV 进一步利用预填充阶段观察窗口的注意力模式压缩提示上下文[6-7]。这些方法证明了历史重要性的非均匀性，但其重要性分数是代理指标，不应直接等同于 Token 对任务结果的因果贡献。"),
    ("body", "另一条路线是降低单个 KV 状态的表示成本。KIVI 针对 Key 与 Value 的分布差异设计免调参的低比特量化，KVQuant 面向极长上下文探索更低精度的 KV 表示，PM-KVQ 则进一步采用渐进式混合精度策略以适应长链式推理[9-11]。量化主要解决存储容量和数据搬运问题，本身并不回答当前解码步应访问多少 Block；在实际系统中，量化、淘汰和稀疏读取可以组合使用，但评价时需要分开报告压缩误差、访问量和内核开销。"),
    ("body", "Query-aware 稀疏检索把决策从“长期保留哪些 Token”推进到“当前 Query 需要读取哪些历史区域”。SparQ Attention 通过压缩 Query 表示降低候选筛选成本；Quest 以 Page/Block 为单位构造 Key 摘要或相关性上界，并依据当前 Query 排序后仅加载 Top-K 页面；RetrievalAttention 则使用近似检索结构从长历史中召回相关 KV[12-14]。这类方法保留了更多可回取信息，因而具有发生增量访问的可能，但候选扫描、摘要维护和数据搬运也会带来额外开销，不能只用理论稀疏率衡量收益。"),
    ("body", "系统与检索协同方向进一步扩大了 Block 稀疏访问的适用范围。ArkVale 和 ClusterKV 分别从分层组织、聚类摘要等角度降低长历史的候选定位成本；InfiniGen 将部分 KV 管理与分层存储结合，减少 GPU 常驻数据；RocketKV 采用“粗粒度永久淘汰+细粒度 Top-K 稀疏注意力”的两阶段方案，在压缩率与质量之间进行折中[15-18]。FreeKV 关注检索友好的 KV 保留，ParisKV 针对长生成中的相关性漂移提高检索鲁棒性，LouisKV 则利用语义边界和跨步局部性减少重复检索[19-21]。这些工作说明，Block 选择、跨步复用和数据布局已经成为相互耦合的系统问题。"),
    ("body", "固定 Top-K 并不是唯一的稀疏规则。Twilight 通过层级 Top-p 剪枝，使每一步保留的注意力质量达到概率阈值，从而产生随注意力分布变化的自适应稀疏度[22]。它是“自适应访问规模”的重要先例，但 Top-p 阈值作用于注意力权重筛选，尚未等同于本文以 Block 数为单位、在检索前预测总预算并在检索后执行风险校准的两阶段方案。总体而言，现有工作已经把 KV Cache 从静态数组转变为可压缩、可检索、可分层管理的数据结构，为动态访问预算研究奠定了基础。"),

    ("h2", "（二）Layer/Head 异质性与预算分配"),
    ("body", "仅在 Token 或 Block 维度做统一处理，会忽略 Transformer 内部明显的结构异质性。RazorAttention 识别具有检索功能的 Head，并对其保留更细的上下文；DuoAttention 将 Head 区分为 Retrieval Head 与 Streaming Head，分别采用不同的缓存策略[23-24]。PyramidInfer 和 PyramidKV 从层间信息流动出发，说明不同层对长上下文的依赖并不相同；ZigZagKV 则用层不确定性刻画层级压缩需求[25-27]。"),
    ("body", "HeadKV 进一步在 Head 粒度进行缓存压缩与检索推理，Ada-KV 则在给定总体资源约束下将预算自适应分配给不同 Head；LAVa 同时考虑 Layer 与 Head 的差异，进行层级缓存淘汰和动态预算分配[28-30]。这些方法的共同问题形式可以写成：在总预算约束下决定各层或各 Head 的 B_l 或 B_(l,h)，目标是最大化质量收益与资源效率。它们主要回答“预算如何在模型结构内分配”，而不一定回答“当前解码步总共需要多少可访问 Block”。"),
    ("body", "Dynamic Memory Compression 代表了连续压缩和记忆状态更新的另一种思路，与离散 Block 选择并不完全相同[31]。因此，在综述中应明确区分 Layer/Head 预算分配、Token 保留和逐步 Block 访问预算：三者可以组合，但决策变量、监督信号和系统执行位置不同。特别需要注意，Ada-KV 与后文的 AdaK 不是同一篇工作：前者是 Head-wise eviction budget allocation，后者才是以 KV Cache Budget Estimation 为主题的逐步预算分析框架。"),

    ("h2", "（三）逐解码步动态预算预测"),
    ("body", "动态预算研究试图摆脱“一次设定、全程不变”的缓存规模。DBudgetKV 的早期版本（后续发展为 ReFreeKV）使用注意力信号判断继续压缩是否可能偏离 Full-Cache 性能，并在满足停止条件时终止剪枝；它更接近自适应停止和风险信号，而不是可回取 Block 的同一步增量检索[32]。ReFreeKV 后续版本强调无需预先指定固定压缩阈值，说明输入长度和任务类型会改变合理的缓存规模。"),
    ("body", "DynamicKV 主要在预填充阶段根据任务特征和层级注意力模式调整 Token 保留量，并周期性更新前序层配置[33]。它证明任务和层之间存在预算差异，但不应被表述为每个生成 Token 都执行的在线总预算预测。与之不同，Shao 提出的 AdaK 将问题直接定义为 KV Cache Budget Estimation，分别研究注意力熵阈值、任务感知查找表和轻量策略网络，并输出每一步的预算上限或 k_t 估计[34]。AdaK 的价值在于把“给定预算后如何筛选”推进到“当前步骤需要多少预算”，但其分析式预算上限与实际 Block 检索、增量修复之间仍存在接口。"),
    ("body", "CONF-KV 根据下一 Token 概率分布构造生成置信度：不确定性较高时倾向于保留更多上下文，不确定性较低时采用更积极的压缩；其预算控制的主信号是 Next-Token 置信度，同时结合历史注意力、近期性和混合精度存储[35]。该工作说明运行时状态可以驱动预算变化，但也提示单一信号可能难以刻画历史相关性分布。综合来看，现有逐步预算方法在决策时点、输入特征和预算单位上尚未统一：有的预测保留 Token 数，有的预测层级上限，有的根据注意力质量停止压缩，只有少数工作直接面向每一步的可访问 Block 总量。"),

    ("h2", "（四）预算低估风险与在线校准"),
    ("body", "预算预测的难点不仅是平均误差，还在于低估与高估的代价不对称。低估可能使关键历史 Block 在当前步未被访问，导致注意力输出和任务质量下降；高估通常增加访存和延迟，也可能改变注意力归一化后的信息稀释。因而，预测器还需要一个在初始检索完成后判断“是否可能漏选重要 Block”的风险机制。"),
    ("body", "现有研究已经提供了若干相邻思路。RefreshKV 在生成过程中周期性恢复全上下文注意力并刷新小缓存，属于周期性校正但在线成本较高[36]；DefensiveKV 从最坏情况出发分析历史重要性稳定假设的脆弱性；ParisKV 讨论 Query—历史 KV 相关性漂移[20,37]。ReST-KV 使用层级输出重构和时空平滑缓解淘汰误差，RetroAttention 则保留轻量输出状态并回溯修正过去的注意力结果[38-39]。这些方法的修正对象和时点不同，有的修复缓存，有的修复输出，并不等价于当前步增加 Block 访问。"),
    ("body", "更接近本文在线校准目标的是 PRR 的 Predict–Reuse–Repair 框架：它预测稀疏模式、跨步复用候选，并用 online-softmax 对漏选 Block 进行增量修复[40]。Uncertainty-gated selection 依据 Top-k 截断边界的间隔和信息价值，在不确定时扩大保留集合[41]。The Risk of KV Cache Compression 则从风险角度讨论压缩失效与质量下降之间的关系[42]。此外，Adaptive Token Release 说明动态释放本身需要质量感知的停止条件[43]。这些工作分别覆盖周期刷新、跨步复用、输出回溯、漏选修复和不确定性门控，但在“固定 Sink/Local 约束、无在线 Full-Attention、当前步最小增量补取、下一步反馈”这一组合设定下，仍缺少统一建模和端到端比较。"),
    ("body", "因此，研究内容 2 不应被写成研究内容 1 的另一个预测器，而应被定义为独立的部署期控制环：初始预算和候选 Block 已经执行后，利用边界分数间隔、尾部质量、局部注意力一致性、跨步复用命中率等低成本反馈估计风险 z_t；若 z_t 超过门限，仅补取最小的 ΔB_t 个候选 Block，并通过在线 softmax 合并或局部重算修正输出，再将反馈传递到后续步骤。这里的“无 Oracle”是指部署阶段不调用 Full-Attention Oracle；离线 Oracle 仍可用于研究内容 1 的覆盖率标签、阈值校准和独立评估。"),

    ("h2", "（五）现有不足与本文切入点"),
    ("body", "第一，现有研究对“预算”的定义不统一。缓存容量、保留 Token 数、层级上限、Top-k 页面数和每步可访问 Block 数经常被放在同一表述下，导致方法之间难以直接比较。对于本文问题，应把预算单位固定为逻辑历史 Block，把 Sink/Local 作为始终可访问的约束集合，把中间历史区域作为动态候选集合，并明确预算预测发生在检索前、风险校准发生在初始检索后。"),
    ("body", "第二，运行时特征必须满足因果性和低开销约束。当前 Query 或上一时刻 Hidden State 可以在解码时获得，但不能使用尚未生成的未来信息；历史 Block 统计也不能通过每一步重新扫描全部 KV 来获得，否则特征提取成本会抵消稀疏访问收益。预测器应优先使用已维护的分数直方图、集中度/离散度、相邻步变化量和候选边界摘要，并单独核算特征维护、候选评分和数据搬运开销。"),
    ("body", "第三，预算低估需要质量感知的闭环控制。仅报告平均访问 Block 数无法反映漏选风险，至少应同时报告预算 MAE、低估率、覆盖率、风险门控的误报/漏报、增量 Block 数、修复成功率、每步延迟和端到端吞吐。若增加 Block 后重新计算注意力，还需说明采用在线 softmax 合并还是局部重算，因为新增候选会改变归一化分母。"),
    ("body", "第四，评测应覆盖不同长度、任务和生成阶段。LongBench、RULER 和 ∞Bench 可分别用于多任务长上下文、有效上下文长度和超长输入评估；SCBench 则从 KV Cache 生命周期出发，强调缓存生成、压缩、检索、加载和释放的系统成本[44-47]。这些基准有助于避免只在单一平均压缩率上得出结论。"),
    ("caption", "表1 研究内容1与研究内容2的边界"),
    ("table", [
        ["比较维度", "研究内容1：预算预测", "研究内容2：风险检测与在线校准"],
        ["决策时点", "初始 Block 检索前", "初始检索完成后、当前步输出形成前后"],
        ["主要输入", "q_t/h_{t-1}、Block 分数分布及历史统计", "边界分数间隔、尾部质量、局部一致性、跨步反馈"],
        ["主要输出", "初始预算 B_t^(0)", "风险 z_t、增量预算 ΔB_t、后续反馈状态"],
        ["Oracle 与训练", "离线 Full-Attention 覆盖率标签；可训练轻量回归/分类器", "部署不调用 Full-Attention；用规则或校准集设定风险门限"],
        ["优化目标", "在覆盖率约束下减少预测误差和初始访问量", "降低低估漏检和质量损失，控制额外访问"],
        ["核心指标", "MAE、低估率、覆盖率、预测开销", "误报/漏报、修复成功率、增量 Block 数、质量与延迟"],
    ]),
    ("body", "由表1可见，两个研究内容共享 Block 检索基础设施，但在决策时点、输入信号、输出变量、训练方式和评价指标上具有清晰差异。研究内容1解决“初始应该访问多少”的前馈预测问题；研究内容2解决“初始访问后是否需要补取以及补取多少”的反馈控制问题。二者可以在系统中串联，却不应在论文表述中合并为一个方法的两个实现模块。本文的创新空间因而应表述为对预算预测、风险门控、增量检索和跨步反馈的特定组合与系统化验证，而不是宣称相关思想此前完全不存在。"),

    ("h1", "三、总结与展望"),
    ("body", "现有 KV Cache 研究已由 Token 淘汰和量化压缩，逐步发展到 Query-aware 稀疏检索、Layer/Head 异质性建模及运行时预算管理。可回取 Block 检索避免了永久删除带来的不可恢复问题，但固定 Top-k 仍难适应不同解码步骤的相关性分布；Layer/Head 预算分配揭示了模型结构异质性，Twilight、AdaK 等工作则说明访问规模可以随运行状态变化[22,28-30,34]。与此同时，刷新、复用和漏选修复工作表明，预算决策还必须考虑预测不确定性与运行时反馈。"),
    ("body", "基于上述认识，本文拟在固定 Sink/Local 约束下，首先利用离线 Full-Attention 构造覆盖率意义上的参考 Block 预算，训练轻量模型根据可因果获得的运行时状态输出每步初始访问预算；随后在部署阶段不调用 Full-Attention Oracle、不更新模型参数，根据初始检索边界和可低成本维护的反馈信号判断预算低估风险，并进行最小增量访问和跨步反馈。研究将重点验证预算覆盖与任务质量的关系、风险门控的误报/漏报代价，以及附加评分和数据搬运开销，最终以质量—访问量—延迟的联合指标评价方法有效性。"),
    ("body", "后续工作还需进一步研究三个问题：其一，如何让预算在相邻解码步之间平滑变化，避免频繁扩缩造成内核和内存管理抖动；其二，如何把风险门控与 GPU/CPU 分层存储、Block 索引和 online-softmax 内核协同设计；其三，如何在更多模型、任务和上下文长度上检验覆盖率标签与真实任务质量的对应关系。只有同时回答算法可靠性和系统可执行性，动态访问预算才可能从离线分析转化为稳定的长上下文推理收益。"),
]


REFERENCES = [
    "[1] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems. 2017.",
    "[2] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention[C]//Proceedings of the 29th Symposium on Operating Systems Principles. 2023. DOI:10.1145/3600006.3613165.",
    "[3] Zhang Z, et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models[C]//Advances in Neural Information Processing Systems. 2023.",
    "[4] Liu Y, et al. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time[C]//Advances in Neural Information Processing Systems. 2023.",
    "[5] Xiao G, et al. Efficient Streaming Language Models with Attention Sinks[C]//International Conference on Learning Representations. 2024. arXiv:2309.17453.",
    "[6] Oren M, et al. Transformers are Multi-State RNNs[EB/OL]. arXiv:2401.06104, 2024.（TOVA）",
    "[7] Li Y, et al. SnapKV: LLM Knows What You Are Looking for Before Generation[C]//Advances in Neural Information Processing Systems. 2024.",
    "[8] Bui N, Sharma S, Lamba S, et al. Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs[C]//International Conference on Learning Representations. 2026.",
    "[9] Liu Z, et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache[C]//Proceedings of the 41st International Conference on Machine Learning. 2024. https://proceedings.mlr.press/v235/liu24bz.html.",
    "[10] Hooper C, et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization[C]//Advances in Neural Information Processing Systems. 2024. arXiv:2401.18079.",
    "[11] Liu T, et al. PM-KVQ: Progressive Mixed-Precision KV Cache Quantization for Long-CoT LLMs[C]//International Conference on Learning Representations. 2026.",
    "[12] Ribar L, et al. SparQ Attention: Bandwidth-Efficient LLM Inference[C]//International Conference on Machine Learning. 2024. arXiv:2312.04985.",
    "[13] Tang J, et al. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference[C]//International Conference on Machine Learning. 2024. arXiv:2406.10774.",
    "[14] Liu D, et al. RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval[EB/OL]. arXiv:2409.10516, 2024.",
    "[15] Chen R, et al. ArkVale: Towards Efficient Long-Context Inference with KV Cache Management[C]//Advances in Neural Information Processing Systems. 2024.",
    "[16] Liu G, et al. ClusterKV: Manipulating KV Cache in LLMs via Clustering[EB/OL]. arXiv:2412.03213, 2024.",
    "[17] Lee W, Lee J, Seo J, et al. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management[C]//Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation. 2024. arXiv:2406.19707.",
    "[18] Behnam P, Fu Y, Zhao R, et al. RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression[C]//Proceedings of the 42nd International Conference on Machine Learning. 2025, 267:3358-3392. https://proceedings.mlr.press/v267/behnam25a.html.",
    "[19] Liu G, et al. FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference[C]//International Conference on Learning Representations. 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/5137e40ee586c4f1fc3615cf8cee4f87-Abstract-Conference.html.",
    "[20] Qi Y, et al. ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs[EB/OL]. arXiv:2602.07721, 2026.",
    "[21] Wu W, Si Q, Pan X, et al. LouisKV: Efficient Long-Context Inference via Semantic-Boundary Retrieval and Temporal Reuse[C]//International Conference on Learning Representations. 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html.",
    "[22] Lin C, Tang J, Yang S, et al. Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning[EB/OL]. arXiv:2502.02770, 2025.（NeurIPS 2025 Spotlight）",
    "[23] Tang H, Lin Y, Lin J, et al. RazorAttention: Efficient KV Cache Compression Through Retrieval Heads[C]//International Conference on Learning Representations. 2025.",
    "[24] Xiao G, Tang J, Zuo J, et al. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads[C]//International Conference on Learning Representations. 2025.",
    "[25] Yang D, Han X, Gao Y, et al. PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference[A]//Findings of the Association for Computational Linguistics: ACL 2024. 2024:3258-3270. DOI:10.18653/v1/2024.findings-acl.195.",
    "[26] Cai Z, et al. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling[EB/OL]. arXiv:2406.02069, 2024.",
    "[27] Zhong M, Liu X, Zhang C, et al. ZigZagKV: Dynamic KV Cache Compression for Long-context Modeling based on Layer Uncertainty[A]//Proceedings of the 31st International Conference on Computational Linguistics. 2025:8897-8907.",
    "[28] Fu Y, Cai Z, Asi A, et al. Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning[EB/OL]. arXiv:2410.19258, 2024.",
    "[29] Feng Y, Lv J, Cao Y, et al. Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference[EB/OL]. arXiv:2407.11550, 2024. https://arxiv.org/abs/2407.11550.",
    "[30] Shen Y, Yuan S, Zhang Z, et al. LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation[A]//Findings of the Association for Computational Linguistics: EMNLP 2025. 2025:13672-13692. DOI:10.18653/v1/2025.findings-emnlp.737.",
    "[31] Nawrot P, et al. Dynamic Memory Compression[EB/OL]. arXiv:2403.09636, 2024.",
    "[32] Ni X, et al. ReFreeKV: Towards Threshold-Free KV Cache Compression[EB/OL]. arXiv:2502.16886, 2025.（早期版本题为 DBudgetKV）",
    "[33] Zhou X, Wang W, Zeng M, et al. DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs[A]//Findings of the Association for Computational Linguistics: EMNLP 2025. 2025:8042-8057. DOI:10.18653/v1/2025.findings-emnlp.426.",
    "[34] Shao T. AdaK: Adaptive KV Cache Budget Estimation Framework for Analyzing Long-Context Large Language Model Inference[J]. Frontiers in Artificial Intelligence, 2026, 9:1892739. DOI:10.3389/frai.2026.1892739.",
    "[35] Li Y, Miao Y. CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM Inference[EB/OL]. arXiv:2605.24786, 2026.",
    "[36] Xu F, Goyal T, Choi E. RefreshKV: Updating KV Cache during Long-Context Generation[EB/OL]. arXiv:2411.05787, 2024.",
    "[37] Feng Y, Guo H, Lv J, et al. DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference[C]//International Conference on Learning Representations. 2026. arXiv:2510.13334.",
    "[38] An Y, Lu C, Zhu K, et al. ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing[C]//International Conference on Learning Representations. 2026. arXiv:2605.08840.",
    "[39] Choi S, Kang B, Jo D, et al. Retrospective Sparse Attention for Efficient Long-Context Generation[EB/OL]. arXiv:2508.09001, 2025.",
    "[40] Wang T, Rattihalli G, Dhakal A, et al. Predict, Reuse, and Repair: Accelerating Dynamic Sparse Attention for Long-Context LLM Decoding[EB/OL]. arXiv:2606.30389, 2026.",
    "[41] Rossi T. Uncertainty-Gated Selection for Block-Sparse Attention[EB/OL]. arXiv:2607.07724, 2026.",
    "[42] Haverbeck L, et al. The Risk of KV Cache Compression[EB/OL]. arXiv:2607.01520, 2026.",
    "[43] Zhang C, et al. Efficient Sparse Attention Needs Adaptive Token Release[A]//Findings of the Association for Computational Linguistics: ACL 2024. 2024. https://aclanthology.org/2024.findings-acl.837/.",
    "[44] Bai Y, et al. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding[A]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024:3111-3132. https://aclanthology.org/2024.acl-long.172/.",
    "[45] Hsieh C-P, et al. RULER: What's the Real Context Size of Your Long-Context Language Models?[EB/OL]. arXiv:2404.06654, 2024.",
    "[46] Zhang X, et al. ∞BENCH: Extending Long Context Evaluation Beyond 100K Tokens[A]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024. https://aclanthology.org/2024.acl-long.814/.",
    "[47] Li Yucheng, Jiang Huiqiang, Wu Qianhui, et al. SCBench: A KV Cache-Centric Analysis of Long-Context Methods[C]//International Conference on Learning Representations. 2025. https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/.",
]


def build_document() -> str:
    parts = []
    parts.append(para(TITLE, kind="title"))
    for line in ["研究生：陈穗章", "指导教师：杜卿", "学号：202521045341", "院（系）：软件学院", "专业：软件工程", "文献检索截止日期：2026年8月31日"]:
        parts.append(para(line, kind="meta"))
    parts.append(para("摘要", kind="label"))
    parts.append(para(ABSTRACT, kind="body"))
    parts.append(para("关键词：" + KEYWORDS, kind="body"))
    for kind, text in SECTIONS:
        if kind == "table":
            parts.append(table(text))
        else:
            parts.append(para(text, kind=kind))
    parts.append(para("参考文献", kind="h1"))
    for ref in REFERENCES:
        parts.append(para(ref, kind="ref"))
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1418" w:right="1418" '
        'w:bottom="1418" w:left="1701" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:docGrid w:type="linesAndChars" w:linePitch="360"/></w:sectPr>'
    )
    return '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + ''.join(parts) + sect + '</w:body></w:document>'


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:rPrDefault>
 <w:pPrDefault><w:pPr><w:widowControl/></w:pPr></w:pPrDefault></w:docDefaults>
 <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
</w:styles>'''


def settings_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>'''


def core_xml() -> str:
    now = date.today().isoformat()
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>{x(TITLE)}</dc:title><dc:creator>陈穗章</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{now}T00:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}T00:00:00Z</dcterms:modified></cp:coreProperties>'''


def write_docx():
    document = build_document()
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/></Relationships>'''
    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>'''
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/settings.xml", settings_xml())
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml)


def write_markdown():
    lines = [f"# {TITLE}", "", "研究生：陈穗章  ", "指导教师：杜卿  ", "学号：202521045341  ", "院（系）：软件学院  ", "专业：软件工程  ", "文献检索截止日期：2026年8月31日", "", "## 摘要", ABSTRACT, "", "**关键词：**" + KEYWORDS, ""]
    for kind, text in SECTIONS:
        if kind == "h1": lines.extend([f"## {text}", ""])
        elif kind == "h2": lines.extend([f"### {text}", ""])
        elif kind == "caption": lines.extend([f"**{text}**", ""])
        elif kind == "table":
            for ridx, row in enumerate(text):
                lines.append("| " + " | ".join(row) + " |")
                if ridx == 0: lines.append("| " + " | ".join(["---"] * len(row)) + " |")
            lines.append("")
        else: lines.extend([text, ""])
    lines.extend(["## 参考文献", ""] + [ref + "  " for ref in REFERENCES])
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_docx()
    write_markdown()
    print(f"Wrote {OUT} and {MD_OUT}; references={len(REFERENCES)}")
