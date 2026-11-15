from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究背景与国内外研究现状.pptx"
NAVY = RGBColor(32, 31, 128)
INDIGO = RGBColor(58, 52, 194)
BLUE = RGBColor(70, 102, 180)
LIGHT_BLUE = RGBColor(225, 235, 250)
PURPLE = RGBColor(226, 222, 246)
LIGHT_PURPLE = RGBColor(239, 235, 249)
ORANGE = RGBColor(239, 101, 50)
LIGHT_ORANGE = RGBColor(253, 232, 219)
GREEN = RGBColor(77, 143, 108)
LIGHT_GREEN = RGBColor(224, 242, 231)
GRAY = RGBColor(105, 105, 112)
LIGHT_GRAY = RGBColor(242, 243, 247)
MID_GRAY = RGBColor(205, 209, 219)
BLACK = RGBColor(15, 15, 20)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(225, 38, 38)


def txt(slide, x, y, w, h, text, size=15, color=BLACK, bold=False,
        align=PP_ALIGN.CENTER):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.01); tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def box(slide, x, y, w, h, text="", fill=WHITE, line=BLUE, size=14,
        bold=False, color=BLACK, radius=True, width=1.1):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(width)
    if text:
        tf = s.text_frame; tf.clear(); tf.word_wrap = True
        tf.margin_left = Inches(0.06); tf.margin_right = Inches(0.06)
        tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def connector(slide, x1, y1, x2, y2, color=NAVY, width=1.5, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow: c.line.end_arrowhead = True
    return c


def title(slide, text, right=""):
    box(slide, 0, 0, 6.35, 0.25, fill=BLACK, line=BLACK, radius=False)
    box(slide, 6.35, 0, 6.98, 0.25, fill=INDIGO, line=INDIGO, radius=False)
    box(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY, radius=False)
    txt(slide, 0.25, 0.30, 10.5, 0.52, text, size=26, color=WHITE, bold=True, align=PP_ALIGN.LEFT)
    if right:
        txt(slide, 10.5, 0.38, 2.55, 0.26, right, size=11, color=WHITE, align=PP_ALIGN.RIGHT)


def footer(slide, text):
    txt(slide, 0.50, 7.08, 12.25, 0.20, text, size=9, color=GRAY, align=PP_ALIGN.RIGHT)
    box(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK, radius=False)
    box(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO, radius=False)


def bullet(slide, x, y, text, width=5.8, color=BLACK, bullet_color=NAVY, size=14):
    box(slide, x, y + 0.08, 0.13, 0.13, fill=WHITE, line=bullet_color, radius=False, width=1.4)
    txt(slide, x + 0.28, y, width, 0.34, text, size=size, color=color, align=PP_ALIGN.LEFT)


def slide_background(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "一、研究背景与工程价值", "长上下文大语言模型推理")
    txt(s, 0.52, 1.01, 12.2, 0.35, "长上下文能力正在进入真实应用，但 KV Cache 访问逐渐成为推理瓶颈。", size=18, bold=True, align=PP_ALIGN.LEFT)

    # Applications row
    txt(s, 0.55, 1.45, 3.35, 0.28, "典型应用场景", size=16, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
    apps = [("企业知识库\nRAG 问答", LIGHT_BLUE, BLUE), ("长文档\n理解与摘要", LIGHT_GREEN, GREEN),
            ("代码仓库\n分析与智能体", LIGHT_ORANGE, ORANGE), ("多轮对话\n与文档办公", PURPLE, NAVY)]
    for i, (t, fill, ln) in enumerate(apps):
        box(s, 0.60 + i * 1.83, 1.83, 1.57, 0.70, t, fill=fill, line=ln, size=14, bold=True)

    # Bottleneck center
    connector(s, 6.65, 2.58, 7.15, 2.58, color=ORANGE, width=2.0, arrow=True)
    box(s, 4.00, 2.82, 2.40, 1.12, "KV Cache\n随上下文长度增长", fill=LIGHT_ORANGE, line=ORANGE, size=18, bold=True)
    box(s, 7.20, 2.05, 2.65, 1.10, "每个解码步\n反复读取历史 K/V", fill=LIGHT_BLUE, line=BLUE, size=17, bold=True)
    box(s, 7.20, 3.50, 2.65, 1.10, "显存带宽压力\nAttention 计算开销", fill=LIGHT_ORANGE, line=ORANGE, size=17, bold=True)
    connector(s, 8.52, 3.15, 8.52, 3.46, color=ORANGE, width=1.8, arrow=True)
    txt(s, 10.20, 2.20, 2.55, 1.0, "输入越长、生成越长，\n访问开销持续累积。", size=17, color=RED, bold=True, align=PP_ALIGN.LEFT)

    # Fixed topk problem and target
    box(s, 0.60, 4.25, 5.75, 1.25, fill=LIGHT_GRAY, line=MID_GRAY, radius=True)
    txt(s, 0.82, 4.38, 5.3, 0.25, "固定 Top-K 的工程问题", size=16, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
    bullet(s, 0.84, 4.78, "预算过大：简单解码步骤读取大量冗余 KV", width=5.1, size=13)
    bullet(s, 0.84, 5.13, "预算过小：复杂步骤可能漏掉远距离关键信息", width=5.1, size=13)
    box(s, 6.75, 4.75, 5.75, 1.00, "研究目标\n根据解码状态动态控制历史 Block 访问规模", fill=LIGHT_GREEN, line=GREEN, size=19, bold=True, radius=True)
    connector(s, 6.40, 4.88, 6.72, 5.18, color=GREEN, width=2.0, arrow=True)
    txt(s, 0.62, 5.88, 12.0, 0.36, "工程收益：减少不必要的 KV 读取和 Attention 计算，在任务性能、延迟与吞吐量之间取得更好的平衡。", size=15, color=BLACK, bold=True, align=PP_ALIGN.LEFT)
    footer(s, "研究背景：长上下文应用需求增长 → KV Cache 访问成本上升 → 需要动态的 Block 访问控制。")


def slide_status(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "二、国内外研究现状", "KV Cache 优化技术路线")
    txt(s, 0.52, 1.00, 12.2, 0.34, "现有研究已形成从缓存保留到稀疏访问、从 Token 到 Layer/Head 的多条技术路线。", size=18, bold=True, align=PP_ALIGN.LEFT)

    # four route cards
    cards = [
        (0.55, 1.60, 2.95, 1.58, "缓存驱逐与保留", "H2O · Scissorhands\nStreamingLLM · SnapKV", "识别长期重要 Token，\n减少缓存保存规模。", LIGHT_BLUE, BLUE),
        (3.75, 1.60, 2.95, 1.58, "Block / Page 稀疏检索", "Quest · RocketKV\nParisKV · FreeKV", "按 Query 选择部分历史\nBlock 参与 Attention。", LIGHT_GREEN, GREEN),
        (6.95, 1.60, 2.95, 1.58, "KV 表示压缩", "KIVI · KVQuant\nPM-KVQ · Transform Coding", "量化或编码单个 KV，\n降低存储与传输成本。", LIGHT_ORANGE, ORANGE),
        (10.15, 1.60, 2.65, 1.58, "Layer / Head 资源分配", "PyramidKV · Ada-KV\nLAVa · DuoAttention", "利用层和头的功能差异，\n实施非均匀资源配置。", PURPLE, NAVY),
    ]
    for x, y, w, h, head, refs, desc, fill, ln in cards:
        box(s, x, y, w, h, fill=fill, line=ln, radius=True)
        txt(s, x + 0.10, y + 0.10, w - 0.20, 0.30, head, size=16, color=ln, bold=True)
        txt(s, x + 0.10, y + 0.50, w - 0.20, 0.42, refs, size=11, color=BLACK, bold=True)
        txt(s, x + 0.10, y + 1.02, w - 0.20, 0.40, desc, size=11, color=GRAY, bold=True)

    # Evolution arrow
    txt(s, 0.55, 3.48, 3.2, 0.25, "研究发展趋势", size=16, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
    connector(s, 0.80, 4.00, 12.45, 4.00, color=NAVY, width=2.0, arrow=True)
    stages = [(0.90, "固定缓存\nToken 保留", LIGHT_BLUE, BLUE), (3.40, "固定稀疏\nTop-K 访问", LIGHT_GREEN, GREEN),
              (5.95, "结构感知\nLayer / Head", PURPLE, NAVY), (8.55, "输入自适应\nTop-p / 动态预算", LIGHT_ORANGE, ORANGE),
              (11.15, "运行时\n闭环控制", LIGHT_GREEN, GREEN)]
    for x, t, fill, ln in stages:
        box(s, x, 3.60, 1.75, 0.78, t, fill=fill, line=ln, size=13, bold=True)

    # Literature summary
    box(s, 0.60, 4.65, 5.95, 1.42, fill=LIGHT_GRAY, line=MID_GRAY, radius=True)
    txt(s, 0.82, 4.78, 5.5, 0.25, "已有成果", size=16, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
    bullet(s, 0.84, 5.16, "证明完整保留全部历史 KV 并非始终必要", width=5.3, size=13)
    bullet(s, 0.84, 5.52, "发现不同 Token、Layer、Head 的上下文依赖存在差异", width=5.3, size=13)
    box(s, 6.85, 4.65, 5.95, 1.42, fill=LIGHT_ORANGE, line=ORANGE, radius=True)
    txt(s, 7.08, 4.78, 5.5, 0.25, "与本课题最直接相关", size=16, color=ORANGE, bold=True, align=PP_ALIGN.LEFT)
    bullet(s, 7.10, 5.16, "Block 稀疏检索减少每步实际读取范围", width=5.3, size=13)
    bullet(s, 7.10, 5.52, "动态预算方法开始关注不同步骤的访问需求差异", width=5.3, size=13)
    footer(s, "代表性工作：H2O/StreamingLLM、Quest/RocketKV、KIVI/KVQuant、PyramidKV/Ada-KV、Twilight/AdaK 等。")


def slide_gap(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "三、现有不足与本课题切入点", "研究问题定位")
    txt(s, 0.52, 1.00, 12.2, 0.34, "现有方法已经能够选择重要 Block，但“每一步访问多少 Block”以及“预算不合适时如何调整”仍有研究空间。", size=17, bold=True, align=PP_ALIGN.LEFT)

    # Problem to solution mapping
    headers = [(0.62, "现有问题", LIGHT_ORANGE, ORANGE), (4.55, "本课题方法", LIGHT_BLUE, BLUE), (8.53, "验证指标", LIGHT_GREEN, GREEN)]
    for x, h, fill, ln in headers:
        box(s, x, 1.60, 3.45, 0.48, h, fill=fill, line=ln, size=18, color=ln, bold=True)
    rows = [
        ("固定 Top-K\n无法适应不同解码步", "采样注意力标签\n训练逐步预算预测器", "任务得分\n预算误差"),
        ("全局选块\n忽略层/头差异", "每层、每个 KV Head\n独立 Query–Key 选块", "Block 召回\n平均访问 Token"),
        ("预算偏小或偏大\n缺少在线修正", "尾部压力决定方向\n边界间隔决定门控", "动态调整效果\n稳定性与延迟"),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = 2.25 + i * 0.98
        box(s, 0.62, y, 3.45, 0.78, a, fill=WHITE, line=MID_GRAY, size=15, bold=True)
        connector(s, 4.13, y + 0.39, 4.48, y + 0.39, color=NAVY, width=1.6, arrow=True)
        box(s, 4.55, y, 3.45, 0.78, b, fill=WHITE, line=BLUE, size=15, bold=True)
        connector(s, 8.05, y + 0.39, 8.42, y + 0.39, color=NAVY, width=1.6, arrow=True)
        box(s, 8.53, y, 3.45, 0.78, c, fill=WHITE, line=GREEN, size=15, bold=True)

    # Overall method pipeline
    txt(s, 0.62, 5.42, 3.0, 0.28, "本课题总体路线", size=16, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
    steps = [(0.68, "采样注意力\n构造近似标签", LIGHT_PURPLE, NAVY),
             (3.14, "因果状态 +\nBlock 统计 → 预算", LIGHT_BLUE, BLUE),
             (5.60, "Layer/Head\nQuery–Key 选块", LIGHT_GREEN, GREEN),
             (8.06, "尾部压力 +\n边界间隔反馈", LIGHT_ORANGE, ORANGE),
             (10.52, "连续预算\n运行时控制", PURPLE, NAVY)]
    for i, (x, t, fill, ln) in enumerate(steps):
        box(s, x, 5.82, 1.85, 0.62, t, fill=fill, line=ln, size=13, bold=True)
        if i < len(steps) - 1:
            connector(s, x + 1.88, 6.13, x + 2.38, 6.13, color=NAVY, width=1.5, arrow=True)
    box(s, 0.62, 6.62, 12.1, 0.46, "研究定位：在现有 Block Scoring 基础上，研究可解释、可反馈、与初始预算来源无关的运行时访问控制。", fill=PURPLE, line=PURPLE, size=14, bold=True)
    footer(s, "本课题重点优化访问预算与运行时控制，不改变大语言模型主体参数。")


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    slide_background(prs); slide_status(prs); slide_gap(prs)
    prs.save(OUT); print(OUT)


if __name__ == "__main__":
    main()
