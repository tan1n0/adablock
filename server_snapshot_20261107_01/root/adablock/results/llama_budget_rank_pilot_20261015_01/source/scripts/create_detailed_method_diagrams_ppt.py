from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究内容一二_详细方法图.pptx"

NAVY = RGBColor(32, 31, 128)
INDIGO = RGBColor(58, 52, 194)
BLUE = RGBColor(70, 102, 180)
LIGHT_BLUE = RGBColor(225, 235, 250)
PURPLE = RGBColor(226, 222, 246)
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


def text_box(slide, x, y, w, h, text, size=14, color=BLACK, bold=False,
             align=PP_ALIGN.CENTER, font="Microsoft YaHei"):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.01)
    tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return s


def rect(slide, x, y, w, h, text="", fill=WHITE, line=BLUE, size=13,
         bold=False, color=BLACK, radius=False, width=1.1):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(width)
    if text:
        tf = s.text_frame
        tf.clear(); tf.word_wrap = True
        tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def line(slide, x1, y1, x2, y2, color=NAVY, width=1.6, arrow=False, dash=False):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow:
        c.line.end_arrowhead = True
    if dash:
        c.line.dash_style = 4
    return c


def top_style(slide, title):
    rect(slide, 0, 0.00, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 0.00, 6.98, 0.25, fill=INDIGO, line=INDIGO)
    rect(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY)
    text_box(slide, 0.23, 0.31, 12.7, 0.50, title, size=27, color=WHITE,
             bold=True, align=PP_ALIGN.LEFT)


def bottom_style(slide):
    rect(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO)


def bullet(slide, y, segments):
    # segments: list[(text, color, bold)]
    x = 0.85
    rect(slide, 0.55, y + 0.08, 0.13, 0.13, fill=WHITE, line=BLACK, width=1.5)
    for txt, color, bold, width in segments:
        text_box(slide, x, y, width, 0.33, txt, size=14, color=color,
                 bold=bold, align=PP_ALIGN.LEFT)
        x += width


def cell_strip(slide, x, y, values, fills, total_w, h=0.27, size=8):
    cell_w = total_w / len(values)
    for i, value in enumerate(values):
        rect(slide, x + i * cell_w, y, cell_w, h, str(value), fill=fills[i],
             line=NAVY, size=size, width=0.7)
    return cell_w


def caption(slide, x, y, w, text):
    text_box(slide, x, y, w, 0.38, text, size=12, bold=True, align=PP_ALIGN.LEFT)


def innovation(slide, text):
    rect(slide, 0.45, 6.58, 12.43, 0.56, fill=PURPLE, line=PURPLE)
    text_box(slide, 0.60, 6.64, 1.05, 0.40, "创新点：", size=20, color=BLUE,
             bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.55, 6.65, 11.05, 0.38, text, size=14, color=BLACK,
             bold=False, align=PP_ALIGN.LEFT)


def slide_content1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容1：基于运行时状态的逐解码步动态 Block 访问预算预测")

    # Panel separators
    line(slide, 4.20, 1.05, 4.20, 4.78, color=NAVY, width=1.4)
    line(slide, 8.12, 1.05, 8.12, 4.78, color=NAVY, width=1.4)

    # (1) Approximate oracle labels
    text_box(slide, 0.48, 1.00, 3.42, 0.30, "离线阶段：构造采样注意力预算标签", size=15, bold=True)
    text_box(slide, 0.45, 1.42, 0.65, 0.25, "历史序列", size=10, bold=True)
    vals = ["S1", "S2", "…", "M1", "M2", "…", "Mₙ", "L1", "L2", "L3", "L4"]
    fills = [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY,
             LIGHT_GRAY, LIGHT_GREEN, LIGHT_GREEN, LIGHT_GREEN, LIGHT_GREEN]
    cell_strip(slide, 1.08, 1.40, vals, fills, 2.68, h=0.30, size=8)
    text_box(slide, 1.04, 1.72, 0.78, 0.22, "Sink", size=8, color=BLUE, bold=True)
    text_box(slide, 2.03, 1.72, 0.92, 0.22, "候选 Middle", size=8, color=GRAY, bold=True)
    text_box(slide, 3.15, 1.72, 0.62, 0.22, "Local", size=8, color=GREEN, bold=True)

    # sampled layers and positions
    text_box(slide, 0.48, 2.10, 0.78, 0.25, "采样位置", size=10, bold=True)
    for j, pos in enumerate(["t", "t+64", "t+128"]):
        rect(slide, 1.25 + j * 0.82, 2.08, 0.68, 0.28, pos, fill=LIGHT_ORANGE,
             line=ORANGE, size=9)
    text_box(slide, 0.48, 2.52, 0.78, 0.25, "选定层", size=10, bold=True)
    for j, layer_name in enumerate(["Layer 12", "Layer 20", "Layer 27"]):
        rect(slide, 1.25 + j * 0.82, 2.50, 0.68, 0.28, layer_name, fill=LIGHT_BLUE,
             line=BLUE, size=8)
    line(slide, 2.28, 2.82, 2.28, 3.12, arrow=True)
    rect(slide, 0.75, 3.17, 3.00, 0.44, "Token 注意力按 Block 聚合", fill=WHITE,
         line=BLUE, size=11, bold=True, radius=True)
    line(slide, 2.25, 3.64, 2.25, 3.90, arrow=True)

    # sorted block mass mini bars
    text_box(slide, 0.50, 3.95, 0.95, 0.24, "Block 质量", size=10, bold=True)
    heights = [0.48, 0.37, 0.28, 0.20, 0.14, 0.09]
    for i, hh in enumerate(heights):
        rect(slide, 1.44 + i * 0.33, 4.43 - hh, 0.24, hh, fill=BLUE if i < 3 else MID_GRAY,
             line=NAVY, width=0.6)
    line(slide, 1.38, 4.44, 3.46, 4.44, color=BLACK, width=0.8)
    text_box(slide, 1.27, 4.46, 2.36, 0.20, "降序累积至覆盖阈值 τ → K*", size=9, color=ORANGE, bold=True)
    caption(slide, 0.55, 4.58, 3.35, "（1）获得预算预测的近似 Oracle 标签")

    # (2) Budget prediction
    text_box(slide, 4.42, 1.00, 3.48, 0.30, "在线阶段：轻量 Policy MLP 预测预算", size=15, bold=True)
    rect(slide, 4.52, 1.48, 1.18, 0.52, "hₜ₋₁\n因果状态", fill=LIGHT_BLUE, line=BLUE, size=11, bold=True)
    rect(slide, 4.52, 2.18, 1.18, 0.52, "query drift\n状态变化", fill=LIGHT_BLUE, line=BLUE, size=10, bold=True)
    rect(slide, 4.52, 2.88, 1.18, 0.72, "得分统计\nmax / mean /\nentropy / tail", fill=LIGHT_ORANGE, line=ORANGE, size=9, bold=True)
    for yy in [1.74, 2.44, 3.24]:
        line(slide, 5.72, yy, 6.15, 2.52, arrow=True)
    rect(slide, 6.18, 1.82, 0.92, 1.42, "Policy\nMLP\n\n冻结参数", fill=PURPLE,
         line=NAVY, size=13, bold=True, radius=True)
    line(slide, 7.12, 2.52, 7.48, 2.52, arrow=True)
    rect(slide, 7.50, 1.54, 0.46, 1.95, "0\n4\n8\n16\n32", fill=WHITE,
         line=NAVY, size=9, bold=True)
    text_box(slide, 7.37, 3.55, 0.72, 0.44, "预算桶\nKₜ", size=10, color=RED, bold=True)

    # hidden summary relation
    text_box(slide, 4.50, 3.92, 1.25, 0.28, "Block Hidden", size=9, bold=True)
    vals2 = ["h1", "h2", "…", "h16"]
    cell_strip(slide, 4.54, 4.20, vals2, [LIGHT_GRAY] * 4, 1.18, h=0.27, size=7)
    line(slide, 5.76, 4.34, 6.14, 4.34, arrow=True)
    rect(slide, 6.18, 4.10, 0.78, 0.46, "均值摘要\nh̄_b", fill=LIGHT_BLUE, line=BLUE, size=9)
    line(slide, 6.98, 4.34, 7.32, 4.34, arrow=True)
    rect(slide, 7.36, 4.10, 0.55, 0.46, "统计\n特征", fill=LIGHT_ORANGE, line=ORANGE, size=9)
    caption(slide, 4.48, 4.58, 3.35, "（2）预测当前解码步需要访问多少 Block")

    # (3) Per-layer/head selection
    text_box(slide, 8.34, 1.00, 4.55, 0.30, "实际检索：每层、每个 KV Head 独立选块", size=15, bold=True)
    text_box(slide, 8.28, 1.47, 0.72, 0.24, "当前 Query", size=9, bold=True)
    rect(slide, 9.00, 1.42, 0.55, 0.34, "qₜ", fill=LIGHT_ORANGE, line=ORANGE, size=12, bold=True)
    line(slide, 9.58, 1.59, 10.00, 1.59, arrow=True)
    text_box(slide, 10.00, 1.34, 2.55, 0.46, "与各层/头的 Post-RoPE\nKey Block centroid 计算余弦得分", size=9, color=GRAY, bold=True)

    y0 = 2.10
    selections = [[0, 2, 5, 7], [0, 3, 4, 7], [0, 1, 6, 7], [0, 2, 4, 7]]
    labels = ["L1-H1", "L1-H2", "L2-H1", "L2-H2"]
    for row, (lab, picks) in enumerate(zip(labels, selections)):
        yy = y0 + row * 0.50
        text_box(slide, 8.34, yy, 0.70, 0.26, lab, size=8, bold=True)
        values = ["S", "1", "2", "3", "4", "5", "6", "L"]
        fills3 = []
        for i in range(8):
            if i in picks:
                fills3.append(LIGHT_GREEN if i in {0, 7} else LIGHT_ORANGE)
            else:
                fills3.append(WHITE)
        cell_strip(slide, 9.03, yy, values, fills3, 2.72, h=0.28, size=7)
        text_box(slide, 11.83, yy, 0.80, 0.28, "Top-Kₜ", size=8, color=RED, bold=True)

    line(slide, 10.40, 4.11, 10.40, 4.34, arrow=True)
    rect(slide, 9.25, 4.35, 2.35, 0.34, "Gather K/V → Sparse Attention", fill=LIGHT_GREEN,
         line=GREEN, size=10, bold=True, radius=True)
    caption(slide, 8.38, 4.72, 4.30, "（3）预算决定数量，Query–Key 得分决定位置")

    # Explanations
    bullet(slide, 4.97, [("离线采用", BLACK, False, 0.75), ("采样重算注意力", RED, True, 1.65),
                         ("构造预算标签，避免保存完整注意力矩阵。", BLACK, False, 4.2)])
    bullet(slide, 5.38, [("在线根据因果状态和 Block 得分统计预测", BLACK, False, 3.45),
                         ("当前步骤的动态预算 Kₜ", RED, True, 2.1), ("。", BLACK, False, 0.2)])
    bullet(slide, 5.79, [("在固定 Sink 与 Local 的基础上，使用", BLACK, False, 3.45),
                         ("当前 Query 分层分头选择 Middle Block", RED, True, 3.5), ("。", BLACK, False, 0.2)])
    innovation(slide, "将“选择哪些 Block”与“访问多少 Block”解耦，使访问预算能够随解码状态变化，并保留分层分头检索的结构差异。")
    bottom_style(slide)


def slide_content2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：面向预算预测误差的运行时动态校准机制")
    line(slide, 4.15, 1.05, 4.15, 4.78, color=NAVY, width=1.4)
    line(slide, 8.30, 1.05, 8.30, 4.78, color=NAVY, width=1.4)

    # (1) score distribution
    text_box(slide, 0.46, 1.00, 3.45, 0.30, "检索反馈：判断预算不足风险", size=15, bold=True)
    text_box(slide, 0.62, 1.42, 3.10, 0.35, "候选 Middle Block 相关性分布 pₜ,b", size=11, bold=True)
    heights = [0.92, 0.76, 0.61, 0.49, 0.41, 0.34, 0.29, 0.25, 0.22, 0.20, 0.18, 0.16]
    for i, hh in enumerate(heights):
        fill = ORANGE if i >= 9 else BLUE
        rect(slide, 0.78 + i * 0.24, 3.13 - hh, 0.18, hh, fill=fill, line=fill, width=0.5)
    line(slide, 0.68, 3.14, 3.73, 3.14, color=BLACK, width=0.9)
    text_box(slide, 0.71, 3.18, 2.10, 0.25, "高分 Block", size=9, color=BLUE, bold=True)
    text_box(slide, 2.68, 3.18, 1.05, 0.25, "尾部 25%", size=9, color=ORANGE, bold=True)
    rect(slide, 0.74, 3.62, 3.05, 0.60, "尾部质量  rₜ = Σ pₜ,b  （b ∈ Tail₂₅%）", fill=LIGHT_ORANGE,
         line=ORANGE, size=12, bold=True, radius=True)
    text_box(slide, 0.72, 4.28, 3.08, 0.32, "rₜ 较大：得分分布较平缓，预算不足风险较高", size=10,
             color=RED, bold=True)
    caption(slide, 0.55, 4.62, 3.35, "（1）复用已有 Block 得分构造低成本反馈")

    # (2) EMA temporal calibration
    text_box(slide, 4.36, 1.00, 3.70, 0.30, "跨步骤状态：抑制单步波动", size=15, bold=True)
    x_positions = [4.72, 5.85, 6.98]
    for i, (xx, tt) in enumerate(zip(x_positions, ["t−1", "t", "t+1"])):
        rect(slide, xx, 1.55, 0.72, 0.38, tt, fill=LIGHT_BLUE, line=BLUE, size=11, bold=True)
        if i < 2:
            line(slide, xx + 0.75, 1.74, xx + 1.09, 1.74, arrow=True)
    rect(slide, 4.48, 2.20, 3.47, 0.60, "r̄ₜ = β r̄ₜ₋₁ + (1−β) rₜ", fill=WHITE, line=BLUE,
         size=14, bold=True, radius=True)
    line(slide, 6.22, 2.84, 6.22, 3.12, arrow=True)
    rect(slide, 4.48, 3.16, 3.47, 0.72,
         "Δₜ = clip[ Δₜ₋₁ + η(r̄ₜ−r₀),\n−Δmax, Δmax ]", fill=PURPLE, line=NAVY,
         size=12, bold=True, radius=True)
    text_box(slide, 4.62, 4.04, 3.20, 0.42, "只维护少量标量状态\n不反向传播、不更新 Policy MLP 参数", size=11,
             color=GRAY, bold=True)
    caption(slide, 4.38, 4.62, 3.60, "（2）利用指数滑动平均形成稳定校准量")

    # (3) logit adjustment
    text_box(slide, 8.50, 1.00, 4.35, 0.30, "预算校准：调整预算桶 Logits", size=15, bold=True)
    text_box(slide, 8.52, 1.43, 1.32, 0.24, "原始预算 Logits", size=10, bold=True)
    buckets = ["0", "4", "8", "16", "32"]
    original = [0.25, 0.32, 0.42, 0.55, 0.48]
    adjusted = [0.25, 0.38, 0.54, 0.76, 0.80]
    for i, (b, hh) in enumerate(zip(buckets, original)):
        rect(slide, 8.56 + i * 0.34, 2.47 - hh, 0.24, hh, fill=MID_GRAY, line=GRAY, width=0.5)
        text_box(slide, 8.54 + i * 0.34, 2.49, 0.28, 0.20, b, size=7)
    line(slide, 8.48, 2.48, 10.28, 2.48, color=BLACK, width=0.8)
    line(slide, 10.38, 2.12, 10.90, 2.12, arrow=True, color=ORANGE, width=2.0)
    text_box(slide, 10.28, 1.65, 0.70, 0.36, "+ Δₜ·rank", size=9, color=ORANGE, bold=True)
    text_box(slide, 11.02, 1.43, 1.40, 0.24, "校准后预算 Logits", size=10, bold=True)
    for i, (b, hh) in enumerate(zip(buckets, adjusted)):
        fill = ORANGE if hh == max(adjusted) else BLUE
        rect(slide, 11.08 + i * 0.34, 2.47 - hh, 0.24, hh, fill=fill, line=fill, width=0.5)
        text_box(slide, 11.06 + i * 0.34, 2.49, 0.28, 0.20, b, size=7)
    line(slide, 11.00, 2.48, 12.80, 2.48, color=BLACK, width=0.8)

    line(slide, 10.66, 2.80, 10.66, 3.10, arrow=True)
    rect(slide, 9.13, 3.16, 3.04, 0.54, "选择校准后的预算 K̃ₜ", fill=LIGHT_ORANGE,
         line=ORANGE, size=14, bold=True, radius=True)
    line(slide, 10.66, 3.73, 10.66, 4.00, arrow=True)
    rect(slide, 8.92, 4.05, 3.48, 0.52, "固定 Sink / Local + Top-K̃ₜ Middle", fill=LIGHT_GREEN,
         line=GREEN, size=12, bold=True, radius=True)
    caption(slide, 8.54, 4.62, 4.10, "（3）风险较高时，提高较大预算桶的选择倾向")

    bullet(slide, 4.97, [("利用已有的", BLACK, False, 0.85), ("候选 Block 得分尾部质量", RED, True, 2.15),
                         ("估计当前预算不足风险，不计算完整注意力。", BLACK, False, 4.5)])
    bullet(slide, 5.38, [("通过", BLACK, False, 0.48), ("指数滑动平均与有界校准量", RED, True, 2.75),
                         ("在相邻解码步骤间传递反馈，降低分数波动。", BLACK, False, 4.5)])
    bullet(slide, 5.79, [("校准量仅作用于预算桶 Logits，", BLACK, False, 2.65),
                         ("保持 MLP 参数冻结", RED, True, 1.75),
                         ("，额外状态和计算开销较低。", BLACK, False, 2.8)])
    innovation(slide, "在缺少在线注意力真值的条件下，利用检索分布形成轻量闭环，使预算预测能够根据运行状态进行自适应修正。")
    bottom_style(slide)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid)
        del prs.slides._sldIdLst[0]
    slide_content1(prs)
    slide_content2(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
