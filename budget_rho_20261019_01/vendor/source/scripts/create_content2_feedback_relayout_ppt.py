from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究内容2_反馈信号详细方法图_重新排版.pptx"
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
MID_GRAY = RGBColor(190, 194, 205)
BLACK = RGBColor(15, 15, 20)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(225, 38, 38)


def text_box(slide, x, y, w, h, text, size=12, color=BLACK, bold=False,
             align=PP_ALIGN.CENTER):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.025); tf.margin_right = Inches(0.025)
    tf.margin_top = Inches(0.005); tf.margin_bottom = Inches(0.005)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def rect(slide, x, y, w, h, text="", fill=WHITE, line=BLUE, size=12,
         bold=False, color=BLACK, radius=False, width=1.0):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(width)
    if text:
        tf = s.text_frame; tf.clear(); tf.word_wrap = True
        tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def line(slide, x1, y1, x2, y2, color=NAVY, width=1.3, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow: c.line.end_arrowhead = True
    return c


def top_style(slide, title):
    rect(slide, 0, 0, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 0, 6.98, 0.25, fill=INDIGO, line=INDIGO)
    rect(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY)
    text_box(slide, 0.23, 0.31, 12.7, 0.50, title, size=25, color=WHITE,
             bold=True, align=PP_ALIGN.LEFT)


def bottom_style(slide):
    rect(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO)


def bars(slide, x, y, w, h, values, colors, baseline=True, gap=0.06):
    n = len(values); bw = (w - gap * (n - 1)) / n
    maxv = max(values) or 1
    for i, (v, col) in enumerate(zip(values, colors)):
        bh = h * v / maxv
        rect(slide, x + i * (bw + gap), y + h - bh, bw, bh, fill=col, line=col, width=0.3)
    if baseline:
        line(slide, x - 0.05, y + h, x + w + 0.05, y + h, color=BLACK, width=0.7)


def signal_card(slide, x, y, w, title, formula, meaning, kind):
    # Card is intentionally white; visual is in the upper half, formula in a tail-style box.
    rect(slide, x, y, w, 1.47, fill=WHITE, line=MID_GRAY, radius=True, width=0.8)
    text_box(slide, x + 0.06, y + 0.05, w - 0.12, 0.23, title, size=12, color=ORANGE, bold=True)
    if kind == "util":
        vals = [0.25, 0.45, 0.70, 0.92, 0.86, 0.60, 0.30]
        bars(slide, x + 0.18, y + 0.36, w - 0.36, 0.45, vals, [BLUE] * 7)
        text_box(slide, x + 0.12, y + 0.84, w - 0.24, 0.18, "Aₜ,b > ε 的 Block 数量 / |Rₜ|", size=8, color=BLUE, bold=True)
    elif kind == "tail":
        vals = [1.00, 0.78, 0.58, 0.43, 0.33, 0.26, 0.23, 0.22]
        colors = [BLUE] * 5 + [ORANGE] * 3
        bars(slide, x + 0.18, y + 0.36, w - 0.36, 0.45, vals, colors)
        text_box(slide, x + 0.12, y + 0.84, w - 0.24, 0.18, "高分 Block              尾部 25%", size=8, color=GRAY, bold=True)
    else:
        # Two boundary bars: selected K and first unselected K+1.
        rect(slide, x + 0.42, y + 0.47, 0.40, 0.34, fill=BLUE, line=BLUE, width=0.3)
        rect(slide, x + 0.91, y + 0.56, 0.40, 0.25, fill=ORANGE, line=ORANGE, width=0.3)
        line(slide, x + 0.38, y + 0.93, x + 1.36, y + 0.93, color=BLACK, width=0.7)
        text_box(slide, x + 0.20, y + 0.86, 0.55, 0.23, "sₜ,K", size=9, color=BLUE, bold=True)
        text_box(slide, x + 0.90, y + 0.86, 0.62, 0.23, "sₜ,K+1", size=9, color=ORANGE, bold=True)
        text_box(slide, x + 1.52, y + 0.47, w - 1.64, 0.43, "间隔大：边界清晰\n间隔小：边界不确定", size=8, color=GRAY, bold=True, align=PP_ALIGN.LEFT)
    rect(slide, x + 0.08, y + 1.05, w - 0.16, 0.34, formula, fill=LIGHT_ORANGE, line=ORANGE, size=10, bold=True, radius=True)
    text_box(slide, x + 0.10, y + 1.39, w - 0.20, 0.20, meaning, size=8, color=GRAY, bold=True)


def make_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：基于检索反馈的运行时动态 Block 预算控制")
    text_box(slide, 0.48, 0.98, 12.3, 0.31,
             "初始预算来源可以不同，但检索结果都通过统一控制器反馈到下一解码步。",
             size=16, bold=True, align=PP_ALIGN.LEFT)
    line(slide, 8.53, 1.32, 8.53, 6.34, color=NAVY, width=1.4)

    # Left section title and execution pipeline
    text_box(slide, 0.48, 1.30, 7.55, 0.30, "（1）检索执行与反馈观测", size=16, bold=True)
    rect(slide, 0.55, 1.70, 2.05, 0.52, "当前 Query + Key Block 表示\n候选得分 sₜ,b", fill=LIGHT_BLUE, line=BLUE, size=12, bold=True, radius=True)
    line(slide, 2.66, 1.96, 3.03, 1.96, color=BLUE, arrow=True)
    rect(slide, 3.10, 1.70, 2.12, 0.52, "Sink / Local / Top-Kₜ\nMiddle Block", fill=LIGHT_GREEN, line=GREEN, size=12, bold=True, radius=True)
    line(slide, 5.23, 1.96, 5.61, 1.96, color=GREEN, arrow=True)
    rect(slide, 5.67, 1.70, 2.12, 0.52, "稀疏 Attention\n已检索集合 Rₜ", fill=WHITE, line=GREEN, size=12, bold=True, radius=True)
    line(slide, 1.58, 2.28, 1.58, 2.58, color=ORANGE, arrow=True)
    line(slide, 4.16, 2.28, 4.16, 2.58, color=ORANGE, arrow=True)
    line(slide, 6.73, 2.28, 6.73, 2.58, color=ORANGE, arrow=True)

    # Three visual signal cards; enough space for formulas and interpretation.
    signal_card(slide, 0.55, 2.66, 2.38, "反馈1：Retrieved Block Utilization", "uₜ = #{Aₜ,b>ε}/|Rₜ|", "低：可能召回过多", "util")
    signal_card(slide, 3.05, 2.66, 2.38, "反馈2：Attention Tail Pressure", "pᵗᵃⁱˡₜ = Σ Bottom25%(Rₜ) Aₜ,b", "高：预算可能偏小", "tail")
    signal_card(slide, 5.55, 2.66, 2.38, "反馈3：Score Boundary Margin", "mₜ = sₜ,K − sₜ,K+1", "小：边界不确定", "margin")
    text_box(slide, 0.56, 4.26, 7.42, 0.27, "三类信号分别刻画：已选 Block 的利用程度、注意力分散程度、检索边界可靠性。", size=10, color=GRAY, bold=True)
    rect(slide, 0.55, 4.64, 7.40, 0.55, "风险判断：uₜ较高 ∧ pᵗᵃⁱˡₜ较高 ∧ mₜ较小  →  当前预算不足风险上升", fill=LIGHT_ORANGE, line=ORANGE, size=13, bold=True, radius=True)
    text_box(slide, 0.60, 5.28, 7.26, 0.29, "反馈信号可接收固定预算、MLP 预算或 Top-p 预算，不依赖单一初始分配方式。", size=10, color=GRAY, bold=True)

    # Right controller
    text_box(slide, 8.82, 1.30, 4.20, 0.30, "（2）通用运行时预算控制器", size=16, bold=True)
    rect(slide, 8.88, 1.76, 3.78, 0.65, "反馈归一化与多信号融合\nrₜ = g(uₜ, pᵗᵃⁱˡₜ, mₜ)", fill=LIGHT_ORANGE, line=ORANGE, size=15, bold=True, radius=True)
    text_box(slide, 8.92, 2.46, 3.68, 0.27, "高风险：利用率高、尾部压力高、边界间隔小", size=10, color=RED, bold=True)
    line(slide, 10.77, 2.74, 10.77, 3.03, color=ORANGE, arrow=True)
    rect(slide, 8.88, 3.10, 3.78, 0.65, "跨步骤平滑\nr̄ₜ = βr̄ₜ₋₁ + (1−β)rₜ", fill=LIGHT_BLUE, line=BLUE, size=15, bold=True, radius=True)
    text_box(slide, 8.94, 3.80, 3.65, 0.27, "抑制单个解码步的分数波动", size=10, color=GRAY, bold=True)
    line(slide, 10.77, 4.09, 10.77, 4.38, color=BLUE, arrow=True)
    rect(slide, 8.88, 4.45, 3.78, 0.70, "有界预算修正\nKₜ₊₁ = clip(Kₜ₊₁ⁱⁿⁱᵗ + δₜ)", fill=PURPLE, line=NAVY, size=15, bold=True, radius=True)
    line(slide, 10.77, 5.19, 10.77, 5.46, color=NAVY, arrow=True)
    rect(slide, 8.88, 5.53, 3.78, 0.72, "下一步访问预算 Kₜ₊₁\n高风险 ↑K；低利用率 ↓K", fill=LIGHT_GREEN, line=GREEN, size=15, bold=True, radius=True)
    text_box(slide, 8.90, 6.34, 3.73, 0.28, "Sink / Local 始终保留；控制器调节 Middle 规模", size=10, color=GRAY, bold=True)

    # Bottom bullets + innovation
    rect(slide, 0.55, 5.77, 0.13, 0.13, fill=WHITE, line=BLACK, width=1.4)
    text_box(slide, 0.84, 5.70, 7.25, 0.32, "控制器不更新 MLP 参数，也不需要在线执行完整 Full Attention。", size=12, bold=True, align=PP_ALIGN.LEFT)
    rect(slide, 0.42, 6.58, 12.47, 0.57, fill=PURPLE, line=PURPLE)
    text_box(slide, 0.58, 6.64, 1.03, 0.40, "创新点：", size=19, color=BLUE, bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.52, 6.65, 11.1, 0.37, "将一次性的预算分配扩展为可反馈的闭环控制：初始预算给出起点，检索反馈评估实际需求，控制器持续调整后续访问规模。", size=13, align=PP_ALIGN.LEFT)
    bottom_style(slide)


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    make_slide(prs); prs.save(OUT); print(OUT)


if __name__ == "__main__":
    main()
