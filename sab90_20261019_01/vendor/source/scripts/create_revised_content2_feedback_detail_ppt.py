from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究内容2_详细方法图_三反馈展开版.pptx"
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
             align=PP_ALIGN.CENTER):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.035); tf.margin_right = Inches(0.035)
    tf.margin_top = Inches(0.01); tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def rect(slide, x, y, w, h, text="", fill=WHITE, line=BLUE, size=13,
         bold=False, color=BLACK, radius=False, width=1.1):
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


def line(slide, x1, y1, x2, y2, color=NAVY, width=1.5, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow: c.line.end_arrowhead = True
    return c


def top_style(slide, title):
    rect(slide, 0, 0, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 0, 6.98, 0.25, fill=INDIGO, line=INDIGO)
    rect(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY)
    text_box(slide, 0.23, 0.31, 12.7, 0.50, title, size=26, color=WHITE,
             bold=True, align=PP_ALIGN.LEFT)


def bottom_style(slide):
    rect(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO)


def bullet(slide, y, segments):
    rect(slide, 0.55, y + 0.08, 0.13, 0.13, fill=WHITE, line=BLACK, width=1.5)
    x = 0.85
    for txt, color, bold, width in segments:
        text_box(slide, x, y, width, 0.29, txt, size=13, color=color,
                 bold=bold, align=PP_ALIGN.LEFT)
        x += width


def innovation(slide, text):
    rect(slide, 0.45, 6.58, 12.43, 0.56, fill=PURPLE, line=PURPLE)
    text_box(slide, 0.60, 6.64, 1.05, 0.40, "创新点：", size=19, color=BLUE,
             bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.55, 6.65, 11.05, 0.38, text, size=13, color=BLACK,
             align=PP_ALIGN.LEFT)


def detail_card(slide, x, y, w, h, title, formula, meaning, action, accent=ORANGE):
    rect(slide, x, y, w, h, fill=LIGHT_ORANGE, line=accent, radius=True, width=1.35)
    rect(slide, x + 0.10, y + 0.10, w - 0.20, 0.40, title, fill=accent,
         line=accent, color=WHITE, size=15, bold=True, radius=True)
    text_box(slide, x + 0.16, y + 0.63, w - 0.32, 0.42, formula, size=13,
             color=BLACK, bold=True)
    text_box(slide, x + 0.14, y + 1.08, w - 0.28, 0.42, meaning, size=11,
             color=GRAY, bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, x + 0.14, y + 1.55, w - 0.28, 0.40, action, size=11,
             color=RED if "增加" in action else GREEN, bold=True, align=PP_ALIGN.LEFT)


def content2_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：基于检索反馈的运行时动态 Block 预算控制")
    text_box(slide, 0.48, 0.98, 12.3, 0.35,
             "预算来源可以不同，但都由统一的运行时反馈控制器持续调整。",
             size=18, bold=True, align=PP_ALIGN.LEFT)

    # Main upper flow: preserve the original content and structure.
    line(slide, 4.28, 1.38, 4.28, 4.28, color=NAVY, width=1.4)
    line(slide, 8.44, 1.38, 8.44, 4.28, color=NAVY, width=1.4)

    # Initial budget source
    text_box(slide, 0.50, 1.33, 3.50, 0.30, "（1）初始预算输入", size=16, bold=True)
    rect(slide, 0.68, 1.82, 1.40, 0.58, "固定预算\nK₀", fill=LIGHT_BLUE, line=BLUE, size=15, bold=True, radius=True)
    rect(slide, 2.30, 1.82, 1.40, 0.58, "MLP 预测\nKₜᵢₙᵢₜ", fill=PURPLE, line=NAVY, size=14, bold=True, radius=True)
    rect(slide, 0.68, 2.62, 1.40, 0.58, "Top-p\nKₜᵢₙᵢₜ", fill=LIGHT_ORANGE, line=ORANGE, size=15, bold=True, radius=True)
    rect(slide, 2.30, 2.62, 1.40, 0.58, "其他预算\n分配器", fill=LIGHT_GRAY, line=GRAY, size=14, bold=True, radius=True)
    for x, y in [(2.08, 2.11), (3.00, 2.11), (2.08, 2.91), (3.00, 2.91)]:
        line(slide, x, y, 2.12, 3.36, color=NAVY, width=0.9, arrow=True)
    rect(slide, 0.85, 3.38, 3.00, 0.58, "统一得到初始预算\nKₜᵢₙᵢₜ", fill=LIGHT_GREEN, line=GREEN, size=16, bold=True, radius=True)
    line(slide, 2.35, 3.98, 2.35, 4.25, color=GREEN, width=1.5, arrow=True)
    rect(slide, 0.72, 4.30, 3.28, 0.45, "固定 Sink + Local，不由控制器删除", fill=WHITE, line=GREEN, size=11, bold=True, radius=True)

    # Retrieval flow
    text_box(slide, 4.50, 1.33, 3.62, 0.30, "（2）检索执行与反馈观测", size=16, bold=True)
    rect(slide, 4.60, 1.82, 3.25, 0.55, "当前 Query + Key Block 表示\n得到候选 Block 得分", fill=LIGHT_BLUE, line=BLUE, size=13, bold=True, radius=True)
    line(slide, 6.23, 2.40, 6.23, 2.61, color=BLUE, arrow=True)
    rect(slide, 4.60, 2.66, 3.25, 0.55, "选择 Sink / Local / Top-Kₜ Middle", fill=LIGHT_GREEN, line=GREEN, size=13, bold=True, radius=True)
    line(slide, 6.23, 3.24, 6.23, 3.45, color=GREEN, arrow=True)
    rect(slide, 4.60, 3.50, 3.25, 0.55, "稀疏 Attention\n得到已检索集合 Rₜ", fill=WHITE, line=GREEN, size=13, bold=True, radius=True)
    line(slide, 6.23, 4.08, 6.23, 4.31, color=ORANGE, arrow=True)
    text_box(slide, 4.75, 4.31, 2.95, 0.25, "从 Rₜ 和分数/注意力中提取反馈", size=11, color=ORANGE, bold=True)

    # Controller
    text_box(slide, 8.70, 1.33, 4.20, 0.30, "（3）通用运行时预算控制器", size=16, bold=True)
    rect(slide, 8.75, 1.82, 3.72, 0.55, "反馈归一化与多信号融合\nrₜ = g(uₜ, pᵗᵃⁱˡ, Δₜ)", fill=LIGHT_ORANGE, line=ORANGE, size=14, bold=True, radius=True)
    line(slide, 10.61, 2.40, 10.61, 2.61, color=ORANGE, arrow=True)
    rect(slide, 8.75, 2.66, 3.72, 0.55, "跨步骤平滑\nr̄ₜ = βr̄ₜ₋₁ + (1−β)rₜ", fill=LIGHT_BLUE, line=BLUE, size=14, bold=True, radius=True)
    line(slide, 10.61, 3.24, 10.61, 3.45, color=BLUE, arrow=True)
    rect(slide, 8.75, 3.50, 3.72, 0.55, "有界预算修正\nKₜ₊₁ = clip(Kₜ₊₁ᶦⁿⁱᵗ + δₜ)", fill=PURPLE, line=NAVY, size=14, bold=True, radius=True)
    line(slide, 10.61, 4.08, 10.61, 4.31, color=NAVY, arrow=True)
    rect(slide, 8.75, 4.31, 3.72, 0.45, "下一步访问预算 Kₜ₊₁", fill=LIGHT_GREEN, line=GREEN, size=14, bold=True, radius=True)

    # Detailed three-feedback section
    text_box(slide, 0.50, 4.90, 12.25, 0.27, "三类反馈信号：分别衡量已选 Block 的实际贡献、检索集合的尾部压力和检索边界的不确定性", size=14, bold=True, align=PP_ALIGN.LEFT)
    detail_card(
        slide, 0.55, 5.22, 3.95, 1.20,
        "反馈 1：Retrieved Block Utilization",
        "uₜ = # {b∈Rₜ : Aₜ,b > ε} / |Rₜ|",
        "已选集合中真正获得注意力的 Block 比例。",
        "uₜ 低 → 访问冗余，减少 K；uₜ 高且边界紧张 → 增加 K。",
    )
    detail_card(
        slide, 4.70, 5.22, 3.95, 1.20,
        "反馈 2：Attention Tail Pressure",
        "pₜᵗᵃⁱˡ = Σ Aₜ,b  （b∈Rₜ 的底部 25%）",
        "观察已召回集合中排名靠后的 Block 是否仍承载注意力。",
        "pₜᵗᵃⁱˡ 高 → 截断较激进，下一步增大 K。",
    )
    detail_card(
        slide, 8.85, 5.22, 3.95, 1.20,
        "反馈 3：Score Boundary Margin",
        "δₜ = sₜ,K − sₜ,K+1",
        "比较最后选中与第一个未选 Block 的相关性得分。",
        "δₜ 小 → 边界不确定，可能漏检，下一步增大 K。",
    )
    text_box(slide, 0.62, 6.43, 12.0, 0.19,
             "注：正式版本使用稀疏 Attention 得到 Aₜ,b；当前原型可先用 Block 得分分布构造 proxy 进行验证。",
             size=10, color=GRAY, align=PP_ALIGN.LEFT)

    innovation(slide, "将预算分配从一次性预测扩展为可反馈的闭环控制：固定预算、MLP 预算或 Top-p 预算均可作为起点，并由运行时控制器持续调整访问规模。")
    bottom_style(slide)


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    content2_slide(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
