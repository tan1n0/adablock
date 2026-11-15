from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究内容2_反馈信号详细方法图.pptx"
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
BLACK = RGBColor(15, 15, 20)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(225, 38, 38)


def text_box(slide, x, y, w, h, text, size=13, color=BLACK, bold=False,
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


def bullet(slide, y, parts):
    rect(slide, 0.54, y + 0.08, 0.13, 0.13, fill=WHITE, line=BLACK, width=1.4)
    x = 0.84
    for txt, color, bold, width in parts:
        text_box(slide, x, y, width, 0.32, txt, size=13, color=color,
                 bold=bold, align=PP_ALIGN.LEFT)
        x += width


def innovation(slide, text):
    rect(slide, 0.43, 6.58, 12.47, 0.57, fill=PURPLE, line=PURPLE)
    text_box(slide, 0.58, 6.64, 1.03, 0.40, "创新点：", size=19, color=BLUE,
             bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.52, 6.65, 11.1, 0.37, text, size=13, color=BLACK,
             align=PP_ALIGN.LEFT)


def make_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：基于检索反馈的运行时动态 Block 预算控制")
    text_box(slide, 0.48, 0.98, 12.3, 0.33,
             "预算来源可以不同，但都由统一的运行时反馈控制器持续调整。",
             size=17, bold=True, align=PP_ALIGN.LEFT)
    line(slide, 4.20, 1.36, 4.20, 4.89, color=NAVY, width=1.3)
    line(slide, 8.35, 1.36, 8.35, 4.89, color=NAVY, width=1.3)

    # Left: execution and three detailed signals
    text_box(slide, 4.48, 1.30, 3.55, 0.30, "（1）检索执行与反馈观测", size=15, bold=True)
    rect(slide, 4.55, 1.72, 3.38, 0.50,
         "当前 Query + Key Block 表示\n得到候选 Block 得分 sₜ,b",
         fill=LIGHT_BLUE, line=BLUE, size=13, bold=True, radius=True)
    line(slide, 6.24, 2.25, 6.24, 2.46, color=BLUE, arrow=True)
    rect(slide, 4.55, 2.51, 3.38, 0.50,
         "选择 Sink / Local / Top-Kₜ Middle\n得到已检索集合 Rₜ",
         fill=LIGHT_GREEN, line=GREEN, size=13, bold=True, radius=True)
    line(slide, 6.24, 3.04, 6.24, 3.26, color=GREEN, arrow=True)
    rect(slide, 4.55, 3.31, 3.38, 0.50,
         "稀疏 Attention\n得到 Aₜ,b（仅在 Rₜ 内计算）",
         fill=WHITE, line=GREEN, size=13, bold=True, radius=True)
    line(slide, 6.24, 3.84, 6.24, 4.02, color=ORANGE, arrow=True)

    # Three signal cards, each with formula and interpretation
    rect(slide, 4.50, 4.08, 1.04, 1.35, fill=LIGHT_ORANGE, line=ORANGE, radius=True)
    text_box(slide, 4.57, 4.14, 0.90, 0.27, "利用率", size=12, color=ORANGE, bold=True)
    text_box(slide, 4.56, 4.45, 0.92, 0.42, "uₜ = #\n{Aₜ,b>ε}/|Rₜ|", size=11, bold=True)
    text_box(slide, 4.55, 4.90, 0.94, 0.43, "低：召回过多\n高：已选多有效", size=9, color=GRAY, bold=True)

    rect(slide, 5.68, 4.08, 1.04, 1.35, fill=LIGHT_ORANGE, line=ORANGE, radius=True)
    text_box(slide, 5.73, 4.14, 0.94, 0.27, "尾部压力", size=12, color=ORANGE, bold=True)
    text_box(slide, 5.74, 4.45, 0.92, 0.42, "pᵗᵃⁱˡₜ = Σ\nBottom25%(Rₜ) Aₜ,b", size=10, bold=True)
    text_box(slide, 5.73, 4.90, 0.94, 0.43, "高：注意力较分散\n预算可能偏小", size=9, color=GRAY, bold=True)

    rect(slide, 6.86, 4.08, 1.04, 1.35, fill=LIGHT_ORANGE, line=ORANGE, radius=True)
    text_box(slide, 6.91, 4.14, 0.94, 0.27, "边界间隔", size=12, color=ORANGE, bold=True)
    text_box(slide, 6.92, 4.45, 0.92, 0.42, "mₜ = sₜ,K−sₜ,K+1", size=11, bold=True)
    text_box(slide, 6.91, 4.90, 0.94, 0.43, "小：边界不确定\n预算风险较高", size=9, color=GRAY, bold=True)
    text_box(slide, 4.52, 5.55, 3.38, 0.34,
             "三类信号分别反映：利用程度、注意力分散度、检索边界可靠性",
             size=10, color=GRAY, bold=True)

    # Right: controller
    text_box(slide, 8.62, 1.30, 4.15, 0.30, "（2）通用运行时预算控制器", size=15, bold=True)
    rect(slide, 8.65, 1.72, 3.75, 0.57,
         "反馈归一化与多信号融合\nrₜ = g(uₜ, pᵗᵃⁱˡₜ, mₜ)",
         fill=LIGHT_ORANGE, line=ORANGE, size=14, bold=True, radius=True)
    text_box(slide, 8.67, 2.34, 3.70, 0.28,
             "高风险：uₜ较高、pᵗᵃⁱˡ较高、mₜ较小",
             size=10, color=RED, bold=True)
    line(slide, 10.53, 2.65, 10.53, 2.90, color=ORANGE, arrow=True)
    rect(slide, 8.65, 2.96, 3.75, 0.57,
         "跨步骤平滑\nr̄ₜ = βr̄ₜ₋₁ + (1−β)rₜ",
         fill=LIGHT_BLUE, line=BLUE, size=14, bold=True, radius=True)
    text_box(slide, 8.68, 3.58, 3.68, 0.27,
             "避免单个解码步的分数波动触发误调节",
             size=10, color=GRAY, bold=True)
    line(slide, 10.53, 3.87, 10.53, 4.11, color=BLUE, arrow=True)
    rect(slide, 8.65, 4.17, 3.75, 0.64,
         "有界预算修正\nKₜ₊₁ = clip(Kₜ₊₁ⁱⁿⁱᵗ + δₜ)",
         fill=PURPLE, line=NAVY, size=14, bold=True, radius=True)
    line(slide, 10.53, 4.84, 10.53, 5.07, color=NAVY, arrow=True)
    rect(slide, 8.65, 5.12, 3.75, 0.61,
         "下一步访问预算 Kₜ₊₁\n高风险 ↑K；低利用率 ↓K",
         fill=LIGHT_GREEN, line=GREEN, size=14, bold=True, radius=True)
    text_box(slide, 8.66, 5.80, 3.72, 0.30,
             "Sink / Local 始终保留；控制器只调节 Middle 访问规模",
             size=10, color=GRAY, bold=True)

    # Bottom explanations and innovation
    bullet(slide, 5.99, [("研究内容2不依赖初始预算来源：", BLACK, False, 2.85),
                         ("固定预算、MLP 或 Top-p 均可接入", RED, True, 3.25), ("。", BLACK, False, 0.2)])
    innovation(slide, "将一次性的预算分配扩展为可反馈的闭环控制：初始预算给出起点，检索反馈评估实际需求，控制器持续调整后续访问规模。")
    bottom_style(slide)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    make_slide(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
