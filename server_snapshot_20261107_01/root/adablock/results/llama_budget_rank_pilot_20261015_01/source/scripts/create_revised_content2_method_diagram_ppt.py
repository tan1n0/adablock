from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "研究内容2_详细方法图_修订版.pptx"
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
    tf.margin_left = Inches(0.03); tf.margin_right = Inches(0.03)
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


def line(slide, x1, y1, x2, y2, color=NAVY, width=1.6, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow: c.line.end_arrowhead = True
    return c


def top_style(slide, title):
    rect(slide, 0, 0, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 0, 6.98, 0.25, fill=INDIGO, line=INDIGO)
    rect(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY)
    text_box(slide, 0.23, 0.31, 12.7, 0.50, title, size=27, color=WHITE,
             bold=True, align=PP_ALIGN.LEFT)


def bottom_style(slide):
    rect(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK)
    rect(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO)


def bullet(slide, y, segments):
    rect(slide, 0.55, y + 0.08, 0.13, 0.13, fill=WHITE, line=BLACK, width=1.5)
    x = 0.85
    for txt, color, bold, width in segments:
        text_box(slide, x, y, width, 0.33, txt, size=14, color=color,
                 bold=bold, align=PP_ALIGN.LEFT)
        x += width


def innovation(slide, text):
    rect(slide, 0.45, 6.58, 12.43, 0.56, fill=PURPLE, line=PURPLE)
    text_box(slide, 0.60, 6.64, 1.05, 0.40, "创新点：", size=20, color=BLUE,
             bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.55, 6.65, 11.05, 0.38, text, size=14, color=BLACK,
             align=PP_ALIGN.LEFT)


def content2_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：基于检索反馈的运行时动态 Block 预算控制")
    text_box(slide, 0.48, 0.98, 12.3, 0.35,
             "预算来源可以不同，但都由统一的运行时反馈控制器持续调整。",
             size=18, bold=True, align=PP_ALIGN.LEFT)
    line(slide, 4.20, 1.38, 4.20, 4.82, color=NAVY, width=1.4)
    line(slide, 8.32, 1.38, 8.32, 4.82, color=NAVY, width=1.4)

    # Panel 1: initial budget source
    text_box(slide, 0.50, 1.37, 3.45, 0.30, "（1）初始预算输入", size=16, bold=True)
    rect(slide, 0.70, 1.92, 1.42, 0.62, "固定预算\nK₀", fill=LIGHT_BLUE, line=BLUE, size=16, bold=True, radius=True)
    rect(slide, 2.27, 1.92, 1.42, 0.62, "MLP 预测\nKₜᵢₙᵢₜ", fill=PURPLE, line=NAVY, size=15, bold=True, radius=True)
    rect(slide, 0.70, 2.82, 1.42, 0.62, "Top-p\nKₜᵢₙᵢₜ", fill=LIGHT_ORANGE, line=ORANGE, size=16, bold=True, radius=True)
    rect(slide, 2.27, 2.82, 1.42, 0.62, "其他预算\n分配器", fill=LIGHT_GRAY, line=GRAY, size=15, bold=True, radius=True)
    for x, y in [(2.12, 2.23), (3.00, 2.23), (2.12, 3.13), (3.00, 3.13)]:
        line(slide, x, y, 2.08, 3.83, color=NAVY, width=1.0, arrow=True)
    rect(slide, 0.86, 3.86, 2.98, 0.67, "统一得到初始预算\nKₜᵢₙᵢₜ", fill=LIGHT_GREEN, line=GREEN, size=17, bold=True, radius=True)
    line(slide, 2.80, 4.55, 2.80, 4.83, color=GREEN, width=1.7, arrow=True)
    rect(slide, 0.72, 4.95, 3.27, 0.58, "固定 Sink + Local，不由控制器删除", fill=WHITE, line=GREEN, size=12, bold=True, radius=True)
    text_box(slide, 0.60, 5.66, 3.35, 0.32, "控制器关注历史 Middle Block 的访问规模", size=11, color=GRAY, bold=True)

    # Panel 2: retrieval and feedback
    text_box(slide, 4.48, 1.37, 3.54, 0.30, "（2）检索执行与反馈观测", size=16, bold=True)
    rect(slide, 4.62, 1.90, 3.12, 0.56, "当前 Query + Key Block 表示\n得到候选 Block 得分", fill=LIGHT_BLUE, line=BLUE, size=14, bold=True, radius=True)
    line(slide, 6.18, 2.49, 6.18, 2.78, color=BLUE, arrow=True)
    rect(slide, 4.62, 2.87, 3.12, 0.58, "选择 Sink / Local / Top-Kₜ Middle", fill=LIGHT_GREEN, line=GREEN, size=14, bold=True, radius=True)
    line(slide, 6.18, 3.47, 6.18, 3.76, color=GREEN, arrow=True)
    rect(slide, 4.62, 3.85, 3.12, 0.56, "稀疏 Attention\n得到已检索集合 Rₜ", fill=WHITE, line=GREEN, size=14, bold=True, radius=True)
    line(slide, 6.18, 4.43, 6.18, 4.70, color=ORANGE, arrow=True)
    # Three signal cards
    rect(slide, 4.58, 4.78, 0.95, 0.72, "利用率\nuₜ", fill=LIGHT_ORANGE, line=ORANGE, size=14, bold=True, radius=True)
    rect(slide, 5.67, 4.78, 0.95, 0.72, "尾部压力\npᵗᵃⁱˡ", fill=LIGHT_ORANGE, line=ORANGE, size=13, bold=True, radius=True)
    rect(slide, 6.76, 4.78, 0.95, 0.72, "边界间隔\nΔₜ", fill=LIGHT_ORANGE, line=ORANGE, size=13, bold=True, radius=True)
    text_box(slide, 4.58, 5.70, 3.15, 0.38, "反馈既可用于固定预算，也可用于预测预算", size=11, color=GRAY, bold=True)

    # Panel 3: controller
    text_box(slide, 8.60, 1.37, 4.25, 0.30, "（3）通用运行时预算控制器", size=16, bold=True)
    rect(slide, 8.65, 1.92, 3.72, 0.63, "反馈归一化与多信号融合\nrₜ = g(uₜ, pᵗᵃⁱˡ, Δₜ)", fill=LIGHT_ORANGE, line=ORANGE, size=15, bold=True, radius=True)
    line(slide, 10.51, 2.58, 10.51, 2.88, color=ORANGE, arrow=True)
    rect(slide, 8.65, 2.97, 3.72, 0.63, "跨步骤平滑\nr̄ₜ = βr̄ₜ₋₁ + (1−β)rₜ", fill=LIGHT_BLUE, line=BLUE, size=15, bold=True, radius=True)
    line(slide, 10.51, 3.63, 10.51, 3.94, color=BLUE, arrow=True)
    rect(slide, 8.65, 4.03, 3.72, 0.63, "有界预算修正\nKₜ₊₁ = clip(Kₜ₊₁ᶦⁿⁱᵗ + δₜ)", fill=PURPLE, line=NAVY, size=15, bold=True, radius=True)
    line(slide, 10.51, 4.70, 10.51, 4.99, color=NAVY, arrow=True)
    rect(slide, 8.65, 5.08, 3.72, 0.60, "下一步访问预算 Kₜ₊₁\n必要时触发增量扩展", fill=LIGHT_GREEN, line=GREEN, size=15, bold=True, radius=True)
    text_box(slide, 8.58, 5.82, 3.90, 0.34, "不更新预算模型参数，不执行在线反向传播", size=11, color=GRAY, bold=True)

    bullet(slide, 5.98, [("研究内容2不依赖某一种预算预测器，", BLACK, False, 3.10),
                         ("固定预算也可以动态调整", RED, True, 2.35), ("。", BLACK, False, 0.2)])
    innovation(slide, "将预算分配从一次性预测扩展为可反馈的闭环控制：初始预算负责给出起点，运行时控制器根据实际检索效果持续调整访问规模。")
    bottom_style(slide)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    content2_slide(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
