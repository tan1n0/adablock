from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = "通用运行时预算控制器_连续预算版.pptx"
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


def text_box(slide, x, y, w, h, text, size=16, color=BLACK, bold=False,
             align=PP_ALIGN.CENTER):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.01); tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.name = "Microsoft YaHei"
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return s


def rect(slide, x, y, w, h, text="", fill=WHITE, line=BLUE, size=16,
         bold=False, color=BLACK, radius=True, width=1.2):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
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


def line(slide, x1, y1, x2, y2, color=NAVY, width=1.8, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if arrow: c.line.end_arrowhead = True
    return c


def top_style(slide, title):
    rect(slide, 0, 0, 6.35, 0.25, fill=BLACK, line=BLACK, radius=False)
    rect(slide, 6.35, 0, 6.98, 0.25, fill=INDIGO, line=INDIGO, radius=False)
    rect(slide, 0, 0.25, 13.333, 0.68, fill=NAVY, line=NAVY, radius=False)
    text_box(slide, 0.27, 0.31, 12.6, 0.50, title, size=27, color=WHITE,
             bold=True, align=PP_ALIGN.LEFT)


def bottom_style(slide):
    rect(slide, 0, 7.25, 6.35, 0.25, fill=BLACK, line=BLACK, radius=False)
    rect(slide, 6.35, 7.25, 6.98, 0.25, fill=INDIGO, line=INDIGO, radius=False)


def make_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top_style(slide, "研究内容2：基于检索反馈的运行时动态 Block 预算控制")
    text_box(slide, 0.50, 1.02, 12.25, 0.35,
             "（2）通用运行时预算控制器：不使用固定预算桶，连续调整访问规模。",
             size=19, bold=True, align=PP_ALIGN.LEFT)

    # Input feedback strip
    rect(slide, 0.70, 1.75, 3.60, 0.75,
         "检索反馈\n尾部压力 pₜᵗᵃⁱˡ + 边界间隔 mₜ",
         fill=LIGHT_ORANGE, line=ORANGE, size=18, bold=True, radius=True)
    line(slide, 4.35, 2.12, 4.90, 2.12, color=ORANGE, arrow=True)
    rect(slide, 5.00, 1.75, 3.25, 0.75,
         "反馈融合\nrₜ = g(pₜᵗᵃⁱˡ, mₜ)",
         fill=LIGHT_BLUE, line=BLUE, size=18, bold=True, radius=True)
    line(slide, 8.30, 2.12, 8.62, 2.12, color=BLUE, arrow=True)

    # Main controller column
    rect(slide, 8.72, 1.42, 3.95, 4.55, fill=LIGHT_GRAY, line=MID_GRAY if False else BLUE,
         radius=True, width=1.0)
    text_box(slide, 8.92, 1.56, 3.55, 0.33, "连续预算控制器", size=20, color=NAVY, bold=True)
    rect(slide, 9.00, 2.05, 3.38, 0.70,
         "跨步骤平滑\nr̄ₜ = βr̄ₜ₋₁ + (1−β)rₜ",
         fill=LIGHT_BLUE, line=BLUE, size=17, bold=True, radius=True)
    text_box(slide, 9.06, 2.80, 3.25, 0.28, "抑制单个解码步的反馈波动", size=11, color=GRAY, bold=True)
    line(slide, 10.69, 3.12, 10.69, 3.42, color=BLUE, arrow=True)
    rect(slide, 9.00, 3.49, 3.38, 0.82,
         "连续预算修正\nκₜ₊₁ = clip(κₜ₊₁ⁱⁿⁱᵗ + δₜ, κmin, κmax)",
         fill=PURPLE, line=NAVY, size=16, bold=True, radius=True)
    text_box(slide, 9.06, 4.37, 3.25, 0.30, "κ：连续预算，不受离散桶限制", size=11, color=GRAY, bold=True)
    line(slide, 10.69, 4.72, 10.69, 5.02, color=NAVY, arrow=True)
    rect(slide, 9.00, 5.09, 3.38, 0.65,
         "Block 数取整\nKₜ₊₁ = ⌈κₜ₊₁⌉",
         fill=LIGHT_GREEN, line=GREEN, size=17, bold=True, radius=True)

    # Bottom conversion and output
    line(slide, 10.69, 5.80, 10.69, 6.13, color=GREEN, arrow=True)
    rect(slide, 4.35, 5.93, 4.18, 0.62,
         "实际访问 Token\nBₜ₊₁ = Kₜ₊₁ × block_size",
         fill=LIGHT_GREEN, line=GREEN, size=17, bold=True, radius=True)
    line(slide, 8.55, 6.24, 9.00, 6.24, color=GREEN, arrow=True)
    text_box(slide, 0.74, 2.82, 3.55, 0.38,
             "高尾部压力 → δₜ > 0 → 增加预算\n低尾部压力 → δₜ < 0 → 减少预算",
             size=14, color=RED, bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 0.74, 3.75, 3.60, 0.55,
             "边界间隔 mₜ 作为门控：\n间隔小才允许明显调整，间隔大则保持稳定。",
             size=13, color=GRAY, bold=True, align=PP_ALIGN.LEFT)
    rect(slide, 0.70, 4.70, 3.65, 0.70,
         "固定预算 / MLP 预算 / Top-p 预算\n均可作为 κₜ₊₁ⁱⁿⁱᵗ",
         fill=WHITE, line=ORANGE, size=15, bold=True, radius=True)

    # Explanations and innovation
    rect(slide, 0.55, 6.58, 12.25, 0.57, fill=PURPLE, line=PURPLE, radius=False)
    text_box(slide, 0.72, 6.64, 1.03, 0.40, "核心变化：", size=19, color=BLUE, bold=True, align=PP_ALIGN.LEFT)
    text_box(slide, 1.70, 6.65, 10.75, 0.37,
             "控制器维护连续预算 κ，而不是在 0/4/8/16/32 等预算桶之间跳转；最终仅因硬件 Block 粒度取整。",
             size=13, color=BLACK, align=PP_ALIGN.LEFT)
    bottom_style(slide)


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    while prs.slides:
        rid = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rid); del prs.slides._sldIdLst[0]
    make_slide(prs); prs.save(OUT); print(OUT)


if __name__ == "__main__":
    main()
