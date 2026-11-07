from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


OUT = "研究内容一二_方法图.pptx"

NAVY = RGBColor(47, 48, 171)
BLUE = RGBColor(76, 114, 176)
LIGHT_BLUE = RGBColor(231, 241, 251)
PURPLE = RGBColor(232, 225, 244)
LIGHT_PURPLE = RGBColor(244, 240, 249)
ORANGE = RGBColor(239, 145, 54)
LIGHT_ORANGE = RGBColor(253, 241, 225)
GREEN = RGBColor(77, 145, 115)
LIGHT_GREEN = RGBColor(228, 244, 235)
GRAY = RGBColor(90, 90, 90)
LIGHT_GRAY = RGBColor(245, 246, 248)
BLACK = RGBColor(25, 25, 25)
WHITE = RGBColor(255, 255, 255)


def set_text(shape, text, size=16, color=BLACK, bold=False, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def box(slide, x, y, w, h, text, fill=WHITE, line=BLUE, size=15, bold=False,
        color=BLACK, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(1.3)
    set_text(s, text, size=size, color=color, bold=bold)
    return s


def label(slide, x, y, w, h, text, size=13, color=GRAY, bold=False, align=PP_ALIGN.LEFT):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(s, text, size=size, color=color, bold=bold, align=align)
    return s


def arrow(slide, x1, y1, x2, y2, color=NAVY, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    c.line.end_arrowhead = True
    return c


def title_bar(slide, title, subtitle):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.72))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    label(slide, 0.42, 0.11, 9.5, 0.42, title, size=25, color=WHITE, bold=True)
    label(slide, 9.0, 0.18, 3.85, 0.28, subtitle, size=11, color=WHITE, align=PP_ALIGN.RIGHT)


def footer(slide, text):
    label(slide, 0.45, 7.15, 12.4, 0.2, text, size=9, color=GRAY, align=PP_ALIGN.RIGHT)


def add_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, "长上下文推理的动态 Block 访问方法", "研究内容一 + 研究内容二")
    label(slide, 0.65, 1.02, 12, 0.45, "核心目标：在保持长上下文任务能力的同时，减少每个解码步不必要的 KV Cache 访问。", size=20, bold=True)

    box(slide, 0.75, 2.0, 3.0, 1.3, "当前解码状态\nHidden State / Query", fill=LIGHT_BLUE, line=BLUE, size=19, bold=True)
    arrow(slide, 3.78, 2.65, 4.55, 2.65)
    box(slide, 4.62, 1.82, 3.1, 1.65, "研究内容一\n动态预算预测\n预测本步需要多少 Block", fill=PURPLE, line=NAVY, size=18, bold=True)
    arrow(slide, 7.78, 2.65, 8.55, 2.65)
    box(slide, 8.62, 2.0, 3.0, 1.3, "Block 检索与 Attention\nKey/Query 相关性排序", fill=LIGHT_GREEN, line=GREEN, size=17, bold=True)
    arrow(slide, 10.12, 3.34, 10.12, 4.18, color=ORANGE)
    box(slide, 8.62, 4.25, 3.0, 1.3, "研究内容二\n运行时预算校准\n根据检索反馈修正后续预算", fill=LIGHT_ORANGE, line=ORANGE, size=17, bold=True)
    arrow(slide, 8.62, 4.9, 7.78, 4.9, color=ORANGE)
    label(slide, 4.85, 4.48, 2.55, 0.55, "不更新 MLP 参数\n只调整预算输出", size=15, color=ORANGE, bold=True)
    arrow(slide, 4.62, 5.3, 3.72, 5.3, color=ORANGE)
    label(slide, 0.8, 5.0, 2.95, 0.8, "下一解码步\n预算更加匹配当前需求", size=17, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    footer(slide, "图：动态预算预测—稀疏检索—运行时校准总体流程")


def add_content1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, "研究内容一：逐解码步动态 Block 访问预算预测", "离线标签 + 在线预测")
    label(slide, 0.55, 0.88, 12.3, 0.35, "根据不同解码步骤的信息需求，预测历史中间区域应访问的 Block 数量。", size=18, bold=True)

    # Offline branch
    box(slide, 0.55, 1.45, 3.55, 4.95, "离线预算标签构造", fill=LIGHT_PURPLE, line=NAVY, size=19, bold=True)
    box(slide, 0.88, 2.05, 2.9, 0.72, "采样位置的注意力\n（若干层、若干解码位置）", fill=WHITE, line=BLUE, size=14)
    arrow(slide, 2.33, 2.82, 2.33, 3.15)
    box(slide, 0.88, 3.25, 2.9, 0.72, "Token Attention →\nBlock Attention Mass", fill=WHITE, line=BLUE, size=14)
    arrow(slide, 2.33, 4.03, 2.33, 4.34)
    box(slide, 0.88, 4.44, 2.9, 0.82, "覆盖阈值 τ\n得到近似 Oracle Budget", fill=LIGHT_ORANGE, line=ORANGE, size=15, bold=True)
    label(slide, 0.83, 5.56, 3.0, 0.55, "K* = 覆盖 τ 注意力质量所需的最小 Block 数", size=12, color=GRAY, align=PP_ALIGN.CENTER)

    # Online branch
    box(slide, 4.45, 1.45, 8.3, 4.95, "在线动态预算预测与检索", fill=LIGHT_BLUE, line=BLUE, size=19, bold=True)
    box(slide, 4.82, 2.05, 2.1, 0.84, "hₜ₋₁\n当前 Hidden State", fill=WHITE, line=BLUE, size=16, bold=True)
    box(slide, 4.82, 3.28, 2.1, 0.84, "Block Hidden\n均值摘要", fill=WHITE, line=BLUE, size=16, bold=True)
    box(slide, 4.82, 4.51, 2.1, 0.84, "Query-aware\n得分统计特征", fill=WHITE, line=BLUE, size=15, bold=True)
    arrow(slide, 6.98, 2.48, 7.56, 3.49)
    arrow(slide, 6.98, 3.70, 7.56, 3.70)
    arrow(slide, 6.98, 4.93, 7.56, 3.91)
    box(slide, 7.62, 2.82, 2.35, 1.75, "冻结的 Policy MLP\n\n输出预算桶 Kₜ\n（0/4/8/16/32…）", fill=PURPLE, line=NAVY, size=16, bold=True)
    arrow(slide, 9.99, 3.70, 10.45, 3.70)
    box(slide, 10.5, 2.25, 1.85, 1.0, "固定保留\nSink + Local", fill=LIGHT_GREEN, line=GREEN, size=15, bold=True)
    box(slide, 10.5, 4.0, 1.85, 1.0, "动态检索\nMiddle Blocks", fill=LIGHT_ORANGE, line=ORANGE, size=15, bold=True)
    arrow(slide, 11.42, 3.28, 11.42, 3.98, color=ORANGE)
    label(slide, 7.62, 5.08, 4.73, 0.65, "实际选块：当前 Query 与每层/每头 Key Block 表示计算得分并独立排序。", size=13, color=GRAY, align=PP_ALIGN.CENTER)
    footer(slide, "注：Hidden State 均值用于预算特征；分层分头检索时使用对应层/头的 Key centroid。")


def add_content2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, "研究内容二：面向预算误差的运行时动态校准", "基于检索分布反馈")
    label(slide, 0.55, 0.88, 12.3, 0.35, "不重新执行 Full Attention，不更新 MLP 参数，根据已有检索分数修正后续预算。", size=18, bold=True)

    box(slide, 0.62, 1.55, 2.5, 4.8, "初始预算预测", fill=LIGHT_BLUE, line=BLUE, size=19, bold=True)
    box(slide, 0.94, 2.42, 1.86, 0.92, "Policy MLP\n输出 Kₜ", fill=PURPLE, line=NAVY, size=18, bold=True)
    arrow(slide, 3.15, 2.88, 3.72, 2.88)

    box(slide, 3.78, 1.55, 3.05, 4.8, "当前步骤检索", fill=LIGHT_GREEN, line=GREEN, size=19, bold=True)
    box(slide, 4.13, 2.25, 2.35, 0.78, "Query–Key Block 得分", fill=WHITE, line=GREEN, size=15, bold=True)
    arrow(slide, 5.3, 3.08, 5.3, 3.39, color=GREEN)
    box(slide, 4.13, 3.5, 2.35, 1.02, "选择 Sink / Local /\nTop-K Middle Blocks", fill=WHITE, line=GREEN, size=15, bold=True)
    arrow(slide, 5.3, 4.56, 5.3, 4.86, color=GREEN)
    box(slide, 4.13, 4.98, 2.35, 0.78, "稀疏 Attention", fill=WHITE, line=GREEN, size=15, bold=True)
    arrow(slide, 6.85, 2.88, 7.42, 2.88)

    box(slide, 7.48, 1.55, 2.55, 4.8, "反馈信号", fill=LIGHT_ORANGE, line=ORANGE, size=19, bold=True)
    box(slide, 7.82, 2.28, 1.88, 0.9, "尾部质量 rₜ\n（最低 25%）", fill=WHITE, line=ORANGE, size=15, bold=True)
    arrow(slide, 8.76, 3.22, 8.76, 3.53, color=ORANGE)
    box(slide, 7.82, 3.64, 1.88, 0.93, "指数滑动平均\nr̄ₜ", fill=WHITE, line=ORANGE, size=15, bold=True)
    arrow(slide, 8.76, 4.62, 8.76, 4.93, color=ORANGE)
    box(slide, 7.82, 5.04, 1.88, 0.78, "预算风险", fill=WHITE, line=ORANGE, size=15, bold=True)
    arrow(slide, 10.08, 4.0, 10.68, 4.0, color=ORANGE)

    box(slide, 10.75, 1.55, 2.0, 4.8, "预算校准", fill=LIGHT_PURPLE, line=NAVY, size=19, bold=True)
    box(slide, 11.05, 2.37, 1.4, 1.05, "Δₜ\n限制幅度", fill=WHITE, line=NAVY, size=16, bold=True)
    arrow(slide, 11.75, 3.55, 11.75, 3.88, color=NAVY)
    box(slide, 11.05, 4.0, 1.4, 1.1, "调整预算\nLogits", fill=WHITE, line=NAVY, size=16, bold=True)
    arrow(slide, 10.75, 5.65, 9.95, 5.65, color=ORANGE)
    label(slide, 7.5, 6.53, 5.2, 0.35, "反馈作用于后续解码步；Sink/Local 固定保留。", size=13, color=GRAY, align=PP_ALIGN.RIGHT)
    footer(slide, "当前原型：tail_ema 预算校准；增量扩展检索可作为后续研究方向。")


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    # Remove default first slide if present.
    while prs.slides:
        rId = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[0]
    add_overview(prs)
    add_content1(prs)
    add_content2(prs)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
