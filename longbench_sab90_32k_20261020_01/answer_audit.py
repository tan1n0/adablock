"""Read-only answer audit; preserve original scores and predictions."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'answer_audit_20261021'
OUT.mkdir(exist_ok=True)
rows = sorted([json.loads(line) for file in ROOT.glob('qa_[0-3].jsonl')
               for line in file.read_text(encoding='utf-8').splitlines()],
              key=lambda r:r['evaluation_index'])
assert len(rows) == len({r['evaluation_index'] for r in rows}) == 300
notes = {
 12: 'B 省略了参考答案中的 SCWS 和 correlation，回答更笼统；A 68.42，B 31.25。',
 70: 'A 提到 Meaning Extraction Method，B 改成 LIWC/词汇资源，缺少参考答案的关键方法名。',
 90: '原文的 2,359 是开发集 2,000 加测试集 359；B 将 2,359 全部称为测试集，新增了错误细节。已核对源上下文。',
 97: 'A 明确给出 Transformer base，B 用 context-agnostic MT 描述同类基线，CADec 均保留；不能仅凭 F1 差距认定模型实体完全错误。',
 79: '核心答案均为两个 reward 的 harmonic mean，B 更长导致 F1 下降。',
 80: 'B 补充参考答案要求的 Coverage，但也增加其他内容，F1 反而下降；词重叠得分不等于内容完整性。',
 140: 'A 和 B 数值及单位一致，仅 LaTeX 单位前的空白不同，F1 从 0 到 50；格式敏感，不是检索改善。',
 135: 'L=8 和 L=14 均正确，仅连接词 and 不同，F1 从 88.89 到 100。',
 167: '五个模型名称均一致，仅省略 and，B F1 从 100 降到 90.91。',
 187: '两者均回答 lattice QCD 的 nuclear liquid-gas transition；B 更简短，F1 更高。',
 192: '两者核心均为隐私、有效性质疑和运行费用；B 措辞更接近参考答案，F1 提高 19.43。',
 126: '两者表达同一主体导数公式，但都触及 64-token 生成上限，A 写出更多分段条件；格式与输出截断混杂。',
 267: 'Full/S/A 为 Thames，B 为 Ravensbourne，参考答案为 Thames。原文同时出现两条河：地区位于 Thames 南岸，地名来源于 Ravensbourne 的渡口；疑似关系选择差异，不能据此证明相关块缺失。',
 292: 'B 输出 Sherry Boucher，A 为 Barbara Peppard，参考答案为 Sherry Boucher；B 含正确实体但夹带其他婚史且达到生成上限，不是完全正确的答案。',
 211: 'Full 包含正确对象 Karen Fairchild，S/A/B 均只答 Luke Bryan；这是所有稀疏方案共有的关系对象遗漏。',
 218: '参考答案 University of Glasgow；各方法均未给出正确学校，低分差主要来自共同错误答案的措辞和截断。',
 147: '本地 multifieldqa_en 样本含中文题，四组均触及 64-token 上限且英文词级 F1 均为 0；单列为评测适配问题，未修改评分。'
}
summary = {}
changed = []
for task in ['qasper','multifieldqa_en','musique']:
    sub = [r for r in rows if r['task']==task]
    pairs = {}
    for left,right in [('B','A'),('A','S'),('S','full'),('A','full'),('B','full')]:
        delta = [r['methods'][left]['f1']-r['methods'][right]['f1'] for r in sub]
        pairs[left+'-'+right] = dict(text_changed=sum(r['methods'][left]['prediction']!=r['methods'][right]['prediction'] for r in sub),
            score_changed=sum(d!=0 for d in delta),wins=sum(d>0 for d in delta),losses=sum(d<0 for d in delta),mean_delta=sum(delta)/len(delta))
    summary[task] = dict(n=len(sub),any_text_changed=sum(len({m['prediction'] for m in r['methods'].values()})>1 for r in sub),pairs=pairs)
for r in rows:
    if len({m['prediction'] for m in r['methods'].values()})<=1:
        continue
    changed.append(dict(evaluation_index=r['evaluation_index'],task=r['task'],source_index=r['source_index'],source_id=r['source_id'],
        question=r['question'],answers=r['answers'],note=notes.get(r['evaluation_index'],''),
        methods={n:{k:r['methods'][n][k] for k in ['prediction','f1','logical_fraction','padded_fraction','hit_generation_limit']} for n in ['full','S','A','B']}))
payload = dict(scope='All 300 stored answers; no generation or rescoring. IDs are zero-based evaluation_index.',
               limitation='Text/reference review; only selected cases checked against source context. No causal attribution to layer/head selection.',summary=summary,cases=changed)
(OUT/'differences.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
esc = lambda s:html.escape(str(s))
page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>回答差异逐题审计</title><style>body{max-width:1250px;margin:32px auto;font:16px/1.6 system-ui;padding:0 16px;color:#18232f}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{border:1px solid #ccd5df;padding:10px;vertical-align:top;overflow-wrap:anywhere}td:first-child,th:first-child{width:70px}pre{white-space:pre-wrap}details{margin:18px 0;border:1px solid #ccd5df;padding:14px}summary{cursor:pointer;font-weight:bold}.note{background:#fff4d5;padding:12px}</style><h1>300 题回答差异审计</h1><p>94 道题在四组中至少有一组回答文本不同。编号为日志 evaluation_index（从 0 开始）。保留原始 F1，不做事后重评分。正文判断依据回答和参考答案，部分重点题核对原始上下文；不能由回答差异直接归因到排序或预算。</p>']
page.append('<pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
for r in changed:
    a,b=r['methods']['A'],r['methods']['B']
    page.append(f'<details><summary>{esc(r["task"])} #{r["evaluation_index"]} | B−A={b["f1"]-a["f1"]:+.2f} | {esc(r["question"])}</summary>')
    page.append('<p>参考答案：'+esc(json.dumps(r['answers'],ensure_ascii=False))+'</p>')
    if r['note']:page.append('<p class="note">'+esc(r['note'])+'</p>')
    page.append('<table><tr><th>方法</th><th>原始回答、分数与开销</th></tr>')
    for n,m in r['methods'].items():
        page.append(f'<tr><td>{n}</td><td><pre>{esc(m["prediction"])}</pre>F1={m["f1"]:.4f}；有效读取={m["logical_fraction"]:.2%}；达到生成上限={m["hit_generation_limit"]}</td></tr>')
    page.append('</table></details>')
page.append('</html>')
(OUT/'answer_differences.html').write_text('\n'.join(page),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('changed cases',len(changed))
