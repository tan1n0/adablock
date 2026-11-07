from pathlib import Path
import json,time,traceback
R=Path('/root/adablock');O=R/'results/longbench_ranker_hotpot8_32k128_20261004';OLD=R/'results/longbench_ranker_hotpot8_20261004'
try:
 while not (O/'COMPLETE').exists():
  p=O/'status.json'
  if p.exists() and json.loads(p.read_text()).get('stage')=='failed':raise RuntimeError(p.read_text())
  time.sleep(15)
 names=['full','cobs','quest','learned']
 rec={n:[json.loads(l) for l in (O/(n+'.jsonl')).read_text().splitlines()] for n in names}
 assert all(len(rs)==8 for rs in rec.values())
 a={'counts':{n:len(rs) for n,rs in rec.items()},'input_truncated':{},'output_truncated':{},'input_consistency':True,'selector_differences':{}}
 for n,rs in rec.items():
  a['input_truncated'][n]=[r['index'] for r in rs if r['input_truncated']]
  a['output_truncated'][n]=[r['index'] for r in rs if r['output_truncated']]
  for i,r in enumerate(rs):
   f=rec['full'][i]
   assert r['input_ids_sha256']==f['input_ids_sha256']
   assert r['retained_prompt_tokens']==r['original_prompt_tokens']
   assert r['retained_prompt_tokens']+128<=32768
   assert r['trace'][0]['token_id']==f['trace'][0]['token_id']
   if n!='full':
    assert r['audit']['max_selected_history_tokens']<=1056
    assert r['stats']['avg_requested_middle_blocks']==32
    assert r['audit']['attention_calls']==28*(r['stats']['generated_tokens']-1)
   if n=='learned':assert set(r['audit']['learned_calls'])=={'12','20','27'}
 for n in ['quest','learned']:
  a['selector_differences'][n]=[]
  for i,r in enumerate(rec[n]):
   ca={x['layer']:x['head0_sha256'] for x in rec['cobs'][i]['audit']['first_selections']}
   cb={x['layer']:x['head0_sha256'] for x in r['audit']['first_selections']}
   a['selector_differences'][n].append({'index':i,'layers':[k for k in ca if ca[k]!=cb[k]]})
 s=json.loads((O/'summary.json').read_text());old=json.loads((OLD/'summary.json').read_text())
 comparison={n:{'old8k32':old[n],'new32k128':s[n],'score_difference':s[n]['score']-old[n]['score']} for n in names}
 (O/'completion_audit.json').write_text(json.dumps(a,indent=2))
 (O/'comparison_with_8k.json').write_text(json.dumps(comparison,indent=2))
 labels={'full':'Full attention','cobs':'COBS','quest':'Quest AABB','learned':'Learned三层+其余COBS'}
 lines=['','### 32K/128重跑完成结果','','四种方案均完成8/8，实际prompt上限32640（模型位置32768减预留128），所有输入完整保留，四方案同样本input token哈希一致。训练三层Learned与其余COBS混合方式、epoch7 checkpoint、middle32预算、其余配置不变。','','| 方法 | 原8K/32 F1 | 新32K/128 F1 | 新总秒数 | 峰值allocated MiB | 峰值reserved MiB | 输出截断样本（0起始） |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for n in names:
  v=s[n];lines.append(f"| {labels[n]} | {old[n]['score']:.2f} | {v['score']:.2f} | {v['seconds']:.2f} | {v['peak_allocated_mib']:.1f} | {v['peak_reserved_mib']:.1f} | {a['output_truncated'][n]} |")
 lines+=['','逐条原始/保留prompt token数：','','| 样本（0起始） | 原始 | 保留 | 输入截断 |','| --- | ---: | ---: | --- |']
 for r in rec['full']:lines.append(f"| {r['index']} | {r['original_prompt_tokens']} | {r['retained_prompt_tokens']} | {r['input_truncated']} |")
 lines+=['','评分仍为官方QA token-F1多参考取max再做均值乘100。此次同时扩大输入与输出上限，分数变化不能单独归因于其中一项。仅8条有序样本、单预算，非全量LongBench，不作统计显著性结论。计时含tokenization、dense prefill、摘要构造、decode和trace，不含模型加载及探针；单次顺序运行且各方法生成长度可不同，不能作为纯decode吞吐基准。显存为PyTorch峰值，包括模型和prefill、不含CPU完整KV与驱动内存；稀疏仅作用decode（第一token为dense prefill），不宣称稀疏prefill或永久KV驱逐。','','本轮产物results/longbench_ranker_hotpot8_32k128_20261004：config.json、input_manifest.json、memory_preflight.json、path_verification.json、四方法jsonl、summary.json、completion_audit.json、comparison_with_8k.json。代码scripts/run_ranker_hotpot8_32k128_20261004.py，旧8K结果完整保留。']
 with (R/'docs/ranker_full_training_20261003.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 (O/'AUDIT_COMPLETE').write_text('Completed and documented')
 print(json.dumps(comparison,indent=2))
except Exception:
 (O/'AUDIT_FAILED').write_text(traceback.format_exc());raise
