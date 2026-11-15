from pathlib import Path
import json,hashlib,time
R=Path('/root/adablock');O=R/'results/longbench_ranker_hotpot8_20261004'
while not (O/'COMPLETE').exists():
 if json.loads((O/'status.json').read_text()).get('stage')=='failed':raise RuntimeError('Evaluation failed')
 time.sleep(15)
names=['full','cobs','quest','learned']
records={n:[json.loads(l) for l in (O/(n+'.jsonl')).read_text().splitlines()] for n in names}
assert all(len(v)==8 for v in records.values())
audit={'completed_samples':{n:len(v) for n,v in records.items()},'same_prompt_lengths':True,'generated_limit32':{},'selector_differences':{},'peak_memory_scope':'PyTorch allocator peak includes model and dense prefill; excludes CPU KV and CUDA driver; not decode-only KV savings','timing_scope':'single sequential run; includes tokenization/prefill/digest construction/decode/traces; excludes model loading and preliminary probe; GPU synchronized; no repeated latency confidence interval','quality_scope':'8 ordered examples; 7 truncated to8192; first token comes from dense prefill; sparse affects subsequent tokens; mixed Learned only layers12/20/27; not full LongBench or all-layer Learned'}
for n,rs in records.items():
 audit['generated_limit32'][n]=[]
 for i,r in enumerate(rs):
  f=records['full'][i]
  assert r['index']==i and r['stats']['prompt_tokens']==f['stats']['prompt_tokens'] and r['stats']['used_prompt_tokens']==f['stats']['used_prompt_tokens']
  assert r['trace'][0]['token_id']==f['trace'][0]['token_id']
  if r['stats']['generated_tokens']==32:
   audit['generated_limit32'][n].append({'index':i,'last_token_id':r['trace'][-1]['token_id'],'last_token_text':r['trace'][-1]['token_text']})
  if n!='full':
   assert r['stats']['avg_requested_middle_blocks']==32
   assert r['audit']['max_selected_history_tokens']<=1056
   assert r['audit']['attention_calls']==28*(r['stats']['generated_tokens']-1)
  if n=='learned':assert set(r['audit']['learned_calls'])=={'12','20','27'}
for n in ['quest','learned']:
 audit['selector_differences'][n]=[]
 for i,r in enumerate(records[n]):
  a={v['layer']:v['head0_sha256'] for v in records['cobs'][i]['audit']['first_selections']};b={v['layer']:v['head0_sha256'] for v in r['audit']['first_selections']}
  audit['selector_differences'][n].append({'index':i,'different_layers':[k for k in a if a[k]!=b[k]]})
audit['different_predictions_from_full']={n:[i for i,r in enumerate(rs) if r['prediction']!=records['full'][i]['prediction']] for n,rs in records.items()}
audit['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.jsonl')}
(O/'completion_audit.json').write_text(json.dumps(audit,indent=2))
s=json.loads((O/'summary.json').read_text())
labels={'full':'Full attention','cobs':'COBS','quest':'Quest AABB','learned':'Learned三层+其余COBS'}
lines=['','### HotpotQA 八样本最终结果','','四种方案均完成8/8；所有同样本prompt长度、截断长度与首token均一致。以下为官方LongBench QA token-F1（各参考答案取最大、再做样本均值乘100），不是离线attention coverage。','','| 方法 | F1 | 8条总秒数 | PyTorch峰值allocated MiB | 峰值reserved MiB | 生成tokens |','| --- | ---: | ---: | ---: | ---: | ---: |']
for n in names:
 v=s[n];lines.append(f"| {labels[n]} | {v['score']:.2f} | {v['seconds']:.2f} | {v['peak_allocated_mib']:.1f} | {v['peak_reserved_mib']:.1f} | {v['generated_tokens']} |")
lines+=['','路径验证：1024-token全候选稀疏与full前4个生成token一致；2048-token Learned探针真实裁剪到最多1056历史tokens，调用只在12/20/27层。正式记录每层首个head的选块哈希，Quest与COBS不同；Learned训练层也改变选块，后续COBS层可受隐藏状态传播影响。每个稀疏生成step覆盖28层，无reuse。全部历史KV存在CPU，dense prefill与第一生成token保持full；不宣称端到端稀疏prefill或永久KV驱逐。','','生成32-token上限记录：'+json.dumps(audit['generated_limit32'],ensure_ascii=False)+'。保留原输出，不基于结果延长生成重测。前7条输入中间截断到8192，最后1条2317；input_manifest.json保存输入token哈希。仅8条有序样本、单预算，不能据同分推断方法总体等价，也未证明此前coverage提升转化成任务得分提升。','','计时为单次顺序运行，含tokenization、dense prefill、摘要构造、解码及trace记录，排除模型加载和路径探针；生成长度不同，不能直接当纯decode吞吐基准。稀疏原型CPU KV和逐层搬运/摘要构造开销明显，未展示加速。显存为PyTorch allocator峰值（含模型与dense prefill），不含CPU KV/驱动显存，不是稀疏cache独立占用。','','产物：results/longbench_ranker_hotpot8_20261004/{summary.json,completion_audit.json,config.json,input_manifest.json,path_verification.json,full.jsonl,cobs.jsonl,quest.jsonl,learned.jsonl}；代码scripts/run_ranker_hotpot8_20261004.py；官方评分来源https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py。尚未进行全量LongBench、更长上下文、多预算或重复速度测量。']
with (R/'docs/ranker_full_training_20261003.md').open('a') as f:f.write('\n'.join(lines)+'\n')
(O/'AUDIT_COMPLETE').write_text('Results checked and project documentation updated.')
print(json.dumps({'summary':s,'audit':audit},indent=2))
