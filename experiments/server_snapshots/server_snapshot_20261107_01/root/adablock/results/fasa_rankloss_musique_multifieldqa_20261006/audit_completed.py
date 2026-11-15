from pathlib import Path
import json,hashlib,subprocess
W=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results={}
for task,n,cap in [('musique',200,32),('multifieldqa_en',150,64)]:
 pair={}
 for method in ['fasa','all_layers']:
  d=W/task/method;r=d/'run'
  rows=[json.loads(x) for x in (r/'predictions.jsonl').read_text().splitlines()]
  cfg=json.loads((r/'config.json').read_text());summary=json.loads((r/'summary.json').read_text())
  assert len(rows)==n and len({x['id']for x in rows})==n and [x['index']for x in rows]==list(range(n))
  assert cfg['max_new_tokens']==cap and cfg['script_sha256']==sha(d/'run_eval.py')
  assert cfg['dataset_sha256']==sha(Path('/root/adablock/data/longbench')/(task+'.jsonl'))
  assert abs(summary['F1']-sum(x['F1']for x in rows)/n)<1e-10
  info={'samples':n,'max_new_tokens':cap,'prediction_sha256':sha(r/'predictions.jsonl'),'input_manifest_sha256':sha(r/'input_manifest.json'),'output_capped':sum(x['output_truncated']for x in rows)}
  for x in rows:
   assert 1<=len(x['token_ids'])<=cap and x['audit']['generate_args']['max_new_tokens']==cap
   assert x['audit']['generate_args']['min_length']==x['input_tokens']+1
  if method=='all_layers':
   bb=[x['budget_audit']for x in rows]
   for x,b in zip(rows,bb):
    steps=len(x['token_ids'])-1
    assert b['fallback_calls']==0 and b['full_cache_preserved'] and b['positions_checked']
    assert b['prefill_calls']==[1]*28 and b['decode_calls']==[steps]*28
    assert b['learned_selector_calls']==[steps]*28 and b['learned_score_calls']==[steps]*28
    assert b['ranker_forward_calls']==[steps*4]*28
    assert (b['block_size'],b['sink_blocks'],b['local_blocks'],b['middle_blocks'])==(16,2,32,32)
   active=[b for b in bb if sum(b['learned_selector_calls'])]
   info.update(learned_layers=list(range(28)),fallback_calls=0,selector_calls=sum(sum(b['learned_selector_calls'])for b in bb),ranker_forward_calls=sum(sum(b['ranker_forward_calls'])for b in bb),history_tokens_per_kv_head_range=[min(b['historical_tokens_min']for b in active),max(b['historical_tokens_max']for b in active)],tokens_including_self_per_kv_head_range=[min(b['total_with_self_min']for b in active),max(b['total_with_self_max']for b in active)],union_across_four_kv_heads_range=[min(b['kv_union_min']for b in active),max(b['kv_union_max']for b in active)])
  else:
   active=[];calls=0
   for x in rows:
    steps=len(x['token_ids'])-1;b=x['budget_audit']
    assert b['prefill_full']and b['layer0_dense']and b['selection_per_query_head']==256
    for i in range(28):
     s=b['layers'][str(i)];assert s['prefill_calls']==1 and s['decode_calls']==steps and s['fa2_calls']==steps+1
     assert s['selector_calls']==(steps if i else 0)
     if s['selector_calls']:
      assert s['head_token_count_min']==s['head_token_count_max']==256
      active.append(s);calls+=s['selector_calls']
   info.update(sparse_layers=list(range(1,28)),dense_layers=[0],selector_calls=calls,selected_tokens_per_query_head=256,union_per_kv_group_range=[min(min(s['kv_union_min'])for s in active),max(max(s['kv_union_max'])for s in active)],union_per_kv_group_mean=sum(sum(s['kv_union_sum'])for s in active)/(4*calls),union_across_query_heads_range=[min(s['all_head_union_min']for s in active),max(s['all_head_union_max']for s in active)])
  pair[method]=rows;results[task+'/'+method]=info
 assert all(a['id']==b['id']and a['input_ids_sha256']==b['input_ids_sha256']and a['token_ids'][0]==b['token_ids'][0]for a,b in zip(pair['fasa'],pair['all_layers']))
results['protocol']={'same_budget':False,'same_protocol_full_available':False,'paper_full_gap_resolved':False,'checkpoint_sha256':sha(Path('/root/adablock/checkpoints/ranker_all28_curve_20261005/best.pt')),'clock_audit':json.loads((W/'clock_audit.json').read_text())}
(W/'final_integrity_budget_audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
