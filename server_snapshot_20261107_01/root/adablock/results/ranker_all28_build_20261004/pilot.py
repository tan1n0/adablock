import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import sys,json,time,traceback,collections
ROOT=Path('/root/adablock');sys.path.insert(0,str(ROOT));OUT=ROOT/'results/ranker_all28_build_20261004'
import torch
from transformers import AutoModel,AutoTokenizer
from scripts.build_ranking_dataset import ranking_metadata,collect_document
from utils.ranking_data import RankingWriter,document_identity
def main():
 torch.set_num_threads(4);start=time.perf_counter()
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
 model=AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct',torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 meta=ranking_metadata(model,'Qwen/Qwen2.5-7B-Instruct',list(range(28)),16,32,32,4);meta.update(sample_stride=64,attention_backend='sdpa',base_dtype='bfloat16')
 rows=[json.loads(l) for l in (OUT/'pilot_input.jsonl').read_text().splitlines()]
 results=[]
 for row,oldindex in zip(rows,[2,6]):
  target=OUT/f'pilot_doc_{oldindex:06d}';writer=RankingWriter(target,meta)
  ids=tok(row['text'],return_tensors='pt',truncation=True,max_length=32768)['input_ids'];torch.cuda.reset_peak_memory_stats()
  t=time.perf_counter();collect_document(model,ids.to(model.device),writer,document_identity(row,row['text']),64,8);manifest=writer.finish();torch.cuda.synchronize();seconds=time.perf_counter()-t
  assert {e['layer_id'] for e in manifest['shards']}==set(range(28))
  diffs={};maxmass=0;recs=None
  for e in manifest['shards']:
   sh=torch.load(target/e['file'],weights_only=True,map_location='cpu')
   assert all(torch.isfinite(v).all() for v in sh['digests'].values())
   if recs is None:recs=[r['token_position'] for r in sh['records']]
   assert [r['token_position'] for r in sh['records']]==recs
   for r in sh['records']:
    assert ((r['block_ids']+1)*16<=r['token_position']).all()
    assert all(torch.isfinite(r[k]).all() for k in ['query','teacher_mass','fixed_mass'])
    err=(r['fixed_mass']+r['teacher_mass'].sum(-1)-1).abs().max().item();maxmass=max(maxmass,err);assert err<2e-4
   if e['layer_id'] in [12,20,27]:
    old=torch.load(ROOT/f"data/ranker_cobs_train_full_32k_20261003/doc_{oldindex:06d}_layer_{e['layer_id']:03d}.pt",weights_only=True,map_location='cpu')
    assert len(old['records'])==len(sh['records'])
    dd={}
    for k,v in sh['digests'].items():dd['digest_'+k]=float((v-old['digests'][k]).abs().max())
    for x,y in zip(sh['records'],old['records']):
     assert x['token_position']==y['token_position'] and torch.equal(x['block_ids'],y['block_ids'])
     for k in ['query','teacher_mass','fixed_mass']:dd[k]=max(dd.get(k,0),float((x[k]-y[k]).abs().max()))
    diffs[str(e['layer_id'])]=dd
  rec={'old_index':oldindex,'tokens':ids.numel(),'seconds':seconds,'bytes':sum(p.stat().st_size for p in target.glob('*.pt')),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scenes':sum(e['scenes'] for e in manifest['shards']),'records_per_layer':len(recs),'normalization_max_error':maxmass,'comparison_max_abs_differences':diffs}
  results.append(rec);(OUT/'pilot_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(rec),flush=True)
 (OUT/'pilot_COMPLETE').write_text('Pilot extraction and audit completed')
 print('PILOT_COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  (OUT/'pilot_error.txt').write_text(traceback.format_exc());raise
