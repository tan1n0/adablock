import os,sys,json,time,traceback,hashlib,shutil,gc,fcntl
from pathlib import Path
GPU=int(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']=str(GPU);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path('/root/adablock');sys.path.insert(0,str(ROOT));RUN=ROOT/'results/ranker_all28_build_20261004'
import torch
from transformers import AutoModel,AutoTokenizer
from scripts.build_ranking_dataset import ranking_metadata,collect_document
from utils.ranking_data import RankingWriter,document_identity
def write(p,obj):
 p=Path(p);t=p.with_suffix('.tmp')
 with t.open('w') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2));f.flush();os.fsync(f.fileno())
 t.replace(p)
def sha(p):
 with p.open('rb') as f:
  digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest();os.fsync(f.fileno());return digest
def audit_doc(target,oldroot,index,oldmanifest):
 man=json.loads((target/'manifest.json').read_text())
 assert man['documents']==[oldmanifest['documents'][index]]
 assert man['metadata']==dict(oldmanifest['metadata'],layers=list(range(28)))
 assert len(man['shards'])==28 and {e['layer_id'] for e in man['shards']}==set(range(28))
 base=torch.load(oldroot/f'doc_{index:06d}_layer_012.pt',weights_only=True,map_location='cpu')
 expected=[r['token_position'] for r in base['records']];masserr=0.;diffs={};files={}
 for e in man['shards']:
  p=target/e['file'];x=torch.load(p,weights_only=True,map_location='cpu')
  assert x['schema']==man['metadata']['schema'] and x['layer_id']==e['layer_id']
  assert len(x['records'])==len(expected) and e['scenes']==len(expected)*4
  assert [r['token_position'] for r in x['records']]==expected
  assert all(torch.isfinite(v).all() for v in x['digests'].values())
  for rec,ref in zip(x['records'],base['records']):
   assert torch.equal(rec['block_ids'],ref['block_ids'])
   assert ((rec['block_ids']+1)*16<=rec['token_position']).all()
   assert all(torch.isfinite(rec[k]).all() for k in ['query','teacher_mass','fixed_mass'])
   assert rec['teacher_mass'].min()>=0 and rec['fixed_mass'].min()>=0
   err=(rec['fixed_mass']+rec['teacher_mass'].sum(-1)-1).abs().max().item();masserr=max(masserr,err);assert err<2e-4
  if e['layer_id'] in [12,20,27]:
   old=base if e['layer_id']==12 else torch.load(oldroot/f"doc_{index:06d}_layer_{e['layer_id']:03d}.pt",weights_only=True,map_location='cpu')
   ds={}
   for k,v in x['digests'].items():ds['digest_'+k]=float((v-old['digests'][k]).abs().max())
   for u,v in zip(x['records'],old['records']):
    for k in ['query','teacher_mass','fixed_mass']:ds[k]=max(ds.get(k,0),float((u[k]-v[k]).abs().max()))
   assert max(ds.values())==0,('old layer mismatch',index,e['layer_id'],ds)
   diffs[str(e['layer_id'])]=ds
  files[e['file']]={'bytes':p.stat().st_size,'sha256':sha(p)}
 return {'index':index,'layers':28,'scenes':sum(e['scenes'] for e in man['shards']),'records_per_layer':len(expected),'mass_normalization_max_error':masserr,'finite_and_causal':True,'old_three_layers_bitwise_tensor_equal':True,'comparison_max_abs_differences':diffs,'bytes':sum(v['bytes'] for v in files.values()),'files':files}
def main():
 lock=(RUN/f'worker_{GPU}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4)
 plan=json.loads((RUN/'plan.json').read_text());jobs=[]
 for split in ['validation','train']:
  cfg=plan[split];rows=[json.loads(l) for l in Path(cfg['input']).read_text().splitlines()];oldroot=Path(cfg['old']);old=json.loads((oldroot/'manifest.json').read_text())
  assert [document_identity(r,r['text']) for r in rows]==old['documents']
  for i,row in enumerate(rows):
   if i%4==GPU:jobs.append((split,i,row,cfg,oldroot,old))
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
 model=AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct',torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 meta=ranking_metadata(model,'Qwen/Qwen2.5-7B-Instruct',list(range(28)),16,32,32,4);meta.update(sample_stride=64,attention_backend='sdpa',base_dtype='bfloat16')
 start=time.perf_counter()
 for j,(split,i,row,cfg,oldroot,old) in enumerate(jobs):
  dest=Path(cfg['output'])/'docs'/f'doc_{i:06d}'
  if dest.exists():
   receipt=json.loads((dest/'audit.json').read_text());assert receipt['index']==i and receipt['old_three_layers_bitwise_tensor_equal']
   assert json.loads((dest/'manifest.json').read_text())['documents']==[old['documents'][i]]
   for file,info in receipt['files'].items():assert (dest/file).stat().st_size==info['bytes']
   continue
  if shutil.disk_usage(ROOT).free<80*1024**3:raise RuntimeError('Disk free below reserved 80GiB; preserve old data and stop')
  staging=dest.with_name(dest.name+'.partial')
  if staging.exists():staging.rename(staging.with_name(staging.name+'.interrupted_'+str(time.time_ns())))
  write(RUN/f'worker_{GPU}_status.json',{'stage':'running','split':split,'index':i,'job_offset':j,'jobs':len(jobs),'elapsed_seconds':time.perf_counter()-start,'pid':os.getpid()})
  writer=RankingWriter(staging,meta);ids=tok(row['text'],return_tensors='pt',truncation=True,max_length=32768)['input_ids'];torch.cuda.reset_peak_memory_stats()
  t=time.perf_counter();collect_document(model,ids.to(model.device),writer,document_identity(row,row['text']),64,8);writer.finish();torch.cuda.synchronize()
  extract_seconds=time.perf_counter()-t
  receipt=audit_doc(staging,oldroot,i,old);receipt.update(tokens=ids.numel(),extract_seconds=extract_seconds,total_seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),split=split)
  write(staging/'audit.json',receipt);staging.rename(dest)
  with (RUN/f'worker_{GPU}_progress.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in receipt.items() if k not in ['files','comparison_max_abs_differences']})+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps({'split':split,'index':i,'seconds':receipt['total_seconds'],'bytes':receipt['bytes']}),flush=True);gc.collect()
 write(RUN/f'worker_{GPU}_status.json',{'stage':'completed','jobs':len(jobs),'pid':os.getpid(),'wall_seconds':time.perf_counter()-start})
if __name__=='__main__':
 try:main()
 except Exception:
  write(RUN/f'worker_{GPU}_status.json',{'stage':'failed','pid':os.getpid(),'traceback':traceback.format_exc()});raise
