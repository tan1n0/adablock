import os,sys,json,time,subprocess,fcntl,hashlib,shutil,traceback,collections
from pathlib import Path
ROOT=Path('/root/adablock');RUN=ROOT/'results/ranker_all28_build_20261004';PY='/root/miniconda3/envs/ada/bin/python'
def write(p,obj):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def doc_status(text):
 p=ROOT/'docs/ranker_full_training_20261003.md';s=p.read_text();a='<!-- ALL28_DATA_BUILD_START -->';b='<!-- ALL28_DATA_BUILD_END -->';block=a+'\n'+text+'\n'+b
 if a in s:s=s[:s.index(a)]+block+s[s.index(b)+len(b):]
 else:s+='\n\n### 全28层rankloss数据构造（独立目录）\n\n'+block+'\n'
 p.write_text(s)
def counts(plan):
 out={}
 for split in ['train','validation']:
  ds=Path(plan[split]['output'])/'docs'
  paths=[p for p in ds.glob('doc_*/audit.json') if len(p.parent.name)==10 and p.parent.name[4:].isdigit()]
  out[split]={'complete_documents':len(paths),'expected_documents':plan[split]['documents'],'bytes':sum(json.loads(p.read_text())['bytes'] for p in paths)}
 return out
def finalize(plan):
 summaries={}
 for split in ['train','validation']:
  cfg=plan[split];dest=Path(cfg['output']);old=json.loads((Path(cfg['old'])/'manifest.json').read_text());entries=[];scenes=collections.Counter();totalbytes=0;maxerr=0
  for i,identity in enumerate(old['documents']):
   d=dest/'docs'/f'doc_{i:06d}';man=json.loads((d/'manifest.json').read_text());audit=json.loads((d/'audit.json').read_text())
   assert man['documents']==[identity] and man['metadata']==dict(old['metadata'],layers=list(range(28)))
   assert audit['old_three_layers_bitwise_tensor_equal'] and audit['finite_and_causal']
   assert {e['layer_id'] for e in man['shards']}==set(range(28))
   for e in man['shards']:
    assert (d/e['file']).stat().st_size==audit['files'][e['file']]['bytes']
    entries.append(dict(e,file=str((d/e['file']).relative_to(dest)),document_index=i));scenes[e['layer_id']]+=e['scenes']
   totalbytes+=audit['bytes'];maxerr=max(maxerr,audit['mass_normalization_max_error'])
  assert len(entries)==len(old['documents'])*28 and len(set(scenes.values()))==1
  expected=sum(e['scenes'] for e in old['shards'] if e['layer_id']==12)
  assert all(v==expected for v in scenes.values())
  write(dest/'manifest.json',{'metadata':dict(old['metadata'],layers=list(range(28))),'documents':old['documents'],'shards':entries})
  summaries[split]={'documents':len(old['documents']),'shards':len(entries),'scenes':sum(scenes.values()),'scenes_per_layer':dict(scenes),'bytes':totalbytes,'max_mass_normalization_error':maxerr,'all_shards_finite_causal_and_old3_exact':True}
 assert all(sha(path)==digest for path,digest in plan['source_sha256'].items())
 write(RUN/'completion_audit.json',{'summary':summaries,'isolation':plan['isolation'],'source_hashes_unchanged':True,'all_documents_atomic_committed':True,'all_shards_hashed_at_commit':True,'final_size_check_passed':True,'no_training_started':True})
 write(RUN/'summary.json',summaries);doc_status('阶段：completed。全部28层训练及匹配验证已完成；原三层所有特征/标签逐元素一致，finite/质量归一化/无未来候选审计通过。\n\n'+json.dumps(summaries,ensure_ascii=False,indent=2)+'\n\n输出及审计见 '+str(RUN)+'；没有启动训练。')
 (RUN/'COMPLETE').write_text('Full28 train and validation extraction complete, audited, no training')
def main():
 lock=(RUN/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((RUN/'plan.json').read_text())
 assert (RUN/'pilot_COMPLETE').exists()
 pilot=json.loads((RUN/'pilot_results.json').read_text())
 assert all(max(v for d in p['comparison_max_abs_differences'].values() for v in d.values())==0 for p in pilot)
 assert all(sha(path)==digest for path,digest in plan['source_sha256'].items())
 procs={};start=time.perf_counter()
 for gpu in range(4):
  f=(RUN/f'worker_{gpu}.log').open('a')
  procs[gpu]=subprocess.Popen([PY,'-u',str(RUN/'worker.py'),str(gpu)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT))
 while True:
  c=counts(plan);status={'stage':'running','manager_pid':os.getpid(),'workers':{str(g):{'pid':p.pid,'exit_code':p.poll()} for g,p in procs.items()},'counts':c,'wall_seconds':time.perf_counter()-start,'disk_free_bytes':shutil.disk_usage(ROOT).free}
  write(RUN/'status.json',status)
  doc_status('阶段：running；train '+str(c['train']['complete_documents'])+'/1751，validation '+str(c['validation']['complete_documents'])+'/16。真实四卡提取，按序列原子提交；中断后重新运行manager.py恢复，已完成序列不重算。不会修改30分钟闲置关机策略；自动关机判定未知。\n\n'+json.dumps(status,ensure_ascii=False,indent=2))
  if all(p.poll() is not None for p in procs.values()):break
  time.sleep(30)
 if any(p.returncode!=0 for p in procs.values()):
  status['stage']='failed_or_partial';write(RUN/'status.json',status);doc_status('阶段：failed_or_partial，保留全部已提交序列和错误日志。'+json.dumps(status,ensure_ascii=False));return
 finalize(plan);write(RUN/'status.json',{'stage':'completed','wall_seconds':time.perf_counter()-start,'counts':counts(plan)})
if __name__=='__main__':
 try:main()
 except Exception:
  write(RUN/'manager_error.json',{'traceback':traceback.format_exc()});raise
