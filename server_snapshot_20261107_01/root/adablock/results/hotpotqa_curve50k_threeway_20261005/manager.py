import os,sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
ROOT=Path("/root/adablock");sys.path.insert(0,str(ROOT))
from utils.longbench_eval import normalize_answer
W=ROOT/"results/hotpotqa_curve50k_threeway_20261005"
PY="/root/miniconda3/envs/ada/bin/python"
RUNNER=ROOT/"scripts/run_hotpotqa_curve50k_threeway_20261005.py"
DOC=ROOT/"docs/ranker_full_training_20261003.md"
def write(name,x):
 p=W/name;t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2));t.replace(p)
def note(x):
 with DOC.open("a") as f:f.write("\n\n"+x+"\n")
def records(method):
 rs=[]
 for shard in range(4):
  p=W/method/("shard_"+str(shard))/(method+".jsonl")
  if p.exists():
   for l in p.read_text().splitlines():
    try:rs.append(json.loads(l))
    except json.JSONDecodeError:pass
 return sorted(rs,key=lambda x:x["index"])
def summary(rs):
 n=len(rs)
 return dict(samples=n,F1=sum(r["score"] for r in rs)/n,EM=sum(100*any(normalize_answer(r["prediction"])==normalize_answer(a) for a in r["answers"]) for r in rs)/n,input_truncation_rate=sum(r["input_truncated"] for r in rs)/n,output_truncation_rate=sum(r["output_truncated"] for r in rs)/n,output_limit_rate=sum(r["output_hit_limit"] for r in rs)/n,sample_seconds=sum(r["seconds"] for r in rs),generated_tokens=sum(r["stats"]["generated_tokens"] for r in rs),peak_reserved_mib=max(r["peak_reserved_mib"] for r in rs))
def main():
 note("## Authorized frozen50k HotpotQA three-way full200 run\nIndependent directory: "+str(W)+". Learned then COBS then Quest,4-way sample shards, cached Qwen revision a09a35458c702b33eeacc393d103063234e8bc28,32768 context/32640 input/128 greedy output,block16,sink32tokens,local32blocks,middle32,full SDPA prefill,all28 layers. Frozen best SHA d76365ed6317fa73fdbc12ddc520fffab8e70e3c304686bb634936bb0d3862ae. Per-sample fsync journals and input-hash equality gates; no checkpoint tuning. Historical Full is quality-only baseline, not a speed comparison.")
 allrs={};sums={}
 reference=json.loads((W/"reference_input_manifest.json").read_text())
 fullpath=ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full/full.jsonl"
 full=[json.loads(l) for l in fullpath.read_text().splitlines()]
 assert len(full)==200 and [r["index"] for r in full]==list(range(200))
 assert all(r["input_ids_sha256"]==reference[i]["input_ids_sha256"] for i,r in enumerate(full))
 allrs["full"]=full;sums["full"]=summary(full)
 for method in ["learned","cobs","quest"]:
  start=time.perf_counter();procs={};attempts={i:0 for i in range(4)}
  def launch(i):
   d=W/method/("shard_"+str(i));d.mkdir(parents=True,exist_ok=True)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(i),RANKER_EVAL_SHARD=str(i),RANKER_EVAL_METHOD=method,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
   attempts[i]+=1
   log=(d/("attempt_"+str(attempts[i])+".log")).open("a")
   p=subprocess.Popen([PY,str(RUNNER)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
   procs[i]=p
  for i in range(4):launch(i)
  while True:
   for i,p in list(procs.items()):
    code=p.poll()
    if code is not None:
     if code!=0:
      if attempts[i]<2:launch(i)
      else:raise RuntimeError(str((method,i,code)))
   rs=records(method)
   write("status.json",dict(stage="running",method=method,completed=len(rs),pids={i:p.pid for i,p in procs.items()},attempts=attempts,elapsed_seconds=time.perf_counter()-start,summaries=sums))
   if all(p.poll()==0 for p in procs.values()):break
   time.sleep(20)
  rs=records(method)
  assert [r["index"] for r in rs]==list(range(200))
  for i,r in enumerate(rs):
   assert r["input_ids_sha256"]==reference[i]["input_ids_sha256"]
   assert set(r["audit"]["layer_calls"])==set(map(str,range(28)))
   assert r["stats"]["avg_requested_middle_blocks"]==32
   if method=="learned":assert set(r["audit"]["learned_calls"])==set(map(str,range(28)))
  allrs[method]=rs;sums[method]=summary(rs);sums[method]["wall_seconds"]=time.perf_counter()-start
  write(method+"_summary.json",sums[method]);write(method+"_merged.json",rs)
  write("summaries.json",sums);note("Completed "+method+" full200: "+json.dumps(sums[method]))
 pairs={}
 for a,b in [("learned","full"),("cobs","full"),("quest","full"),("learned","cobs"),("learned","quest")]:
  aa,bb=allrs[a],allrs[b]
  dif=[x["score"]-y["score"] for x,y in zip(aa,bb)]
  pairs[a+"_minus_"+b]=dict(mean_F1_difference=sum(dif)/200,F1_wins=sum(d>1e-10 for d in dif),F1_ties=sum(abs(d)<=1e-10 for d in dif),F1_losses=sum(d<-1e-10 for d in dif),mean_EM_difference=sums[a]["EM"]-sums[b]["EM"],per_sample_F1_difference=dif)
 write("paired_comparisons.json",pairs)
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],text=True)
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True)
 write("gpu_release_receipt.json",dict(gpus=gpu,compute_apps=apps))
 write("status.json",dict(stage="completed",summaries=sums))
 (W/"COMPLETE").write_text("All3 methods complete200; paired results and GPU receipt saved.")
 note("Completed all3 full200; paired comparisons: "+json.dumps({k:{a:b for a,b in v.items() if a!="per_sample_F1_difference"} for k,v in pairs.items()})+"\nGPU receipt: "+gpu+"\nAnswer F1/EM measure QA quality; attention coverage is a separate earlier evaluation. No paper-exact or speedup claim.")
if __name__=="__main__":
 try:main()
 except Exception:
  write("manager_error.json",dict(traceback=traceback.format_exc()));raise
