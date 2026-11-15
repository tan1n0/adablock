from pathlib import Path
import os,sys,json,time,subprocess,traceback,datetime,hashlib
from collections import Counter
R=Path("/root/adablock");O=R/"results/longbench_ranker_all28_hotpotqa_20261004";OLD=R/"results/longbench_ranker_hotpotqa_full_20261004";PY="/root/miniconda3/envs/ada/bin/python"
sys.path.insert(0,str(R))
from utils.longbench_eval import normalize_answer
DOC=R/"docs/ranker_full_training_20261003.md";START="<!-- HOTPOT_ALL28_STATUS_START -->";END="<!-- HOTPOT_ALL28_STATUS_END -->"
def atomic(p,x):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return {}
def rows(shard):
 p=O/("shard_"+str(shard))/"learned.jsonl";rs=[]
 if p.exists():
  for l in p.read_text().splitlines():
   try:rs.append(json.loads(l))
   except json.JSONDecodeError:break
 return rs
def docstatus(states,stage):
 s=DOC.read_text();before,after=s.split(START,1);_,after=after.split(END,1)
 lines=["","阶段："+stage+"；已落盘："+str(sum(len(rows(k)) for k in range(4)))+"/200。","","| 分片/GPU | 样本数 | 状态 |","| --- | ---: | --- |"]
 for k in range(4):lines.append(f"| {k} | {len(rows(k))}/50 | {states[k]} |")
 p=DOC.with_suffix(".tmp");p.write_text(before+START+"\n".join(lines)+"\n"+END+after);p.replace(DOC)
def f1(pred,answers):
 vals=[]
 for ans in answers:
  p=normalize_answer(pred).split();g=normalize_answer(ans).split();hit=sum((Counter(p)&Counter(g)).values())
  vals.append(2*hit/(len(p)+len(g)) if hit else 0)
 return max(vals)*100
def finalize():
 rs=sorted([r for k in range(4) for r in rows(k)],key=lambda r:r["index"])
 assert [r["index"] for r in rs]==list(range(200))
 manifest=read(O/"input_manifest.json");source=read(O/"selected_rows.json")
 ref=read(O/"full_reference_receipt.json");fp=OLD/"full/full.jsonl"
 assert hashlib.sha256(fp.read_bytes()).hexdigest()==ref["prediction_sha256"]
 full=[json.loads(l) for l in fp.read_text().splitlines()]
 assert len(full)==200 and len({r["_id"] for r in source})==200
 for k in range(4):
  c=read(O/("shard_"+str(k))/"config.json")
  for key in ["max_input_length","max_new_tokens","block_size","sink_tokens","local_window_blocks","middle_blocks","seed","dtype","prompt"]:
   assert c[key]==ref["config"][key],key
 for i,r in enumerate(rs):
  assert r["input_ids_sha256"]==manifest[i]["input_ids_sha256"]==full[i]["input_ids_sha256"]
  assert r["trace"][0]["token_id"]==full[i]["trace"][0]["token_id"]
  assert r["audit"]["attention_calls"]==28*(r["stats"]["generated_tokens"]-1)
  assert set(r["audit"]["learned_calls"])=={str(k) for k in range(28)}
  assert all(v==r["stats"]["generated_tokens"]-1 for v in r["audit"]["learned_calls"].values())
  assert r["audit"]["max_selected_history_tokens"]<=1056 and r["stats"]["avg_requested_middle_blocks"]==32
  assert abs(f1(r["prediction"],source[i]["answers"])-r["score"])<1e-9
 with (O/"learned_all28.jsonl").open("x") as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 summary={"samples":200,"learned_all28_f1":sum(r["score"] for r in rs)/200,"full_reference_f1":sum(f1(r["prediction"],source[i]["answers"]) for i,r in enumerate(full))/200,"input_truncated_indices":[r["index"] for r in rs if r["input_truncated"]],"output_truncated_indices":[r["index"] for r in rs if r["output_truncated"]],"sum_sample_seconds":sum(r["seconds"] for r in rs),"peak_allocated_mib":max(r["peak_allocated_mib"] for r in rs),"peak_reserved_mib":max(r["peak_reserved_mib"] for r in rs),"generated_tokens":sum(r["stats"]["generated_tokens"] for r in rs)}
 atomic(O/"summary.json",summary)
 cases=[{"index":i,"dataset_id":source[i]["_id"],"answers":source[i]["answers"],"full":{"prediction":full[i]["prediction"],"score":full[i]["score"]},"learned_all28":{"prediction":r["prediction"],"score":r["score"]}} for i,r in enumerate(rs) if r["prediction"]!=full[i]["prediction"]]
 atomic(O/"per_sample_differences.json",cases)
 atomic(O/"completion_audit.json",{"exact_indices_0_to_199":True,"unique_ids":200,"original_checkpoint_metadata_layers":[12,20,27],"inference_ranker_layers":list(range(28)),"checkpoint_unmodified":hashlib.sha256((R/"checkpoints/ranker_ddp_fresh_global64_20261003/best.pt").read_bytes()).hexdigest()==ref["config"]["checkpoint_sha256"],"every_decode_step_all28_ranker_verified":True,"baseline_fallback_forbidden":True,"input_hashes_equal_full":True,"official_f1_independently_recomputed":True,"old_stopped_tasks_not_resumed":True})
 with DOC.open("a") as f:
  f.write("\n\n### 全28层实验最终结果\n\n"+json.dumps(summary,ensure_ascii=False,indent=2)+"\n\n完整200条无重复遗漏；各稀疏decode step均核验28层ranker调用，无baseline fallback。checkpoint训练metadata仍为12/20/27，其他25层embedding没有本轮训练监督，实验衡量原checkpoint直接跨层使用；不重训、不映射层号。full200对照复用已完成预测，配置和逐条input hash严格一致，不复用其时间做并行速度比。4卡ID分片共享CPU/内存/IO，样本耗时之和不是墙钟时间；峰值显存含模型和dense prefill，不含CPU完整KV与驱动内存。逐条差异、截断索引和完整预测见results/longbench_ranker_all28_hotpotqa_20261004。\n")
 (O/"COMPLETE").write_text("All28 full200 complete and audited")
def main():
 assert read(O/"probe/status.json").get("stage")=="probe_completed"
 assert read(OLD/"status.json").get("stage")=="stopped_by_user"
 used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
 assert all(int(x)<256 for x in used.split()),"GPUs not idle"
 fp=OLD/"full/full.jsonl"
 atomic(O/"full_reference_receipt.json",{"prediction_path":str(fp),"prediction_sha256":hashlib.sha256(fp.read_bytes()).hexdigest(),"config":read(OLD/"full/config.json"),"comparison":"quality only; old full run elapsed time not a parallel speed baseline"})
 procs={};tries={k:0 for k in range(4)};finished={};start=time.time();lastmonitor=0
 def launch(k):
  tries[k]+=1;d=O/("shard_"+str(k));d.mkdir(exist_ok=True)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(k),RANKER_EVAL_SHARD=str(k),RANKER_PROBE_ONLY="0")
  log=(d/f"attempt_{tries[k]}.log").open("x")
  p=subprocess.Popen([PY,"-u","scripts/run_ranker_all28_hotpotqa_20261004.py"],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  procs[k]=p;atomic(d/"launch.json",{"pid":p.pid,"gpu":k,"shard":k,"attempt":tries[k]})
 for k in range(4):launch(k)
 while True:
  for k,p in list(procs.items()):
   rc=p.poll()
   if rc is None or k in finished:continue
   d=O/("shard_"+str(k))
   if rc==0 and (d/"COMPLETE").exists():finished[k]="completed"
   elif tries[k]<2 and "out of memory" not in (d/f"attempt_{tries[k]}.log").read_text().lower():launch(k)
   else:finished[k]="blocked_exit_"+str(rc)
  states={k:finished.get(k,"running") for k in range(4)}
  data={"stage":"running","manager_pid":os.getpid(),"workers":{str(k):{"pid":procs[k].pid,"state":states[k],"completed_samples":len(rows(k)),"attempt":tries[k]} for k in range(4)},"elapsed_seconds":time.time()-start}
  atomic(O/"status.json",data);docstatus(states,"running")
  if time.time()-lastmonitor>=900:
   data["gpu_snapshot"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu,temperature.gpu","--format=csv"],text=True,timeout=20)
   with (O/"monitor_15min.jsonl").open("a") as f:f.write(json.dumps(data)+"\n")
   lastmonitor=time.time()
  if len(finished)==4:break
  time.sleep(30)
 if any(v!="completed" for v in finished.values()):
  atomic(O/"status.json",{"stage":"blocked","states":finished});docstatus(finished,"blocked");return
 finalize();atomic(O/"status.json",{"stage":"completed","wall_seconds":time.time()-start});docstatus(finished,"completed")
try:main()
except Exception:
 atomic(O/"status.json",{"stage":"blocked","traceback":traceback.format_exc()});raise
