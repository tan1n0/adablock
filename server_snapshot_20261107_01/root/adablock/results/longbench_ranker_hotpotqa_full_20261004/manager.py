from pathlib import Path
import os,sys,json,time,subprocess,traceback,datetime,hashlib
R=Path("/root/adablock");O=R/"results/longbench_ranker_hotpotqa_full_20261004";PY="/root/miniconda3/envs/ada/bin/python"
NAMES=["full","cobs","quest","learned"];DOC=R/"docs/ranker_full_training_20261003.md"
START="<!-- HOTPOT_FULL_STATUS_START -->";END="<!-- HOTPOT_FULL_STATUS_END -->"
def atomic(p,obj):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return {}
def rows(n):
 p=O/n/(n+".jsonl")
 if not p.exists():return []
 result=[]
 for l in p.read_text().splitlines():
  try:result.append(json.loads(l))
  except json.JSONDecodeError:break
 return result
def docstatus(states,stage):
 text=DOC.read_text();before,after=text.split(START,1);_,after=after.split(END,1)
 lines=["","阶段："+stage+"；最近服务器时刻："+datetime.datetime.now().astimezone().isoformat()+"（服务器钟有偏差）。","","| 方法 | 已落盘样本 | 状态 |","| --- | ---: | --- |"]
 for n in NAMES:lines.append(f"| {n} | {len(rows(n))}/200 | {states[n]} |")
 temp=DOC.with_suffix(".tmp");temp.write_text(before+START+"\n".join(lines)+"\n"+END+after);temp.replace(DOC)
def finalize():
 allrows={n:rows(n) for n in NAMES};manifest=read(O/"input_manifest.json")
 assert all(len(rs)==200 for rs in allrows.values())
 summaries={};cases=[]
 for n,rs in allrows.items():
  for i,r in enumerate(rs):
   assert r["index"]==i and r["input_ids_sha256"]==manifest[i]["input_ids_sha256"]
   assert r["trace"][0]["token_id"]==allrows["full"][i]["trace"][0]["token_id"]
   if n!="full":
    assert r["stats"]["avg_requested_middle_blocks"]==32
    assert r["audit"]["max_selected_history_tokens"]<=1056
    assert r["audit"]["attention_calls"]==28*(r["stats"]["generated_tokens"]-1)
   if n=="learned":assert set(r["audit"]["learned_calls"])=={"12","20","27"}
  summaries[n]={"samples":200,"official_qa_f1":sum(r["score"] for r in rs)/200,"measured_seconds":sum(r["seconds"] for r in rs),"peak_allocated_mib":max(r["peak_allocated_mib"] for r in rs),"peak_reserved_mib":max(r["peak_reserved_mib"] for r in rs),"input_truncated_indices":[r["index"] for r in rs if r["input_truncated"]],"output_truncated_indices":[r["index"] for r in rs if r["output_truncated"]],"generated_tokens":sum(r["stats"]["generated_tokens"] for r in rs)}
 for i in range(200):
  values={n:{"prediction":rs[i]["prediction"],"score":rs[i]["score"],"output_truncated":rs[i]["output_truncated"]} for n,rs in allrows.items()}
  if len({v["prediction"] for v in values.values()})>1:
   cases.append({"index":i,"dataset_id":manifest[i]["dataset_id"],"answers":allrows["full"][i]["answers"],"methods":values})
 atomic(O/"summary.json",summaries);atomic(O/"per_sample_differences.json",cases)
 atomic(O/"completion_audit.json",{"counts":{n:len(rs) for n,rs in allrows.items()},"input_hashes_match":True,"budgets_and_attention_calls_verified":True,"checkpoint":"epoch7","learned_scope":"layers12/20/27; other layers COBS","full_task":True,"reused_old_records":False,"timing_scope":"Concurrent methods on four GPUs; shared CPU/memory/I/O contention, one run only, not isolated inference speed; per-sample includes prefill+digest+decode+traces; excludes model load/probes/retry overhead","memory_scope":"PyTorch allocator peak including weights and dense prefill; excludes CPU KV and CUDA driver memory"})
 lines=["","### 全量HotpotQA完成结果","","| 方法 | 样本数 | 官方QA F1 | 样本累计秒数 | 峰值allocated MiB | 输入截断数 | 输出截断数 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for n,v in summaries.items():lines.append(f"| {n} | 200 | {v['official_qa_f1']:.4f} | {v['measured_seconds']:.2f} | {v['peak_allocated_mib']:.1f} | {len(v['input_truncated_indices'])} | {len(v['output_truncated_indices'])} |")
 lines+=["","Learned仅12/20/27层，其他层COBS；未训练新模型。四方案完整200条、相同输入哈希及预算路径核验通过。所有新样本重新生成，未复用旧8条预测。并行共享CPU/内存/IO，时间仅作带竞争的运行参考；包含prefill和摘要构造，不是单独decode吞吐。峰值显存包括权重与dense prefill，完整KV仍存CPU；不宣称稀疏prefill或永久KV驱逐。逐条差异及截断索引见results/longbench_ranker_hotpotqa_full_20261004/{summary.json,per_sample_differences.json,completion_audit.json}。"]
 with DOC.open("a") as f:f.write("\n".join(lines)+"\n")
 (O/"COMPLETE").write_text("All 4 methods 200 samples and final audit complete")
def main():
 used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
 assert len(used.split())==4 and all(int(x)<256 for x in used.split()),"GPUs are no longer idle"
 procs={};attempts={n:0 for n in NAMES};finished={};started=time.time()
 def launch(n):
  attempts[n]+=1;folder=O/n;folder.mkdir(exist_ok=True)
  env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(NAMES.index(n));env["RANKER_EVAL_METHOD"]=n;env["HF_HUB_OFFLINE"]="1";env["TRANSFORMERS_OFFLINE"]="1"
  log=(folder/f"attempt_{attempts[n]}.log").open("x")
  p=subprocess.Popen([PY,"-u","scripts/run_ranker_hotpotqa_full_20261004.py"],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  procs[n]=p
  atomic(folder/"launch.json",{"pid":p.pid,"attempt":attempts[n],"physical_gpu":NAMES.index(n),"command":[PY,"-u","scripts/run_ranker_hotpotqa_full_20261004.py"],"environment":{"CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"],"RANKER_EVAL_METHOD":n}})
 for n in NAMES:launch(n)
 while True:
  for n,p in list(procs.items()):
   rc=p.poll()
   if rc is None:continue
   if n in finished:continue
   if rc==0 and (O/n/"COMPLETE").exists():finished[n]="completed"
   elif attempts[n]<2 and "out of memory" not in (O/n/f"attempt_{attempts[n]}.log").read_text().lower():
    launch(n)
   else:finished[n]="blocked_exit_"+str(rc)
  states={n:finished.get(n,"running") for n in NAMES}
  atomic(O/"status.json",{"stage":"running","manager_pid":os.getpid(),"workers":{n:{"pid":p.pid,"gpu":NAMES.index(n),"attempt":attempts[n],"state":states[n],"completed_samples":len(rows(n))} for n,p in procs.items()},"elapsed_seconds":time.time()-started})
  docstatus(states,"running")
  if len(finished)==4:break
  time.sleep(30)
 if any(s!="completed" for s in finished.values()):
  atomic(O/"status.json",{"stage":"blocked","states":finished});docstatus(finished,"blocked");return
 finalize();atomic(O/"status.json",{"stage":"completed","seconds":time.time()-started,"methods":NAMES});docstatus(finished,"completed")
try:main()
except Exception:
 atomic(O/"status.json",{"stage":"blocked","traceback":traceback.format_exc()});raise
