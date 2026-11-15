from pathlib import Path
import os,json,subprocess,time,traceback
W=Path(__file__).parent;R=Path("/root/adablock");F=R/"results/fasa_full_reproduction_20261005_multigpu"
def save(n,x):(W/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
try:
 assert json.loads((W/"calibration/status.json").read_text())["stage"]=="completed"
 start=time.perf_counter();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="0,1,2,3",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONNOUSERSITE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
 with (W/"run.log").open("a") as log:
  p=subprocess.Popen([str(R/".venvs/fasa_full_20261005/bin/python"),str(W/"run_sparse.py")],cwd=str(W),env=env,stdout=log,stderr=subprocess.STDOUT)
  save("status.json",dict(stage="running",manager_pid=os.getpid(),worker_pid=p.pid));rc=p.wait()
 assert rc==0,dict(worker_exit=rc)
 rs=[json.loads(l) for l in (W/"run/predictions.jsonl").read_text().splitlines()]
 full=[json.loads(l) for l in (F/"run/predictions.jsonl").read_text().splitlines()]
 assert len(rs)==len(full)==200
 assert all(a["id"]==b["id"] and a["input_ids_sha256"]==b["input_ids_sha256"] for a,b in zip(rs,full))
 summary=json.loads((W/"run/summary.json").read_text())
 comparison=dict(Full_F1=sum(r["F1"] for r in full)/200,Full_EM=sum(r["EM"] for r in full)/200,FASA_F1=summary["F1"],FASA_EM=summary["EM"],delta_F1=summary["F1"]-sum(r["F1"] for r in full)/200,delta_EM=summary["EM"]-sum(r["EM"] for r in full)/200,F1_wins=sum(a["F1"]>b["F1"]+1e-9 for a,b in zip(rs,full)),F1_ties=sum(abs(a["F1"]-b["F1"])<=1e-9 for a,b in zip(rs,full)),F1_losses=sum(a["F1"]<b["F1"]-1e-9 for a,b in zip(rs,full)),EM_wins=sum(a["EM"]>b["EM"] for a,b in zip(rs,full)),EM_ties=sum(a["EM"]==b["EM"] for a,b in zip(rs,full)),EM_losses=sum(a["EM"]<b["EM"] for a,b in zip(rs,full)),token_changed=sum(a["token_ids"]!=b["token_ids"] for a,b in zip(rs,full)),first_token_changed=sum(a["token_ids"][0]!=b["token_ids"][0] for a,b in zip(rs,full)))
 save("comparison_full.json",comparison)
 changes=[dict(index=a["index"],id=a["id"],sparse=a["prediction"],full=b["prediction"],answers=a["answers"],sparse_F1=a["F1"],full_F1=b["F1"],first_token_same=a["token_ids"][0]==b["token_ids"][0]) for a,b in zip(rs,full) if a["token_ids"]!=b["token_ids"]]
 save("changed_cases.json",changes)
 selectors=[s for r in rs for s in r["budget_audit"]["layers"].values() if s["selector_calls"]]
 calls=sum(s["selector_calls"] for s in selectors)
 budget=dict(query_heads=28,kv_heads=4,head_token_budget=256,frequency_chunks=16,feature_dimensions=32,sparse_layers=list(range(1,28)),prefill_full=True,layer0_dense=True,fixed_sink_local=False,full_kv_cache_retained=True,selector_calls=calls,expected_selector_calls=27*sum(len(r["token_ids"])-1 for r in rs),per_head_count_min=min(s["head_token_count_min"] for s in selectors),per_head_count_max=max(s["head_token_count_max"] for s in selectors),per_kv_group_union_min=min(min(s["kv_union_min"]) for s in selectors),per_kv_group_union_max=max(max(s["kv_union_max"]) for s in selectors),per_kv_group_union_mean=sum(sum(s["kv_union_sum"]) for s in selectors)/(calls*4),all_query_heads_union_min=min(s["all_head_union_min"] for s in selectors),all_query_heads_union_max=max(s["all_head_union_max"] for s in selectors),all_query_heads_union_mean=sum(s["all_head_union_sum"] for s in selectors)/calls,self_token_head_fraction=sum(s["self_selected_heads"] for s in selectors)/(calls*28),token0_head_fraction=sum(s["token0_selected_heads"] for s in selectors)/(calls*28))
 assert budget["selector_calls"]==budget["expected_selector_calls"]
 save("budget_audit_summary.json",budget)
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv"],text=True)
 final=dict(worker_exit=rc,manager_wall_seconds=time.perf_counter()-start,summary=summary,comparison=comparison,budget=budget,gpu_after_exit=gpu,calibration=json.loads((W/"calibration/summary.json").read_text()))
 save("final_audit.json",final)
 report="# Official FASA sparse256 / FC16 HotpotQA200\n\n"+json.dumps(final,ensure_ascii=False,indent=2)+"\n\nSame cached model, inputs, BF16 evaluation, generation/scoring and four-GPU placement as the completed Full. Calibration uses first1 independent Qasper example per paper (official CLI default8), FP16 per official profiler, only budget256 overlaps; arithmetic mean over decode queries creates the unpublished agreement aggregation format. Other budget slots are NaN and must not be reused. No Hotpot answer tuning.\n\nOfficial Qwen selector operates after repeating four KV heads to28 query heads: each query head selects256 tokens using16 frequency chunks/32 feature dimensions. Layer0 and all prefills are dense. No mandatory sink/local/self region. Persistent KV cache remains full; union across query heads can exceed256. Finite raw logits, actual per-layer FA2/selector call counts and selected-index uniqueness/range are checked on every sample. Original selector results and FA2 returns are not modified by observers. Timing includes audit overhead and is not an optimized speedup claim. Do not compare this256-token-per-query-head budget as a fair equal-budget result against our1056 setting.\n"
 (W/"REPORT.md").write_text(report)
 with (R/"docs/ranker_full_training_20261003.md").open("a") as f:f.write("\n\n"+report+"\nArtifacts: "+str(W)+"\n")
 save("status.json",dict(stage="completed",summary=summary,comparison=comparison));print("COMPLETE",json.dumps(comparison),flush=True)
except Exception:
 save("status.json",dict(stage="failed",traceback=traceback.format_exc()));traceback.print_exc();raise
