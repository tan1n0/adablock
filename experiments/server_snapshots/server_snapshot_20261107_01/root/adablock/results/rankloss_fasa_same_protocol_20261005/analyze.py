"""Read-only paired analysis of the four frozen 200-sample runs."""
from pathlib import Path
import json,hashlib,itertools,numpy as np
W=Path(__file__).parent;R=W.parent
paths={"Full":R/"fasa_full_reproduction_20261005_multigpu/run",
"FASA":R/"fasa_sparse_native256_20261005/run",
"RankLoss_layer0_full":W/"layer0_full/run","RankLoss_all_layers":W/"all_layers/run"}
rows={name:[json.loads(l) for l in (p/"predictions.jsonl").read_text().splitlines()] for name,p in paths.items()}
full=rows["Full"]
for name,r in rows.items():
 assert len(r)==200 and [x["index"] for x in r]==list(range(200))
 assert [(x["id"],x["input_ids_sha256"]) for x in r]==[(x["id"],x["input_ids_sha256"]) for x in full]
 assert all(x["token_ids"][0]==y["token_ids"][0] for x,y in zip(r,full))
assert hashlib.sha256((paths["FASA"]/"predictions.jsonl").read_bytes()).hexdigest()==json.loads((W/"fasa_reuse_audit.json").read_text())["prediction_sha256"]
summaries={}
for name,r in rows.items():
 summaries[name]=dict(samples=200,F1=float(np.mean([x["F1"] for x in r])),EM=float(np.mean([x["EM"] for x in r])),
  input_truncated=0,output_truncated=sum(x["output_truncated"] for x in r),
  cap_indices=[x["index"] for x in r if x["output_truncated"]],ended_eos=sum(x["ended_eos"] for x in r),
  sample_seconds=sum(x["seconds"] for x in r),prediction_sha256=hashlib.sha256((paths[name]/"predictions.jsonl").read_bytes()).hexdigest())
comparisons={}
rng=np.random.default_rng(42);bootstrap=rng.integers(0,200,size=(10000,200))
for A,B in itertools.combinations(rows,2):
 pair={}
 for metric in ("F1","EM"):
  delta=np.array([a[metric]-b[metric] for a,b in zip(rows[A],rows[B])])
  pair[metric]=dict(delta_A_minus_B=float(delta.mean()),wins_A=int((delta>1e-9).sum()),
   ties=int((abs(delta)<=1e-9).sum()),losses_A=int((delta< -1e-9).sum()),
   paired_bootstrap_95pct=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist())
 pair["changed_token_sequences"]=[i for i,(a,b) in enumerate(zip(rows[A],rows[B])) if a["token_ids"]!=b["token_ids"]]
 pair["first_token_changed"]=sum(a["token_ids"][0]!=b["token_ids"][0] for a,b in zip(rows[A],rows[B]))
 comparisons[A+" minus "+B]=pair
audits={}
for variant in ("layer0_full","all_layers"):
 name="RankLoss_"+variant;rs=rows[name];learned=set(range(1 if variant=="layer0_full" else 0,28))
 for x in rs:
  b=x["budget_audit"];expected=len(x["token_ids"])-1;target=[expected if i in learned else 0 for i in range(28)]
  assert b["prefill_calls"]==[1]*28 and b["decode_calls"]==[expected]*28
  assert b["learned_selector_calls"]==target==b["learned_score_calls"]
  assert b["ranker_forward_calls"]==[4*v for v in target]
  assert b["fallback_calls"]==0 and b["full_cache_preserved"] and b["positions_checked"]
 audits[variant]=dict(actual_learned_layers=sorted(learned),actual_selector_calls=sum(sum(x["budget_audit"]["learned_selector_calls"]) for x in rs),
  actual_ranker_calls=sum(sum(x["budget_audit"]["ranker_forward_calls"]) for x in rs),
  fallback_calls=0,first_tokens_all_match_full=True,full_cache_preserved=True,
  historical_tokens_min=min(x["budget_audit"]["historical_tokens_min"] for x in rs if x["budget_audit"]["historical_tokens_min"] is not None),
  historical_tokens_max=max(x["budget_audit"]["historical_tokens_max"] for x in rs if x["budget_audit"]["historical_tokens_max"] is not None),
  total_tokens_min=min(x["budget_audit"]["total_with_self_min"] for x in rs if x["budget_audit"]["total_with_self_min"] is not None),
  total_tokens_max=max(x["budget_audit"]["total_with_self_max"] for x in rs if x["budget_audit"]["total_with_self_max"] is not None),
  devices=sorted({v["device"] for x in rs for v in x["budget_audit"]["examples"].values()}))
result=dict(summaries=summaries,pairs=comparisons,rankloss_audits=audits,
 inference="Paired bootstrap resamples the same 200 examples; exploratory uncertainty, not a new benchmark.",
 limitations=["FASA256 per query head and RankLoss66 historical blocks per KV head plus current self are native unequal budgets.",
 "All methods preserve full KV cache. Adapter uses GPU cache gathering and FA2; prototype used CPU history and SDPA.",
 "Elapsed time includes digest construction, Python ranking, synchronizations and audits; not fused-kernel throughput.",
 "Only 200 official HotpotQA examples and one checkpoint; no training or budget tuning in this experiment."])
(W/"comparison.json").write_text(json.dumps(result,indent=2))
lines=["# RankLoss and FASA under the same generation protocol","",
"Frozen 50k checkpoint; official 200 HotpotQA inputs; Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28. Transformers 4.43.0, Torch 2.5.1+cu121, FlashAttention2 2.7.4.post1, BF16, same four-GPU placement. Greedy generation, repetition penalty 1.05, max_new_tokens 32, dual EOS 151645/151643, min_length=input_length+1. Full and FASA results reused after protocol/hash verification.","",
"| Configuration | F1 | EM | Output capped | Sample seconds |","|---|---:|---:|---:|---:|"]
for name,s in summaries.items():lines.append(f"| {name} | {s['F1']:.4f} | {s['EM']:.2f} | {s['output_truncated']} | {s['sample_seconds']:.1f} |")
lines+=["","All 200 first tokens match Full in every configuration. Prefill is native dense attention. Every generated decode step was audited for expected layer calls, no COBS fallback, position/cache consistency, finite outputs and native budgets.","","## Paired differences (A minus B)",""]
for pair,p in comparisons.items():
 f=p["F1"];e=p["EM"]
 lines.append(f"- {pair}: F1 {f['delta_A_minus_B']:+.4f} pp (wins/ties/losses {f['wins_A']}/{f['ties']}/{f['losses_A']}; bootstrap95% {f['paired_bootstrap_95pct']}), EM {e['delta_A_minus_B']:+.2f} pp; {len(p['changed_token_sequences'])} changed token sequences.")
lines+=["","Layer-0 intervention is isolated by the two RankLoss variants; generation and the remaining 27 layers use identical code and weights. Interpret the paired score difference on this finite set rather than a universal effect.","","## Validation and limitations","",
"Synthetic tests compare exact selected indices and incremental digests against the frozen original controller at short, boundary and partial-block lengths on GPUs 1, 2 and 3. GQA mapping and full cache are checked exactly; selected FA2 attention agrees with FP32 softmax within max absolute 0.002 and relative L2 0.24%. Both variants pass separate first-eight-example smoke gates before 200-example continuation.",""]
lines += ["- "+x for x in result["limitations"]]
lines+=["","Artifacts: adapter_tests.json, fasa_reuse_audit.json, source_hashes.json, rankloss_adapter.py, manager.py, each variant's run/config.json, run/rankloss_runtime_audit.json, run/smoke8_passed.json, run/predictions.jsonl, run/summary.json, and comparison.json.",""]
(W/"REPORT.md").write_text("\n".join(lines))
print(json.dumps(result),flush=True)
