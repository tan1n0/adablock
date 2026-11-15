"""Offline paired analysis of immutable saved HotpotQA predictions. No model inference."""
import json,sys,hashlib,math,re,csv
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path("/root/adablock");sys.path.insert(0,str(ROOT))
from utils.longbench_eval import normalize_answer
W=ROOT/"results/hotpotqa_curve50k_threeway_20261005"
OUT=Path(__file__).parent
SEED=20261005; B=20000
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[]
def load(p):
 sources.append(dict(path=str(p),sha256=sha(p)))
 return json.loads(p.read_text())
rows=load(W/"selected_rows.json");manifest=load(W/"reference_input_manifest.json")
def journal(p):
 sources.append(dict(path=str(p),sha256=sha(p)))
 return [json.loads(l) for l in p.read_text().splitlines()]
def shards(d,m):
 a=[]
 for p in sorted(d.glob("shard_*/"+m+".jsonl")):a.extend(journal(p))
 return sorted(a,key=lambda x:x["index"])
data={m:shards(W/m,m) for m in ["learned","cobs","quest"]}
data["full"]=journal(ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full/full.jsonl")
data["old"]=shards(ROOT/"results/longbench_ranker_all28_hotpotqa_20261004","learned")
def score(pred,answers):
 p=normalize_answer(pred).split()
 vals=[]
 for ans in answers:
  g=normalize_answer(ans).split();common=sum((Counter(p)&Counter(g)).values())
  vals.append(200*common/(len(p)+len(g)) if common else 0.)
 return max(vals),100.*any(normalize_answer(pred)==normalize_answer(a) for a in answers)
assert len(rows)==len(manifest)==200 and len({r["_id"] for r in rows})==200
metrics={};maxerr=0.;per=[]
for method,rs in data.items():
 assert len(rs)==200 and [r["index"] for r in rs]==list(range(200))
 vals=[]
 for i,r in enumerate(rs):
  assert r["input_ids_sha256"]==manifest[i]["input_ids_sha256"]
  assert r["answers"]==rows[i]["answers"]
  if r.get("sample_id") is not None:assert r["sample_id"]==rows[i]["_id"]
  f,e=score(r["prediction"],r["answers"]);maxerr=max(maxerr,abs(f-r["score"]))
  assert abs(f-r["score"])<1e-10
  vals.append([f,e])
  per.append(dict(index=i,sample_id=rows[i]["_id"],method=method,F1=f,EM=e,prediction=r["prediction"],input_hash=r["input_ids_sha256"],truncated=r["output_truncated"]))
 metrics[method]=np.asarray(vals)
assert load(ROOT/"results/longbench_ranker_all28_hotpotqa_20261004/selected_rows.json")==rows
assert load(ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full/selected_rows.json")==rows
configs={m:load(W/m/"shard_0/config.json") for m in ["learned","cobs","quest"]}
configs["old"]=load(ROOT/"results/longbench_ranker_all28_hotpotqa_20261004/shard_0/config.json")
configs["full"]=load(ROOT/"results/longbench_ranker_hotpotqa_full_20261004/full/config.json")
keys=["prompt","requested_context_limit","model_total_positions","max_input_length","max_new_tokens","block_size","sink_tokens","local_window_blocks","middle_blocks","seed","dtype","prefill","decode"]
protocol={m:{k:configs[m][k]==configs["learned"][k] for k in keys} for m in data}
assert all(all(v.values()) for v in protocol.values())
source_checks={m:{k:h==configs["learned"]["source_sha256"].get(k) for k,h in configs[m]["source_sha256"].items()} for m in data}
assert all(all(v.values()) for v in source_checks.values())
dump("alignment_audit.json",dict(samples=200,unique_source_ids=200,all_index_input_reference_checks=True,archived_score_max_absolute_error=maxerr,protocol=protocol,common_source_hashes_match=source_checks,normalization_source=sha(ROOT/"utils/longbench_eval.py"),legacy_rows_have_no_sample_id="For legacy records index+exact selected_rows+input hash+references establish ID mapping."))
dump("source_manifest.json",sources)
dump("per_question_scores.json",per)
sets={m:np.flatnonzero(np.abs(metrics["learned"][:,0]-metrics[m][:,0])>1e-10).tolist() for m in ["full","cobs","quest","old"]}
sets["full_losses"]=np.flatnonzero(metrics["learned"][:,0]<metrics["full"][:,0]-1e-10).tolist()
primary=["cobs","quest","full_losses"]
intersections={a+"&"+b:sorted(set(sets[a])&set(sets[b])) for j,a in enumerate(primary) for b in primary[j+1:]}
intersections["all_three"]=sorted(set.intersection(*(set(sets[k]) for k in primary)))
union=sorted(set.union(*(set(sets[k]) for k in ["full","cobs","quest","old"])))
dump("difference_sets.json",dict(sets=sets,primary_pairwise_intersections=intersections,primary_union=sorted(set.union(*(set(sets[k]) for k in primary))),all_four_comparisons_union=union,all_four_count=len(union),raw_text_difference_indices={m:[i for i in range(200) if data["learned"][i]["prediction"]!=data[m][i]["prediction"]] for m in ["full","cobs","quest","old"]}))
rng=np.random.default_rng(SEED)
bootstrap={m:[] for m in ["full","cobs","quest","old"]}
for start in range(0,B,500):
 idx=rng.integers(0,200,size=(min(500,B-start),200))
 for m in bootstrap:bootstrap[m].append((metrics["learned"]-metrics[m])[idx].mean(axis=1))
stats={}
for m,bs in bootstrap.items():
 delta=metrics["learned"]-metrics[m];ci=np.quantile(np.concatenate(bs),[.025,.975],axis=0,method="linear")
 out={}
 for j,name in enumerate(["F1","EM"]):
  d=delta[:,j];wins=int((d>1e-10).sum());losses=int((d<-1e-10).sum());n=wins+losses
  p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
  out[name]=dict(mean_difference=float(d.mean()),percentile95=[float(ci[0,j]),float(ci[1,j])],wins=wins,ties=200-n,losses=losses,exact_two_sided_sign_p=p)
 stats[m]=out
dump("paired_statistics.json",dict(seed=SEED,replicates=B,unit="question; same resampled indices for all methods and both metrics",method="paired nonparametric percentile bootstrap, numpy quantile linear 2.5/97.5 percentiles, differences in percentage points",caveat="Intervals are pointwise, not multiplicity adjusted; diagnostics on this fixed test; sign tests auxiliary.",comparisons_learned_minus=stats))
truncated=sorted({i for rs in data.values() for i,r in enumerate(rs) if r["output_truncated"]})
assert truncated==[95]
keep=[i for i in range(200) if i not in truncated]
sensitivity={}
for m in bootstrap:
 d=metrics["learned"]-metrics[m]
 sensitivity[m]=dict(primary200_F1=float(d[:,0].mean()),primary200_EM=float(d[:,1].mean()),excluded_indices=truncated,paired199_F1=float(d[keep,0].mean()),paired199_EM=float(d[keep,1].mean()),excluded_F1_contribution_to_200_mean=float(d[truncated,0].sum()/200))
dump("truncation_sensitivity.json",dict(diagnostic_only=True,all_methods_exclude_same_questions=True,comparisons=sensitivity))
dump("summary.json",{m:dict(F1=float(v[:,0].mean()),EM=float(v[:,1].mean())) for m,v in metrics.items()})
annotations=json.loads((OUT/"annotations.json").read_text())
assert set(map(int,annotations))==set(union)
cases=[];ctxdir=OUT/"contexts";ctxdir.mkdir(exist_ok=True)
for i in union:
 row=rows[i];ctx=row["context"];ann=annotations[str(i)];evidence=[]
 passages=list(re.finditer(r"Passage (\d+):\n([^\n]+)",ctx))
 for term in ann["anchors"]:
  matches=list(re.finditer(re.escape(term),ctx,re.I))
  for m in matches[:4]:
   a=max(0,m.start()-160);b=min(len(ctx),m.end()+330)
   passage=next((p for p in reversed(passages) if p.start()<=m.start()),None)
   evidence.append(dict(term=term,term_count=len(matches),match_char=m.start(),excerpt_start=a,excerpt_end=b,line=ctx.count("\n",0,m.start())+1,passage=int(passage.group(1)) if passage else None,title=passage.group(2) if passage else None,text=ctx[a:b]))
  if not matches:evidence.append(dict(term=term,term_count=0))
 (ctxdir/(str(i).zfill(3)+".txt")).write_text(ctx)
 cases.append(dict(index=i,id=row["_id"],question=row["input"],references=row["answers"],context_path="contexts/"+str(i).zfill(3)+".txt",context_sha256=hashlib.sha256(ctx.encode()).hexdigest(),comparisons=[m for m in ["full","cobs","quest","old"] if i in sets[m]],predictions={m:dict(text=rs[i]["prediction"],F1=float(metrics[m][i,0]),EM=float(metrics[m][i,1]),truncated=rs[i]["output_truncated"],generated_tokens=rs[i]["stats"]["generated_tokens"]) for m,rs in data.items()},review=ann,evidence=evidence))
dump("review_cases.json",cases)
md=["# HotpotQA prediction review","All indices are zero-based. Original scoring is unchanged. Evidence offsets are Unicode character offsets in the exact context; lines are one-based. Labels are qualitative review, not new benchmark scores.\n"]
for c in cases:
 md += ["## Question "+str(c["index"]),c["question"],"ID: "+c["id"],"Reference: "+json.dumps(c["references"],ensure_ascii=False),"Diagnosis: "+c["review"]["diagnosis"],"Evidence strength / limits: "+c["review"]["confidence"],"Context: ["+c["context_path"]+"]("+c["context_path"]+")"]
 for m,p in c["predictions"].items():md += ["### "+m+" | F1 "+str(round(p["F1"],6))+" | EM "+str(p["EM"])+" | truncated "+str(p["truncated"]),p["text"]]
 md += ["### Evidence"]
 for e in c["evidence"]:
  if e["term_count"]:md += ["- Passage "+str(e["passage"])+" ("+str(e["title"])+"), line "+str(e["line"])+", chars ["+str(e["excerpt_start"])+","+str(e["excerpt_end"])+"): "+json.dumps(e["text"],ensure_ascii=False)]
  else:md += ["- Exact case-insensitive anchor absent: "+e["term"]+". Absence alone is not proof no equivalent statement exists."]
(OUT/"REVIEW_ALL_DIFFERENCES.md").write_text("\n\n".join(md))
print(json.dumps(dict(summary={m:dict(F1=float(v[:,0].mean()),EM=float(v[:,1].mean())) for m,v in metrics.items()},sets=sets,intersections=intersections,primary_union=sorted(set.union(*(set(sets[k]) for k in primary))),union=union,statistics=stats,sensitivity=sensitivity),ensure_ascii=False,indent=2))
