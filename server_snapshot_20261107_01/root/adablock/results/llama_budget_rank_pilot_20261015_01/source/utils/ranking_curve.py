"""Fixed balanced learning-curve plan; existing scene loss and score semantics."""
import json,hashlib,collections
from pathlib import Path
import numpy as np
import torch
from bisect import bisect_right
from utils.ranking_batch import BatchRankingDataset

class MMapDataset(BatchRankingDataset):
    def __init__(self,directory,with_baselines=False):
        super().__init__(directory,cache_size=4);self.with_baselines=with_baselines
    def __getitem__(self,index):
        index=int(index);si=bisect_right(self.ends,index)
        if si not in self.cache:
            self.cache[si]=torch.load(self.directory/self.shards[si]["file"],weights_only=True,map_location="cpu",mmap=True)
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        s=super().__getitem__(index)
        if self.with_baselines:
            e=self.shards[si];off=index-(self.ends[si-1] if si else 0);ri,h=divmod(off,e["heads"])
            x=self.cache[si];sel=x["records"][ri]["_batch_selection"]
            s.update(minimum=x["digests"]["minimum"][h,sel],maximum=x["digests"]["maximum"][h,sel])
        return s

class SelectedDataset:
    def __init__(self,base,indices):self.base=base;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.base[int(self.indices[i])]

def make_plan(train_dir,val_dir,out,batches=60000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);train=MMapDataset(train_dir);val=MMapDataset(val_dir)
    bins=[4096,8192,16384,32768];pools={b:[] for b in bins};lookup={}
    starts=[0]+train.ends[:-1]
    for i,d in enumerate(train.manifest["documents"]):pools[int(d["source"].split("_")[-1])].append(i)
    for e,start in zip(train.shards,starts):lookup[e["document_index"],e["layer_id"]]=(start,e["records"])
    orders={};cursors={b:0 for b in bins}
    def doc(b):
        if b not in orders or cursors[b]==len(orders[b]):
            orders[b]=rng.permutation(pools[b]);cursors[b]=0
        d=int(orders[b][cursors[b]]);cursors[b]+=1;return d
    plan=np.empty((batches,64),dtype=np.int64)
    for chunk in range((batches+7)//8):
        for bi,b in enumerate(bins):
            d=doc(b);layer=(chunk+7*bi)%28;start,n=lookup[d,layer]
            quartiles=np.array_split(np.arange(n),4)
            picked=np.concatenate([rng.choice(q,8,replace=False) for q in quartiles]);rng.shuffle(picked)
            ids=(start+picked[:,None]*4+np.arange(4)).reshape(8,16)
            rank=(chunk+bi)%4
            for j in range(8):
                step=chunk*8+j
                if step<batches:plan[step,rank*16:(rank+1)*16]=ids[j]
    vrng=np.random.default_rng(20261005);val_records={}
    vstarts=[0]+val.ends[:-1];selected=[]
    for e,start in zip(val.shards,vstarts):
        d=e["document_index"];n=e["records"]
        if d not in val_records:
            val_records[d]=sorted(np.concatenate([vrng.choice(q,4,replace=False) for q in np.array_split(np.arange(n),4)]).tolist())
        for ri in val_records[d]:selected.extend(start+ri*4+h for h in range(4))
    selected=np.asarray(selected,dtype=np.int64)
    np.save(out/"train_plan.npy",plan);np.save(out/"validation_indices.npy",selected)
    weights={}
    for b in bins:
        scenes=sum(e["scenes"] for e in train.shards if e["document_index"] in set(pools[b]))
        weights[str(b)]={"documents":len(pools[b]),"full_scenes":scenes,"full_scene_share":scenes/len(train),"sample_share":.25,
                         "relative_to_full_scene_weight":.25/(scenes/len(train))}
    checks={}
    ends=np.asarray(train.ends)
    for step in [64,5000,20000,50000]:
        ids=plan[:step].reshape(-1);sis=np.searchsorted(ends,ids,side="right")
        layers=np.asarray([e["layer_id"] for e in train.shards])[sis]
        docs=np.asarray([e["document_index"] for e in train.shards])[sis]
        heads=(ids-np.asarray(starts)[sis])%4
        checks[str(step)]={"presentations":len(ids),"unique_scenes":len(np.unique(ids)),"unique_documents":len(np.unique(docs)),
                          "layer_counts":np.bincount(layers,minlength=28).tolist(),"head_counts":np.bincount(heads,minlength=4).tolist()}
    report={"seed":42,"query_rng":"numpy default_rng PCG64; frozen full index plan","batches":batches,"global_batch":64,
            "plan_sha256":hashlib.sha256((out/"train_plan.npy").read_bytes()).hexdigest(),
            "validation_sha256":hashlib.sha256((out/"validation_indices.npy").read_bytes()).hexdigest(),
            "weights":weights,"prefix_checks":checks,"validation_scenes":len(selected),
            "sampling":"Each 8-batch chunk: one doc from each length bucket, shuffled without replacement within bucket cycles; layer=(chunk+7*bucket)%28. Each doc-layer picks 8 distinct queries per position quartile, all 4 KV heads. Rank assignment rotates. Queries can repeat across visits; no importance weighting. Uniform bucket then uniform doc then query-quartile distribution, not full-scene distribution.",
            "validation":"Fixed seed20261005: 4 queries per quartile, same 16 positions across all layers per doc; every independent validation doc, all 28 layers and 4 heads. Selection mean coverage across budgets4,8,16,32,64,128, macro over these fixed scenes. Full independent validation on chosen checkpoint after50k.",
            "manifest_hashes":{k:hashlib.sha256((Path(v)/"manifest.json").read_bytes()).hexdigest() for k,v in [("train",train_dir),("validation",val_dir)]}}
    (out/"plan_audit.json").write_text(json.dumps(report,indent=2))
    return report

def merge_evaluations(parts):
    result={k:v for k,v in parts[0].items() if k not in ["groups","scenes","evaluation_seconds"]}
    groups={}
    for part in parts:
        for g,v in part["groups"].items():
            dst=groups.setdefault(g,{"scenes":0,"methods":{}})
            dst["scenes"]+=v["scenes"]
            for name,metrics in v["methods"].items():
                acc=dst["methods"].setdefault(name,{})
                for k,x in metrics.items():acc[k]=acc.get(k,0.)+x*v["scenes"]
    for v in groups.values():
        for metrics in v["methods"].values():
            for k in metrics:metrics[k]/=v["scenes"]
    result.update(groups=groups,scenes=sum(p["scenes"] for p in parts),
                  evaluation_seconds=max(p["evaluation_seconds"] for p in parts))
    return result
