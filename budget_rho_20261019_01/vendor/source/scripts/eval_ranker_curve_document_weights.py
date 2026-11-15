"""Final full validation with both micro-scene and macro-document weighting."""
import os,sys,json,datetime,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.block_ranker import load_ranker
from utils.ranking_curve import MMapDataset,SelectedDataset,merge_evaluations
from utils.ranking_curve_document_eval import evaluate_batched
W=ROOT/"results/ranker_all28_curve_20261005";O=ROOT/"checkpoints/ranker_all28_curve_20261005"
rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"])
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";torch.use_deterministic_algorithms(True)
torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device("cuda",local)
dist.init_process_group("nccl",timeout=datetime.timedelta(hours=3))
assert (O/"COMPLETE").exists()
selection=json.loads((O/"selection.json").read_text());step=selection["best_step"]
model,payload=load_ranker(O/"best.pt",device);assert payload["training"]["step"]==step
data=MMapDataset(ROOT/"data/ranker_cobs_validation_all28_20261004",True)
indices=np.arange(len(data));local_indices=indices[np.searchsorted(np.asarray(data.ends),indices,side="right")%4==rank]
start=time.perf_counter()
part=evaluate_batched(model,SelectedDataset(data,local_indices),device,batch_size=16)
parts=[None]*4;dist.all_gather_object(parts,part)
if rank==0:
 result=merge_evaluations(parts);result["dataset_scenes"]=len(data)
 previous=json.loads((O/f"full_validation_best_{step:06d}.json").read_text())
 max_difference=max(abs(result["groups"]["overall"]["methods"][m][k]-v) for m,metrics in previous["groups"]["overall"]["methods"].items() for k,v in metrics.items())
 assert max_difference<1e-9,("full validation parity",max_difference)
 docgroups=[result["groups"][f"document/{i}"] for i in range(len(data.manifest["documents"]))]
 assert sum(g["scenes"] for g in docgroups)==len(data)
 def macro(groups):
  return {"documents":len(groups),"methods":{m:{k:sum(g["methods"][m][k] for g in groups)/len(groups) for k in groups[0]["methods"][m]} for m in groups[0]["methods"]}}
 out={"best_step":step,"selection_unchanged":selection["criterion"],"dataset_scenes":len(data),
      "scene_weighted":result["groups"]["overall"],"document_equal":macro(docgroups),
      "scene_weighted_fixed_below_rho":result["groups"]["fixed_below_rho"],
      "document_equal_fixed_below_rho":macro([result["groups"][f"fixed_below_rho/document/{i}"] for i in range(len(docgroups)) if f"fixed_below_rho/document/{i}" in result["groups"]]),
      "full_validation_parity_max_abs":max_difference,"evaluation_seconds":time.perf_counter()-start,
      "document_scene_counts":[g["scenes"] for g in docgroups],
      "semantics":"Scene weighted averages all414064 scenes equally. Document equal first averages all scenes within each of16 documents, then averages16 document means equally. Same checkpoint, candidate blocks, labels, methods and budgets. Fixed-below-rho macro averages conditional document means.",
      "groups":result["groups"]}
 (W/"final_validation_both_weights.json").write_text(json.dumps(out,indent=2))
 with (ROOT/"docs/ranker_full_training_20261003.md").open("a") as f:
  f.write("\n### Final validation: scene and document weighting\n\n"+json.dumps({k:v for k,v in out.items() if k!="groups"},indent=2)+"\n")
 print(json.dumps({k:v for k,v in out.items() if k!="groups"}),flush=True)
dist.barrier(device_ids=[local]);dist.destroy_process_group()
