import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3";os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["TOKENIZERS_PARALLELISM"]="false"
from pathlib import Path
import sys,json,time,hashlib,pickle,traceback
W=Path(__file__).parent;sys.path.insert(0,str(W/"source"))
def save(n,x):(W/"calibration"/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
try:
 import torch,dill
 import identify_important_fre as cal
 from longbench import longbench_pred as official
 assert torch.cuda.device_count()==4
 official.seed_everything(42);torch.set_num_threads(4)
 cal.sparsity_list=[256];cal.total_tokens=0
 snap="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
 placement=json.loads((W/"placement_plan.json").read_text())
 tok=cal.AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False);tok.pad_token_id=tok.eos_token_id
 model=cal.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.float16,attn_implementation="flash_attention_2",local_files_only=True,trust_remote_code=False,device_map=placement["device_map"],max_memory={int(k):v for k,v in placement["max_memory"].items()}).eval()
 assert all(getattr(l.self_attn,"_old_forward",l.self_attn.forward).__module__=="identify_important_fre" for l in model.model.layers)
 rows=json.loads((W/"calibration/selected_rows.json").read_text())
 text=official.build_chat(tok,official.dataset2prompt["qasper"].format(**rows[0]),"Qwen2.5-7B-Instruct")
 ids=tok(text,return_tensors="pt").input_ids
 save("runtime.json",dict(dtype=str(next(model.parameters()).dtype),input_tokens=ids.shape[-1],input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),device_map=model.hf_device_map,default_samples=8,actual_samples=1,computed_budgets=[256],max_new_tokens=128))
 save("status.json",dict(stage="calibrating",pid=os.getpid()))
 start=time.perf_counter()
 result=official.get_pred_single_gpu(rows,official.model2maxlen["Qwen2.5-7B-Instruct"],official.dataset2maxlen["qasper"],official.dataset2prompt["qasper"],"qasper","Qwen2.5-7B-Instruct",str(W/"calibration/official_predictions.jsonl"),tok,model)
 for d in range(4):torch.cuda.synchronize(d)
 assert cal.total_tokens>0
 with (W/"calibration/raw_overlap_records.pkl").open("wb") as f:dill.dump(cal.res_dict,f)
 # Official consumer requires six budget slots. Only index1 (256) is authorized/computed; other slots deliberately NaN.
 agreement=torch.full((6,28,64,28),float("nan"),dtype=torch.float32)
 counts={}
 for group in range(64):
  for layer in range(28):
   values=torch.tensor(cal.res_dict[256][group*2][layer],dtype=torch.float32)
   assert values.shape==(cal.total_tokens,28) and torch.isfinite(values).all()
   assert (values>=0).all() and (values<=1).all()
   agreement[1,layer,group,:]=values.mean(0)
   counts[str(layer)]=int(values.shape[0])
 target=W/"dimension_rankings/Qwen2.5-7B-Instruct";target.mkdir(parents=True,exist_ok=True)
 with (target/"qasper_agreement.pkl").open("wb") as f:pickle.dump(agreement,f)
 save("summary.json",dict(stage="completed",samples=1,decode_queries=cal.total_tokens,counts_per_layer=counts,calibration_seconds=time.perf_counter()-start,agreement_shape=list(agreement.shape),computed_budget_index=1,computed_budget=256,uncomputed_slots_nan=True,agreement_sha256=hashlib.sha256((target/"qasper_agreement.pkl").read_bytes()).hexdigest(),peak_reserved_mib={str(d):torch.cuda.max_memory_reserved(d)/2**20 for d in range(4)}))
 save("status.json",dict(stage="completed"));print("CALIBRATION_COMPLETE",flush=True)
except Exception:
 save("status.json",dict(stage="failed",traceback=traceback.format_exc()));traceback.print_exc();sys.exit(1)
