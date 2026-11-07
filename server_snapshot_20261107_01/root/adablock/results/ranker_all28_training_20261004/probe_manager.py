from pathlib import Path
import subprocess,json,time,os,traceback,sys,hashlib,itertools
R=Path('/root/adablock');W=R/'results/ranker_all28_training_20261004';C=R/'checkpoints/ranker_all28_ddp_probe100032_20261004'
cmd=['/root/miniconda3/envs/ada/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=4','scripts/train_block_ranker_ddp.py','--train-dir','data/ranker_cobs_train_all28_20261004','--val-dir','data/ranker_cobs_validation_all28_20261004','--output-dir',str(C),'--epochs','10','--local-batch','16','--lr','0.0002','--weight-decay','0.01','--coverage','0.95','--seed','42','--cpu-threads','1','--checkpoint-every-scenes','100000','--log-every-batches','250','--max-run-scenes','100032','--trace-indices']
(W/'probe_command.json').write_text(json.dumps(cmd,indent=2))
start=time.perf_counter()
with (W/'probe.log').open('w') as log:
 p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 (W/'probe_status.json').write_text(json.dumps({'stage':'running','torchrun_pid':p.pid,'bounded_scenes':100032}))
 code=p.wait()
if code:raise RuntimeError('Probe exit '+str(code))
sys.path.insert(0,str(R))
import torch
from models.block_ranker import BlockRanker,BlockRankerConfig
from utils.ranking_data import RankingDataset,ShardShuffleSampler
cp=torch.load(C/'resume_latest.pt',weights_only=True,map_location='cpu')
assert cp['epoch']==1 and cp['next_scene_offset']==100032
assert cp['data_metadata']['layers']==list(range(28))
assert len(cp['rank_rng_states'])==4 and all(set(x)=={'torch_cpu','torch_cuda','pair','python'} for x in cp['rank_rng_states'])
torch.manual_seed(42);m=BlockRanker(BlockRankerConfig(**cp['config']))
params=list(m.named_parameters());ids=cp['optimizer_state_dict']['param_groups'][0]['params']
pid=next(i for i,(name,v) in zip(ids,params) if name=='layer_embedding.weight')
moment=cp['optimizer_state_dict']['state'][pid]['exp_avg']
active=[i for i in range(28) if moment[i].abs().sum().item()>0]
delta=(cp['model_state_dict']['layer_embedding.weight']-m.layer_embedding.weight).abs().sum(-1)
assert active==list(range(28)) and (delta>0).all()
trace=cp['trace_indices'];assert len(trace)==100032 and len({i for e,i in trace})==100032 and {e for e,i in trace}=={1}
ds=RankingDataset(R/'data/ranker_cobs_train_all28_20261004')
expected=list(itertools.islice(iter(ShardShuffleSampler(ds,torch.Generator().manual_seed(42))),100032))
assert [i for e,i in trace]==expected
assert all(torch.isfinite(v).all() for v in cp['model_state_dict'].values())
secs=cp['epoch_stats']['active_train_seconds'];rate=100032/secs
result={'stage':'completed','bounded_scenes':100032,'optimizer_steps':cp['epoch_stats']['optimizer_steps'],'valid_scenes':cp['epoch_stats']['valid_scenes'],'active_train_seconds':secs,'scenes_per_second':rate,'train_only_hours_per_epoch':len(ds)/rate/3600,'train_only_hours_10epochs':10*len(ds)/rate/3600,'wall_seconds':time.perf_counter()-start,'all28_embedding_optimizer_moments_nonzero':True,'all28_embedding_rows_changed':True,'embedding_row_abs_deltas':delta.tolist(),'exact_original_seed42_sampler_prefix':True,'duplicate_scenes':0,'full_optimizer_and_rank_rng_checkpoint_verified':True,'initial_model_sha256':cp['provenance']['initial_model_sha256'],'checkpoint':str(C/'resume_latest.pt'),'no_full_run_started':True}
(W/'probe_results.json').write_text(json.dumps(result,indent=2));(W/'probe_status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
with (R/'docs/ranker_full_training_20261003.md').open('a') as f:
 f.write('\n\n### 全28层DDP训练启动前吞吐/正确性检查\n\n新随机初始化，沿用local16/global64、LR2e-4、weight_decay0.01、seed42、原rankloss/结构、epoch目标10；本次仅执行100032场景有界检查，不代表已启动全程。全28层embedding非零优化器动量且参数更新；采样器前缀完全一致且无重复，完整优化器/四rank随机状态/数据offset断点可读。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
