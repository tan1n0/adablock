import os,sys,copy,json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig
from utils.ranking_ddp import DistributedBatchRanker,distributed_scene_loss
from utils.ranking_batch import pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
from utils.ranking_loss import PairSamplingConfig
from tests.test_ranking_batch import samples
rank=int(os.environ["RANK"]);torch.cuda.set_device(rank);torch.set_num_threads(1);dist.init_process_group("nccl")
torch.manual_seed(42);device=torch.device("cuda",rank)
base=BlockRanker(BlockRankerConfig(16,3,2,projection_size=8,hidden_size=16)).to(device);ref=copy.deepcopy(base)
ddp=DDP(DistributedBatchRanker(base),device_ids=[rank],broadcast_buffers=False)
allscenes=samples();local=allscenes[rank*2:(rank+1)*2]
scores,offsets=ddp(local);pairs=sample_batch_pairs(scores.detach().cpu(),local,offsets,PairSamplingConfig(),torch.Generator().manual_seed(43+rank))
loss,mean,count,n=distributed_scene_loss(scores,local,offsets,pairs);loss.backward()
allpairs=[None]*4;dist.all_gather_object(allpairs,pairs)
p=pack_scenes(allscenes,device);rs=forward_packed(ref,p);rl,_=loss_from_pairs(rs,allscenes,p["offsets"],[q for part in allpairs for q in part]);rl.backward()
torch.testing.assert_close(torch.tensor(mean),rl.detach().cpu(),atol=2e-6,rtol=2e-5)
for a,b in zip(base.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=3e-4)
if rank==0:print(json.dumps({"ddp_global_effective_scene_normalization":"passed","global_effective_scenes":count}),flush=True)
dist.destroy_process_group()
