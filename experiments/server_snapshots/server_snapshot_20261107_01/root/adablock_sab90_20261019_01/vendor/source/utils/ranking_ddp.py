"""DDP support for ragged ranking; mean over GLOBAL effective scenes."""
import torch
from torch import nn
import torch.distributed as dist
from utils.ranking_batch import pack_scenes,forward_packed,loss_from_pairs

class DistributedBatchRanker(nn.Module):
    def __init__(self,ranker):
        super().__init__();self.ranker=ranker
    def forward(self,scenes):
        if not scenes:
            # All parameters participate so an empty tail rank joins collectives.
            return sum(p.sum()*0. for p in self.ranker.parameters()).reshape(1),[0]
        packed=pack_scenes(scenes,next(self.ranker.parameters()).device)
        return forward_packed(self.ranker,packed),packed["offsets"]

def distributed_scene_loss(scores,scenes,offsets,pairs):
    _,per=loss_from_pairs(scores,scenes,offsets,pairs)
    local_sum=per.sum() if len(per) else scores.sum()*0.
    totals=torch.stack((local_sum.detach(),scores.new_tensor(float(len(per)))))
    dist.all_reduce(totals,op=dist.ReduceOp.SUM)
    count=int(totals[1].item())
    # DDP averages gradients across ranks, so compensate by world_size.
    scaled=local_sum*(dist.get_world_size()/count) if count else scores.sum()*0.
    mean=float(totals[0]/count) if count else 0.
    return scaled,mean,count,len(per)

def global_batch_for_rank(indices,start,local_batch,rank):
    # No padding/repeating: a last global batch may have empty ranks.
    return indices[start+rank*local_batch:start+(rank+1)*local_batch]
