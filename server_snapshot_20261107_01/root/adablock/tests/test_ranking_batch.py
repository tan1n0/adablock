"""Equivalence checks for optional ragged batch execution."""
import copy
import unittest
import torch
from models.block_ranker import BlockRanker, BlockRankerConfig, to_device
from utils.ranking_batch import pack_scenes, forward_packed, sample_batch_pairs, loss_from_pairs, sample_pairs_cpu
from utils.ranking_loss import PairSamplingConfig, sample_pairs, weighted_rank_loss

def samples():
    gen = torch.Generator().manual_seed(139)
    scenes = []
    for i, n in enumerate([1, 7, 19, 0, 3, 11, 5, 23]):
        d, rank, groups = 16, 3, 2
        fixed = .99 if i == 4 else .12
        mass = torch.rand(n, generator=gen)
        mass = mass / mass.sum().clamp_min(1e-8) * (1-fixed)
        if i == 6:
            mass.fill_((1-fixed)/n)
        scenes.append(dict(query=torch.randn(groups,d,generator=gen),
            mean=torch.randn(n,d,generator=gen),basis=torch.randn(n,rank,d,generator=gen),
            eigenvalues=torch.rand(n,rank,generator=gen),counts=torch.full((n,),16.),
            block_ids=torch.arange(n),teacher_mass=mass,fixed_mass=torch.tensor(fixed),
            token_position=(n+35)*16,block_size=16,layer_id=i%3,kv_head_id=i%2))
    return scenes

class BatchTests(unittest.TestCase):
    def check_device(self, device):
        torch.set_num_threads(2); torch.manual_seed(9)
        base=BlockRanker(BlockRankerConfig(16,3,2,projection_size=8,hidden_size=16,block_chunk_size=4)).to(device)
        new=copy.deepcopy(base); scenes=samples(); p=pack_scenes(scenes,device)
        ref=[base(to_device(s,device)) for s in scenes]
        got=forward_packed(new,p)
        torch.testing.assert_close(got,torch.cat(ref),atol=2e-6,rtol=2e-5)
        cfg=PairSamplingConfig()
        g1=torch.Generator().manual_seed(42); g2=torch.Generator().manual_seed(42)
        # Identical CPU score inputs isolate exact sampler and RNG equivalence.
        cpu=torch.cat(ref).detach().cpu()
        a=[sample_pairs(ref[i],s["teacher_mass"].to(device),s["fixed_mass"].to(device),cfg,g1).cpu() for i,s in enumerate(scenes)]
        b=sample_batch_pairs(cpu,scenes,p["offsets"],cfg,g2)
        self.assertTrue(torch.equal(g1.get_state(),g2.get_state()))
        for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
        old_losses=[weighted_rank_loss(ref[i],s["teacher_mass"].to(device),a[i].to(device)) for i,s in enumerate(scenes) if a[i].numel()]
        old_loss=torch.stack(old_losses).mean()
        new_loss,_=loss_from_pairs(got,scenes,p["offsets"],a)
        torch.testing.assert_close(old_loss,new_loss,atol=2e-6,rtol=2e-5)
        old_loss.backward(); new_loss.backward()
        for x,y in zip(base.parameters(),new.parameters()):
            torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=3e-4)
        op1=torch.optim.AdamW(base.parameters(),lr=2e-4,weight_decay=.01)
        op2=torch.optim.AdamW(new.parameters(),lr=2e-4,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(base.parameters(),1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(new.parameters(),1.,error_if_nonfinite=True)
        op1.step(); op2.step()
        for (name,x),(_,y) in zip(base.named_parameters(),new.named_parameters()):
            if name == "output.bias":
                # Pairwise ranking is invariant to this additive constant.
                # Roundoff-level gradients can be magnified by Adam epsilon.
                self.assertLess(float(x.grad.abs().max()),1e-6)
                self.assertLess(float(y.grad.abs().max()),1e-6)
            else:
                torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-4)
        after_a=torch.cat([base(to_device(s,device)) for s in scenes])
        after_b=forward_packed(new,p)
        torch.testing.assert_close(after_a-after_a.mean(),after_b-after_b.mean(),atol=4e-6,rtol=3e-4)
        # All-empty/covered batch has no update; tied scores preserve CPU tie breaks.
        empty=[dict(s, fixed_mass=torch.tensor(.99)) for s in scenes]
        ep=pack_scenes(empty,device)
        pairs=sample_batch_pairs(torch.ones(len(cpu)),empty,ep["offsets"],cfg,torch.Generator().manual_seed(3))
        loss,_=loss_from_pairs(forward_packed(new,ep),empty,ep["offsets"],pairs)
        self.assertIsNone(loss)
        t1=torch.Generator().manual_seed(4); t2=torch.Generator().manual_seed(4)
        tied=torch.ones(len(cpu))
        x=[sample_pairs(tied[p["offsets"][i]:p["offsets"][i+1]],s["teacher_mass"],s["fixed_mass"],cfg,t1) for i,s in enumerate(scenes)]
        y=sample_batch_pairs(tied,scenes,p["offsets"],cfg,t2)
        self.assertTrue(torch.equal(t1.get_state(),t2.get_state()))
        for aa,bb in zip(x,y): self.assertTrue(torch.equal(aa,bb))
    def test_sampler_random_and_boundary_cases(self):
        for seed in range(100):
            gen=torch.Generator().manual_seed(seed)
            n=[0,1,2,16,127,1024][seed%6]
            mass=torch.rand(n,generator=gen)
            if seed%5==0: mass.fill_(.01)
            mass=mass/mass.sum().clamp_min(1e-8)*.8
            scores=torch.randn(n,generator=gen)
            if seed%3==0: scores.round_()
            cfg=PairSamplingConfig(hard_negatives=seed%5,random_negatives=seed%6,positive_pairs=seed%33)
            fixed=.99 if seed%7==0 else .2
            a=torch.Generator().manual_seed(seed+99);b=torch.Generator().manual_seed(seed+99)
            old=sample_pairs(scores,mass,fixed,cfg,a)
            new=sample_pairs_cpu(scores,mass,fixed,cfg,b)
            self.assertTrue(torch.equal(old,new),f"pairs seed {seed}")
            self.assertTrue(torch.equal(a.get_state(),b.get_state()),f"RNG seed {seed}")
    def test_cpu(self): self.check_device("cpu")
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda(self): self.check_device("cuda:0")

if __name__ == "__main__": unittest.main()
