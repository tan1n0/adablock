import unittest
import torch
import core as c
from utils.ranking_loss import weighted_rank_loss


class Tests(unittest.TestCase):
    def test_boundary_gradient(self):
        scores=torch.tensor([3.,2.,1.],requires_grad=True)
        mass=torch.tensor([.1,.2,.7])
        pairs=c.boundary_pairs(scores,mass,1)
        self.assertTrue(all(j==0 for i,j in pairs.tolist()))
        weighted_rank_loss(scores,mass,pairs).backward()
        self.assertGreater(float(scores.grad[0]),0)
        self.assertLess(float(scores.grad[2]),0)

    def test_boundary_empty(self):
        s=torch.tensor([3.,2.,1.]);m=torch.tensor([.1,.2,.7])
        self.assertEqual(c.boundary_pairs(s,m,0).numel(),0)
        self.assertEqual(c.boundary_pairs(s,m,3).numel(),0)

    def test_boundary_cap_and_order(self):
        s=torch.arange(64.).float();m=torch.arange(64.).flip(0)/100
        pairs=c.boundary_pairs(s,m,32)
        self.assertLessEqual(len(pairs),32)
        self.assertTrue(bool((m[pairs[:,0]]>m[pairs[:,1]]).all()))

    def test_teacher_free_features(self):
        s=dict(query=torch.randn(4,128),layer_id=0,kv_head_id=0,token_position=2000)
        scores=torch.randn(20);a=c.bp.online_features(s,scores)
        s.update(teacher_mass=torch.full((20,),float('nan')),fixed_mass=float('nan'))
        self.assertTrue(torch.equal(a,c.bp.online_features(s,scores)))

    def test_target_label(self):
        mass=torch.tensor([.1,.15,.25]);scores=torch.tensor([3.,2.,1.])
        label,cov,actual,_=c.bp.labels_for_order(mass,.5,scores,budgets=[0,1,2,3],rho=.9)
        self.assertEqual(label,3)
        self.assertGreaterEqual(float(cov[label]),.9-1e-7)


if __name__=='__main__':unittest.main()
