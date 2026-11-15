"""Fast summaries with cached reference-SVD refinement at decision boundaries.

The tolerances are numerical guards, not a formal error bound. They use no
answers or question identities. Each decode owns its mutable summary tensors.
"""
import torch
import gpu_controller as gpu

class Controller(gpu.Controller):
    score_guard=1e-4
    cdf_guard=1e-4

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.exact={}
        self.audit.update(refined_blocks=0,refinement_calls=0,budget_guard_heads=0,guard_budget_calls=0,guard_ranker_calls=0)

    def choose(self,layer,query,scale):
        original=gpu.base.forward_packed
        def guarded(model,packed):
            scores=original(model,packed).view(8,-1).float()
            h,n=scores.shape
            ids=packed['block_ids'][:n]
            digest=self.digests[layer]
            if layer not in self.exact:
                # Capacity covers appended blocks; newly complete blocks stay local.
                self.exact[layer]=torch.zeros((h,(self.position+255)//16+1),dtype=torch.bool,device=query.device)
            exact=self.exact[layer]
            hist=(self.position+15)//16
            sink,local,_=gpu.base.fixed_and_middle_blocks(hist,2,32,hist-1)
            fixed_tokens=sum(min(16,max(0,self.position-b*16)) for b in sink|local)+1
            target=max(0,min(h*n,int(self.ratio*h*(self.position+1)-h*fixed_tokens)//16))
            for iteration in range(4):
                features,order=gpu.features_gpu(packed['query'],scores,self.position,layer)
                probability=self.budget(features).softmax(-1)
                self.audit['guard_budget_calls']+=1
                classes=(probability.cumsum(-1)<self.quantile).sum(-1).clamp_max(8)
                choices=torch.tensor([min(n,k) for k in gpu.base.BUDGETS[:-1]]+[n],device=query.device)
                takes=gpu.allocate_gpu(choices[classes],target,n)
                sorted_scores=scores.gather(1,order)
                left=sorted_scores.gather(1,(takes-1).clamp(0,n-1)[:,None])
                right=sorted_scores.gather(1,takes.clamp(0,n-1)[:,None])
                center=(left+right)*.5
                near_cut=((scores-center).abs()<=self.score_guard)&((takes>0)&(takes<n))[:,None]
                near_cdf=(probability.cumsum(-1)-self.quantile).abs().amin(-1)<=self.cdf_guard
                uncertain=near_cut|near_cdf[:,None]
                if iteration==3:
                    # Conservative escalation if repeated refinements move the boundary.
                    uncertain=torch.ones_like(uncertain)
                todo=uncertain&~exact[:,ids]
                if not bool(todo.any()):break
                hh,ii=todo.nonzero(as_tuple=True);blocks=ids[ii]
                token_ids=blocks[:,None]*16+torch.arange(16,device=query.device)
                keys=self.pairs[layer][0][0]
                x=keys[hh[:,None],token_ids].float()-digest['mean'][hh,blocks,None,:]
                bases=[];values=[]
                for start in range(0,len(hh),128):
                    _,s,vh=torch.linalg.svd(x[start:start+128],full_matrices=False)
                    bases.append(vh[:,:4]);values.append(s[:,:4].square()/16)
                basis=torch.cat(bases);eigen=torch.cat(values)
                digest['basis'][hh,blocks]=basis;digest['eigenvalues'][hh,blocks]=eigen
                exact[hh,blocks]=True
                packed['basis'][hh*n+ii]=basis;packed['eigenvalues'][hh*n+ii]=eigen
                self.audit['refined_blocks']+=len(hh)
                self.audit['refinement_calls']+=1
                self.audit['budget_guard_heads']+=int(near_cdf.sum())
                scores=original(model,packed).view(h,n).float()
                self.audit['guard_ranker_calls']+=1
            return scores.flatten()
        gpu.base.forward_packed=guarded
        try:return super().choose(layer,query,scale)
        finally:gpu.base.forward_packed=original
