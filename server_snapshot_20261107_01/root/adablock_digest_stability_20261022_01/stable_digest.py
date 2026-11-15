import torch

def compact_digest(keys, method='double', chunk=1024):
    h,length,d=keys.shape
    assert length%16==0
    n=length//16
    blocks=keys.float().reshape(h,n,16,d)
    mean=blocks.mean(2)
    flat=(blocks-mean[:,:,None]).reshape(-1,16,d)
    bases=[];values=[]
    for start in range(0,len(flat),chunk):
        x=flat[start:start+chunk]
        if method=='double':
            xd=x.double()
            eigen,u=torch.linalg.eigh(xd@xd.transpose(-1,-2))
            eigen=eigen[:,-4:].flip(-1).clamp_min(0)
            u=u[:,:,-4:].flip(-1)
            basis=(u.transpose(-1,-2)@xd)/eigen.sqrt().clamp_min(1e-30)[:,:,None]
            bases.append(basis.float());values.append((eigen/16).float())
        elif method=='qr':
            q,r=torch.linalg.qr(x.transpose(-1,-2),mode='reduced')
            u,s,_=torch.linalg.svd(r,full_matrices=False)
            bases.append((q@u[:,:,:4]).transpose(-1,-2));values.append(s[:,:4].square()/16)
        else:raise ValueError(method)
    return dict(mean=mean,basis=torch.cat(bases).reshape(h,n,4,d),
                eigenvalues=torch.cat(values).reshape(h,n,4),counts=mean.new_full((n,),16))
