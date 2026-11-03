"""Only ranker-consumed summaries; no extrema, radius or diagonal work."""
import torch


def compact_digest(keys, method='gram', chunk=1024):
    h, length, d = keys.shape
    assert length % 16 == 0
    n = length//16
    blocks = keys.float().reshape(h, n, 16, d)
    mean = blocks.mean(2)
    flat = (blocks-mean[:, :, None]).reshape(-1, 16, d)
    bases, values = [], []
    for start in range(0, len(flat), chunk):
        x = flat[start:start+chunk]
        if method == 'svd':
            _, s, vh = torch.linalg.svd(x, full_matrices=False)
            bases.append(vh[:, :4]); values.append(s[:, :4].square()/16)
        else:
            eigen, u = torch.linalg.eigh(x @ x.transpose(-1, -2))
            eigen = eigen[:, -4:].flip(-1).clamp_min(0)
            u = u[:, :, -4:].flip(-1)
            basis = (u.transpose(-1, -2) @ x)/eigen.sqrt().clamp_min(1e-15)[:, :, None]
            bases.append(basis); values.append(eigen/16)
    return dict(mean=mean, basis=torch.cat(bases).reshape(h, n, 4, d),
                eigenvalues=torch.cat(values).reshape(h, n, 4), counts=mean.new_full((n,), 16))


def build_prefix_digests(pairs, **kwargs):
    complete = pairs[0][0].shape[2]//16*16
    return [compact_digest(k[0, :, :complete], **kwargs) for k, _ in pairs]
