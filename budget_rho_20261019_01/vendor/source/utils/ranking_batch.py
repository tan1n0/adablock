"""Ragged batching for BlockRanker; unchanged model, scene order and pair sampler."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from utils.ranking_loss import sample_pairs

def pack_scenes(scenes, device):
    if not scenes:
        raise ValueError("Cannot pack an empty batch")
    sizes = [len(s["block_ids"]) for s in scenes]
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)
    # Labels, fixed masses and Quest-only summaries stay on CPU.
    keys = ("mean", "basis", "eigenvalues", "counts", "block_ids")
    packed = {k: torch.cat([s[k] for s in scenes]).to(device) for k in keys}
    packed["query"] = torch.stack([s["query"] for s in scenes]).to(device)
    packed["scene_ids"] = torch.repeat_interleave(torch.arange(len(scenes)), torch.tensor(sizes)).to(device)
    for key in ("token_position", "layer_id", "kv_head_id", "block_size"):
        packed[key] = torch.tensor([s[key] for s in scenes], device=device)
    packed["offsets"] = offsets
    return packed

def forward_packed(model, p):
    q, means = p["query"].float(), p["mean"].float()
    if not len(means):
        return model.output.weight.sum().mul(0).expand(0)
    qp = model.query_proj(q)
    mp = model.mean_proj(means)
    identity = torch.cat((model.layer_embedding(p["layer_id"]),
                          model.head_embedding(p["kv_head_id"])), -1)
    outputs = []
    # Same per-scene model; larger compute tiles are only an execution change.
    chunk = model.config.block_chunk_size * len(q)
    for start in range(0, len(means), chunk):
        end = min(start + chunk, len(means))
        ids = p["scene_ids"][start:end]
        qc = q[ids].transpose(0, 1)  # [G, candidates, D]
        first = torch.einsum("gnd,nd->gn", qc, means[start:end])
        projected = torch.einsum("gnd,nrd->gnr", qc, p["basis"][start:end].float())
        second = (projected.square() * p["eigenvalues"][start:end].float()[None]).sum(-1)
        counts = p["counts"][start:end].float()
        qa = qp[ids].transpose(0, 1)
        ma = mp[start:end][None].expand(q.shape[1], -1, -1)
        scalars = torch.stack((first.sign() * first.abs().log1p(),
                               second.clamp_min(0).log1p(),
                               counts.clamp_min(1).log().expand_as(first)), -1)
        features = [qa, ma, qa * ma, scalars,
                    identity[ids][None].expand(q.shape[1], -1, -1)]
        if model.config.use_position:
            t = p["token_position"][ids].float()
            block_end = p["block_ids"][start:end].float() * p["block_size"][ids] + counts - 1
            distance = (t - block_end).clamp_min(1)
            length = t.clamp_min(1)
            pos = torch.stack((distance.log1p() / length.log1p(), distance / length), -1)
            features.append(pos[None].expand(q.shape[1], -1, -1))
        hidden = model.interaction(torch.cat(features, -1)).mean(0)
        outputs.append(model.output(hidden).squeeze(-1))
    return torch.cat(outputs)

def sample_batch_pairs(scores_cpu, scenes, offsets, config, generator):
    """Original sampler, same scene order, CPU RNG calls and stable tie breaks."""
    return [sample_pairs_cpu(scores_cpu[offsets[i]:offsets[i + 1]],
                         s["teacher_mass"], s["fixed_mass"], config, generator)
            for i, s in enumerate(scenes)]

def loss_from_pairs(scores, scenes, offsets, pairs):
    """Mean of independently mass-normalized effective scene losses."""
    valid = [i for i, pair in enumerate(pairs) if pair.numel()]
    if not valid:
        return None, scores.new_empty(0)
    global_pairs, weights, segment_ids = [], [], []
    denominators = []
    for segment, i in enumerate(valid):
        pair = pairs[i].cpu()
        mass = scenes[i]["teacher_mass"].detach().float().cpu()
        w = mass[pair[:, 0]] - mass[pair[:, 1]]
        if not bool((w > 0).all()):
            raise ValueError("Pairs must have strictly decreasing teacher mass")
        global_pairs.append(pair + offsets[i])
        weights.append(w)
        segment_ids.append(torch.full((len(pair),), segment, dtype=torch.long))
        denominators.append(w.sum().clamp_min(1e-8))
    pair = torch.cat(global_pairs).to(scores.device)
    w = torch.cat(weights).to(scores.device)
    segments = torch.cat(segment_ids).to(scores.device)
    numerator = scores.new_zeros(len(valid), dtype=torch.float32)
    values = w * F.softplus(scores[pair[:, 1]].float() - scores[pair[:, 0]].float())
    numerator.scatter_add_(0, segments, values)
    per_scene = numerator / torch.stack(denominators).to(scores.device)
    return per_scene.mean(), per_scene

def batch_loss(model, scenes, device, config, generator):
    packed = pack_scenes(scenes, device)
    scores = forward_packed(model, packed)
    pairs = sample_batch_pairs(scores.detach().float().cpu(), scenes, packed["offsets"], config, generator)
    loss, per_scene = loss_from_pairs(scores, scenes, packed["offsets"], pairs)
    return loss, per_scene, pairs

def sample_pairs_cpu(scores, mass, fixed_mass, config, generator=None):
    """Equivalent CPU sampling; vectorized filtering, unchanged torch RNG calls."""
    import numpy as np
    from utils.ranking_loss import coverage_positive_ids
    config.validate()
    mass_cpu = mass.detach().float().cpu()
    positive = coverage_positive_ids(mass_cpu, fixed_mass, config.coverage)
    if not positive.numel():
        return torch.empty((0, 2), dtype=torch.long)
    values = mass_cpu.numpy()
    score_values = scores.detach().float().cpu().numpy()
    mask = np.zeros(len(values), dtype=bool)
    mask[positive.numpy()] = True
    negative = np.flatnonzero(~mask)
    # Filtering a stable total order equals stable sorting each valid subset.
    ordered = negative[np.argsort(-score_values[negative], kind="stable")]
    picked = positive[torch.randperm(positive.numel(), generator=generator)[:config.max_positives]]
    delta = np.float32(config.delta)
    pairs = set()
    for i in picked.tolist():
        valid = negative[values[i] - values[negative] > delta]
        hard = ordered[values[i] - values[ordered] > delta][:config.hard_negatives]
        pairs.update((i, int(j)) for j in hard)
        remaining = valid[~np.isin(valid, hard)]
        random = remaining[torch.randperm(len(remaining), generator=generator)[:config.random_negatives].numpy()]
        pairs.update((i, int(j)) for j in random)
    internal = set()
    if positive.numel() > 1:
        positive_values = positive.numpy()
        for _ in range(max(32, config.positive_pairs * 10)):
            if len(internal) >= config.positive_pairs:
                break
            indices = torch.randint(positive.numel(), (2,), generator=generator)
            i, j = map(int, positive_values[indices.numpy()])
            if values[i] < values[j]:
                i, j = j, i
            if values[i] - values[j] > delta:
                internal.add((i, j))
    pairs.update(internal)
    return torch.tensor(sorted(pairs), dtype=torch.long).reshape(-1, 2)

class BatchRankingDataset:
    """Same indices/metadata; prefix views avoid repeated CPU advanced gathers."""
    def __init__(self, directory, cache_size=2):
        from utils.ranking_data import RankingDataset
        self.base = RankingDataset(directory, cache_size)
        self.__dict__.update({k: getattr(self.base, k) for k in
                             ("directory", "manifest", "metadata", "shards", "ends", "cache", "cache_size")})

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        from bisect import bisect_right
        if not 0 <= index < len(self):
            raise IndexError(index)
        si = bisect_right(self.ends, index)
        e = self.shards[si]
        off = index - (self.ends[si-1] if si else 0)
        if si not in self.cache:
            self.cache[si] = torch.load(self.directory/e["file"], weights_only=True, map_location="cpu")
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(si)
        shard = self.cache[si]; ri, head = divmod(off,e["heads"])
        r, d = shard["records"][ri], shard["digests"]
        ids, t = r["block_ids"], r["token_position"]
        bs = self.metadata["block_size"]
        if "_batch_selection" not in r:
            if not bool(((ids+1)*bs<=t).all()):
                raise ValueError("Noncausal candidate in ranking shard")
            if len(ids) and torch.equal(ids,torch.arange(int(ids[0]),int(ids[0])+len(ids))):
                r["_batch_selection"] = slice(int(ids[0]),int(ids[-1])+1)
            else:
                r["_batch_selection"] = ids
        select = r["_batch_selection"]
        # Quest extrema are unnecessary during ranker training.
        return {"query":r["query"][head], "mean":d["mean"][head,select],
                "basis":d["covariance_basis"][head,select],
                "eigenvalues":d["covariance_eigenvalues"][head,select],
                "counts":d["counts"][select], "block_ids":ids,"block_size":bs,
                "teacher_mass":r["teacher_mass"][head],"fixed_mass":r["fixed_mass"][head],
                "token_position":t,"layer_id":e["layer_id"],"kv_head_id":head,
                "document_index":e["document_index"]}
