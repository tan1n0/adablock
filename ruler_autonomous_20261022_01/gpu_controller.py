"""Device-resident decisions; one scalar width sync for existing padded SDPA."""
import math
import torch
from torch.nn import functional as F
import baseline_controller as base


def features_gpu(query, scores, position, layer):
    h, n = scores.shape
    mean = scores.mean(-1, keepdim=True)
    centered = scores - mean
    ordered, order = scores.sort(dim=-1, descending=True, stable=True)
    boundaries = torch.tensor([min(k, n)-1 for k in base.BUDGETS[1:-1]], device=scores.device)
    quantiles = torch.quantile(centered, torch.linspace(0, 1, 17, device=scores.device), dim=-1).T
    gaps = ordered[:, :1] - ordered[:, boundaries]
    # Preserve baseline softmax-before-sort computation.
    shares = scores.softmax(-1).gather(1, order).cumsum(-1)[:, boundaries]
    scalars = torch.stack([mean[:, 0]*0+math.log1p(n), mean[:, 0]*0+math.log1p(position),
                           mean[:, 0], scores.std(-1, unbiased=False), ordered[:, 0]-ordered[:, -1]], -1)
    identities = torch.cat([F.one_hot(torch.full((h,), layer, device=scores.device), 32),
                            F.one_hot(torch.arange(h, device=scores.device), 8)], -1).float()
    return torch.cat([query.flatten(1), quantiles, gaps, shares, scalars, identities], -1), order


def allocate_gpu(raw, target, cap):
    # Double precision follows the Python allocator, including stable head tie breaks.
    weights = raw.double().clamp_min(1)
    active = torch.ones_like(raw, dtype=torch.bool)
    continuous = torch.zeros_like(weights)
    remaining = weights.new_tensor(float(target))
    for _ in range(len(raw)):
        proposed = remaining*weights/(weights*active).sum().clamp_min(1)
        saturated = active & (proposed >= cap)
        continuous = torch.where(active, proposed.clamp_max(cap), continuous)
        remaining = remaining-saturated.sum()*cap
        active = active & ~saturated
    counts = continuous.floor().long()
    remainder = torch.where(counts < cap, continuous-counts, -torch.inf)
    order = remainder.argsort(descending=True, stable=True)
    bonus = torch.zeros_like(counts).scatter(0, order, (torch.arange(len(raw), device=raw.device) < target-counts.sum()).long())
    return counts+bonus


def selected_gpu(ids, order, takes, fixed, position):
    h, n = order.shape
    # Existing SDPA requires a host-known rectangular shape. Only this scalar
    # crosses to CPU; scores, classes, per-head quotas and token indices do not.
    max_take = int(takes.max())
    ranked = ids[order[:, :max_take]]
    ranked = torch.where(torch.arange(max_take, device=ids.device)[None] < takes[:, None], ranked, position+1)
    fixed_ids = torch.tensor(sorted(fixed), device=ids.device).expand(h, -1)
    blocks = torch.cat([fixed_ids, ranked], -1).sort(-1).values
    tokens = (blocks[..., None]*16+torch.arange(16, device=ids.device)).flatten(1)
    tokens = torch.where(tokens < position, tokens, position+1)
    tokens = torch.cat([tokens, torch.full((h, 1), position, device=ids.device)], -1).sort(-1).values
    fixed_tokens = sum(min(16, max(0, position-b*16)) for b in fixed)+1
    tokens = tokens[:, :fixed_tokens+16*max_take]
    valid = tokens <= position
    return torch.where(valid, tokens, 0), valid, takes*16+fixed_tokens


class Controller(base.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = False  # Return cost telemetry without retaining token indices.

    def choose(self, layer, query, scale):
        position = self.position
        history = (position+15)//16
        sink, local, middle = base.fixed_and_middle_blocks(history, 2, 32, history-1)
        fixed = sink | local
        ids = torch.tensor(sorted(middle), device=query.device, dtype=torch.long)
        h = self.model.config.num_key_value_heads
        group = self.model.config.num_attention_heads//h
        grouped = query[0, :, 0].float().view(h, group, -1)*scale
        n = len(ids)
        classes = torch.full((h,), -1, device=query.device, dtype=torch.long)
        if self.mode == 'all' or not n:
            takes = torch.full((h,), n, device=query.device, dtype=torch.long)
            order = torch.arange(n, device=query.device).expand(h, -1)
        else:
            digest = self.digests[layer]
            packed = {key: digest[key][:, ids].flatten(0, 1) for key in ['mean', 'basis', 'eigenvalues']}
            packed.update(query=grouped, counts=digest['counts'][ids].repeat(h), block_ids=ids.repeat(h),
                          scene_ids=torch.arange(h, device=query.device).repeat_interleave(n),
                          token_position=torch.full((h,), position, device=query.device),
                          layer_id=torch.full((h,), layer, device=query.device),
                          kv_head_id=torch.arange(h, device=query.device),
                          block_size=torch.full((h,), 16, device=query.device))
            scores = base.forward_packed(self.ranker, packed).view(h, n).float()
            self.audit['ranker_calls'][layer] += 1
            if self.mode == 'adaptive':
                features, order = features_gpu(grouped, scores, position, layer)
                probability = self.budget(features).softmax(-1)
                classes = (probability.cumsum(-1) < self.quantile).sum(-1).clamp_max(len(base.BUDGETS)-1)
                if self.quantile == 1:
                    classes.fill_(len(base.BUDGETS)-1)
                choices = torch.tensor([min(n, k) for k in base.BUDGETS[:-1]]+[n], device=query.device)
                takes = choices[classes]
                self.audit['budget_calls'][layer] += 1
            else:
                takes = torch.full((h,), min(n, int(self.mode.removeprefix('fixed'))), device=query.device)
                order = scores.argsort(descending=True, stable=True)
        fixed_tokens = sum(min(16, max(0, position-b*16)) for b in fixed)+1
        # Autonomous policy: use the predicted absolute block count directly.
        indices, valid, lengths = selected_gpu(ids, order, takes, fixed, position)
        self.step_rows.append(dict(layer=layer, selected_tokens=lengths, middle_blocks=takes, classes=classes,
                                   width=indices.shape[1], candidate_blocks=n))
        if self.capture:
            self.selected.append(dict(position=position, layer=layer, indices=indices, valid=valid, classes=classes))
        return indices, valid

    def finish_step(self):
        # Scalar quota is known on host; detailed telemetry is deferred until after timing.
        rows = self.step_rows
        totals = torch.stack([r['selected_tokens'] for r in rows])
        middle = torch.stack([r['middle_blocks'] for r in rows])
        self.costs.append(dict(read=totals.sum(), padded=sum(len(r['selected_tokens'])*r['width'] for r in rows),
                               full=totals.numel()*(self.position+1)))
        self.steps.append(dict(position=self.position, mean_selected_tokens=totals.float().mean(),
                               mean_middle_blocks=middle.float().mean(), read_fraction=totals.float().mean()/(self.position+1)))
        self.position += 1
