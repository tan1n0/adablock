from cache_buffer import append_pair
import math
from runtime import *
import evaluation_base as base

def allocate(raw, target, cap):
    h=len(raw)
    assert 0<=target<=h*cap
    weights=[max(1.,float(k)) for k in raw]
    continuous=[0.]*h
    active=list(range(h));remaining=float(target)
    while active:
        total=sum(weights[i] for i in active)
        saturated=[i for i in active if remaining*weights[i]/total>=cap]
        if not saturated:
            for i in active:continuous[i]=remaining*weights[i]/total
            break
        for i in saturated:continuous[i]=float(cap)
        remaining-=len(saturated)*cap
        active=[i for i in active if i not in saturated]
    counts=[int(math.floor(x)) for x in continuous]
    leftover=target-sum(counts)
    candidates=sorted([i for i in range(h) if counts[i]<cap],key=lambda i:(-(continuous[i]-counts[i]),i))
    assert 0<=leftover<=len(candidates)
    for i in candidates[:leftover]:counts[i]+=1
    assert sum(counts)==target and all(0<=k<=cap for k in counts)
    return counts

class Controller(base.CostController):
    ratio=.1
    def choose(self, layer, query, scale):
        history_blocks = (self.position + 15) // 16
        sink, local, middle = fixed_and_middle_blocks(history_blocks, 2, 32, history_blocks-1)
        fixed = sink | local
        ids = torch.tensor(sorted(middle), device=query.device, dtype=torch.long)
        h = self.model.config.num_key_value_heads
        group = self.model.config.num_attention_heads // h
        grouped = query[0, :, 0].float().view(h, group, -1) * scale
        n = len(ids)
        classes = [-1] * h
        if self.mode == 'all' or not n:
            takes = [n]*h
            orders = [list(range(n))]*h
        else:
            assert bool(((ids+1)*16 <= self.position).all())
            digest = self.digests[layer]
            scenes = [dict(query=grouped[head], mean=digest['mean'][head, ids],
                           basis=digest['basis'][head, ids], eigenvalues=digest['eigenvalues'][head, ids],
                           counts=digest['counts'][ids], block_ids=ids, block_size=16,
                           token_position=self.position, layer_id=layer, kv_head_id=head) for head in range(h)]
            packed = pack_scenes(scenes, query.device)
            scores = forward_packed(self.ranker, packed).view(h, n).float().cpu()
            self.audit['ranker_calls'][layer] += 1
            if self.mode == 'adaptive':
                features = torch.stack([online_features(dict(scene, query=scene['query'].cpu()), scores[i])
                                        for i, scene in enumerate(scenes)]).to(query.device)
                probability = self.budget(features).softmax(-1).cpu()
                classes = quantile_actions(probability, self.quantile).tolist()
                takes = [n if BUDGETS[c] == 'ALL' else min(n, BUDGETS[c]) for c in classes]
                self.audit['budget_calls'][layer] += 1
                if self.capture and not self.selected:
                    self.feature_probe = (scenes[0], scores[0], features[0].cpu(), classes[0])
            else:
                takes = [min(n, int(self.mode.removeprefix('fixed')))]*h
            orders = scores.argsort(descending=True, stable=True).tolist()
        fixed_tokens = sum(min(16, max(0, self.position-b*16)) for b in fixed)+1
        # Autonomous reference: no fixed-ratio redistribution.
        middle_ids = ids.tolist()
        selected = []
        for head in range(h):
            blocks = fixed | {middle_ids[i] for i in orders[head][:takes[head]]}
            tokens = [t for b in sorted(blocks) for t in range(b*16, min((b+1)*16, self.position))]
            tokens.append(self.position)  # current token exactly once
            assert len(tokens) == len(set(tokens)) and max(tokens) == self.position
            selected.append(tokens)
        if len({len(x) for x in selected}) > 1:
            self.audit['ragged_attention_calls'] += 1
        if self.capture:
            self.selected.append({'position': self.position, 'layer': layer, 'indices': selected})
        self.step_rows.append({'layer':layer, 'selected_tokens':[len(x) for x in selected],
                               'middle_blocks':takes, 'classes':classes})
        return pad_indices(selected, query.device)

    def forward(self, module, layer, hidden, embeddings):
        assert hidden.shape[:2] == (1,1)
        shape = (1,1,-1,module.head_dim)
        q = module.q_proj(hidden).view(shape).transpose(1,2)
        k = module.k_proj(hidden).view(shape).transpose(1,2)
        v = module.v_proj(hidden).view(shape).transpose(1,2)
        q, k = apply_rotary_pos_emb(q,k,*embeddings)
        oldk, oldv = self.pairs[layer]
        assert oldk.shape[2] == self.position
        keys, values = append_pair(oldk, oldv, k, v)
        indices, valid = self.choose(layer,q,float(module.scaling))
        output = selected_attention(q, keys, values, indices, valid, float(module.scaling))
        self.pairs[layer] = (keys,values)
        if self.digests is not None and (self.position+1)%16 == 0:
            update = compact_digest(keys[0,:,-16:])
            before = self.digests[layer]
            assert before['counts'].numel() == (self.position+1)//16 - 1
            self.digests[layer] = {key:torch.cat([before[key],update[key]], 0 if key=='counts' else 1)
                                   for key in before}
            self.audit['digest_updates'] += 1
        output = module.o_proj(output.transpose(1,2).reshape(1,1,-1).contiguous())
        return output, None
