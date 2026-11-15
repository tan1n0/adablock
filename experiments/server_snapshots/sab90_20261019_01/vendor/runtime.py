"""Llama quality prototype with frozen ranker and per-KV-head adaptive budgets."""
import sys
import types
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PILOT = Path('/root/adablock/results/llama_budget_rank_pilot_20261015_01')
sys.path.insert(0, str(PILOT))
from budget_pilot import BudgetClassifier, BUDGETS, online_features, quantile_actions
from utils.block_oracle import fixed_and_middle_blocks
from utils.qk_block_oracle import build_qk_block_digests
from utils.ranking_batch import pack_scenes, forward_packed


def compact_digest(keys):
    digest = build_qk_block_digests(keys, block_size=16, cobs_rank=4)
    return dict(mean=digest.mean, basis=digest.covariance_basis, eigenvalues=digest.covariance_eigenvalues,
                counts=digest.counts)


@torch.no_grad()
def build_prefix_digests(pairs):
    # Incomplete newest block is fixed and cannot be a retrieval candidate.
    complete = pairs[0][0].shape[2] // 16 * 16
    return [compact_digest(k[0, :, :complete]) for k, v in pairs]


def pad_indices(selected, device):
    lengths = torch.tensor([len(x) for x in selected], device=device)
    width = int(lengths.max())
    indices = torch.zeros((len(selected), width), dtype=torch.long, device=device)
    valid = torch.arange(width, device=device)[None] < lengths[:, None]
    for h, ids in enumerate(selected):
        indices[h, :len(ids)] = torch.tensor(ids, device=device)
    return indices, valid


def selected_attention(query, keys, values, indices, valid, scale):
    heads, width = indices.shape
    index = indices[..., None].expand(heads, width, keys.shape[-1])
    k = keys[0].gather(1, index)[None]
    v = values[0].gather(1, index)[None]
    group = query.shape[1] // heads
    if bool(valid.all()):
        # Match Transformers 4.55's native GQA SDPA dispatch when unmasked.
        return F.scaled_dot_product_attention(query.contiguous(), k.contiguous(), v.contiguous(),
                   dropout_p=0., is_causal=False, scale=scale, enable_gqa=True)
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    mask = valid.repeat_interleave(group, 0)[None, :, None, :]
    return F.scaled_dot_product_attention(query.contiguous(), k.contiguous(), v.contiguous(),
                   attn_mask=mask, dropout_p=0., is_causal=False, scale=scale)


class AdaptiveLlama:
    def __init__(self, model, pairs, prefix_digests, ranker, budget, mode, quantile=.95, capture=False):
        assert model.config.model_type == 'llama'
        self.model, self.ranker, self.budget = model, ranker, budget
        self.mode, self.quantile, self.capture = mode, quantile, capture
        self.pairs = list(pairs)
        # Tensor references are immutable; append operations allocate new tensors.
        self.digests = list(prefix_digests) if prefix_digests is not None else None
        self.position = pairs[0][0].shape[2]
        self.original = []
        self.steps = []
        self.selected = []
        self.audit = {'ranker_calls': [0]*len(pairs), 'budget_calls': [0]*len(pairs),
                      'ragged_attention_calls': 0, 'digest_updates': 0}

    def install(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            module = layer.self_attn
            self.original.append(module.forward)
            def forward(module, hidden_states, position_embeddings, attention_mask=None,
                        past_key_value=None, cache_position=None, _layer=layer_id, **kwargs):
                return self.forward(module, _layer, hidden_states, position_embeddings)
            module.forward = types.MethodType(forward, module)

    def uninstall(self):
        for layer, forward in zip(self.model.model.layers, self.original):
            layer.self_attn.forward = forward
        self.original = []

    def begin_step(self):
        self.step_rows = []

    def finish_step(self):
        assert len(self.step_rows) == len(self.pairs)
        assert [r['layer'] for r in self.step_rows] == list(range(len(self.pairs)))
        totals = [v for r in self.step_rows for v in r['selected_tokens']]
        middle = [v for r in self.step_rows for v in r['middle_blocks']]
        self.steps.append({'position': self.position, 'mean_selected_tokens': sum(totals)/len(totals),
                           'mean_middle_blocks': sum(middle)/len(middle),
                           'read_fraction': sum(totals)/len(totals)/(self.position+1),
                           'budget_classes': [i for r in self.step_rows for i in r['classes']]})
        self.position += 1

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
        keys, values = torch.cat([oldk,k],2), torch.cat([oldv,v],2)
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


def ragged_test():
    torch.manual_seed(123)
    q=torch.randn(1,8,1,16)
    k,v=torch.randn(1,2,20,16),torch.randn(1,2,20,16)
    selected=[[0,3,8,19],[1,19]]
    ids,valid=pad_indices(selected,'cpu')
    got=selected_attention(q,k,v,ids,valid,.25)
    expected=torch.cat([F.scaled_dot_product_attention(q[:,head*4:(head+1)*4],
        k[:,head:head+1,positions].repeat_interleave(4,1),v[:,head:head+1,positions].repeat_interleave(4,1),
        dropout_p=0.,scale=.25) for head,positions in enumerate(selected)],1)
    assert torch.allclose(got,expected,atol=2e-7,rtol=1e-6)
    return float((got-expected).abs().max())
