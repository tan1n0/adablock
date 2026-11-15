"""Decompose selection coverage on the recorded Full decoding trajectory."""
import json
from pathlib import Path
import statistics
import subprocess
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/adablock_ruler_3090_free_budget_20261113_full_01')
BUDGETS = (64, 128, 256, 512)
METHODS = ('oracle_token', 'oracle_block', 'oracle_in_coarse', 'B_in_coarse')
LOSSES = ('block_granularity', 'coarse_filter', 'B_ranking')
N = 20


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def work(shard):
    sys.path.insert(0, '/root/adablock_ruler_autonomous_20261022_01')
    import experiment as e
    import torch
    from fast_digest import compact_digest
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from utils.ranking_batch import forward_packed, pack_scenes

    class Capture:
        position = None

        def __call__(self, attn, args, kwargs, output):
            if self.position is None:
                return
            layer = attn.layer_idx
            assert layer == len(self.layers)
            p = self.position
            complete = p // 16
            hidden = kwargs['hidden_states']
            q = attn.q_proj(hidden).view(1, 1, 32, 128).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, *kwargs['position_embeddings'])
            cache = kwargs.get('past_key_value', kwargs.get('past_key_values'))
            key = cache[layer][0][0]
            assert key.shape == (8, p+1, 128)
            q = q[0, :, 0].float().view(8, 4, 128) * attn.scaling
            prob = (q @ key.float().transpose(-1, -2)).softmax(-1).mean(1)
            fixed = prob[:, complete*16:].sum(-1)
            hist = prob[:, :complete*16]
            blocks = hist.view(8, complete, 16).sum(-1)
            if layer not in self.digests:
                self.digests[layer] = compact_digest(key[:, :complete*16])
            d = self.digests[layer]
            old = d['counts'].numel()
            if old < complete:
                extra = compact_digest(key[:, old*16:complete*16])
                d = {name:torch.cat((x, extra[name]), 0 if name=='counts' else 1)
                     for name,x in d.items()}
                self.digests[layer] = d
            assert d['counts'].numel() == complete
            coarse = (q.mean(1, keepdim=True) @ d['mean'].transpose(-1, -2)).squeeze(1)
            ids = coarse.topk(min(256, complete), sorted=False).indices
            scenes = [dict(query=q[h], mean=d['mean'][h, ids[h]], basis=d['basis'][h, ids[h]],
                           eigenvalues=d['eigenvalues'][h, ids[h]], counts=d['counts'][ids[h]],
                           block_ids=ids[h], block_size=16, token_position=p, layer_id=layer, kv_head_id=h)
                      for h in range(8)]
            logits = forward_packed(ranker, pack_scenes(scenes, q.device)).view(8, -1)
            candidate_mass = blocks.gather(1, ids)
            token_best = hist.topk(max(BUDGETS)).values.cumsum(-1)
            block_best = blocks.topk(max(BUDGETS)//16).values.cumsum(-1)
            coarse_best = candidate_mass.topk(max(BUDGETS)//16).values.cumsum(-1)
            coverages = []
            for budget in BUDGETS:
                k = budget//16
                chosen = logits.topk(k, sorted=False).indices
                b_mass = candidate_mass.gather(1, chosen).sum(-1)
                coverages.append(fixed[:, None] + torch.stack((token_best[:, budget-1],
                    block_best[:, k-1], coarse_best[:, k-1], b_mass), dim=-1))
            self.layers.append(torch.stack(coverages))

    with torch.inference_mode():
        model, _, models, _ = e.load()
        ranker = models['B'][0]
        del models
        capture = Capture()
        hooks = [layer.self_attn.register_forward_hook(capture, with_kwargs=True) for layer in model.model.layers]
        rows = json.loads((SOURCE/'selected.json').read_text())
        for i in range(shard, N, 4):
            out = f'case_{i:03d}.json'
            if (ROOT/out).exists():
                continue
            row = rows[i]
            source = json.loads((SOURCE/f'quality_{i:03d}.json').read_text())
            assert row['input_sha256'] == source['input_sha256']
            recorded = source['methods']['full']['ids']
            assert len(recorded) >= 2
            start = time.monotonic()
            pairs, first, _ = e.prefill(model, row['input_ids'])
            assert int(first.argmax()) == recorded[0]
            cache = e.BufferCache.from_legacy_cache(pairs)
            capture.digests = {}
            steps = []
            for step, token in enumerate(recorded[:-1]):
                p = len(row['input_ids']) + step
                capture.position = p
                capture.layers = []
                pos = torch.tensor([p], device='cuda')
                result = model.model(input_ids=torch.tensor([[token]], device='cuda'),
                                     position_ids=pos[None], cache_position=pos,
                                     past_key_values=cache, use_cache=True)
                assert int(model.lm_head(result.last_hidden_state[:, -1:])[0,-1].float().argmax()) == recorded[step+1], (i,step)
                assert len(capture.layers) == 32
                values = torch.stack(capture.layers)
                assert values.isfinite().all().item()
                assert values.min().item() >= 0 and values.max().item() <= 1+2e-6
                assert (values[..., :-1] + 2e-6 >= values[..., 1:]).all().item()
                steps.append(values.cpu().tolist())
            capture.position = None
            capture.digests.clear()
            del pairs,cache,result,values
            e.reset_registry()
            save(out, dict(index=i, task=row['task'], input_sha256=row['input_sha256'],
                           steps=steps, prompt_tokens=len(row['input_ids']), seconds=time.monotonic()-start))
            print(i, row['task'], len(steps), round(time.monotonic()-start, 1), flush=True)
        for h in hooks:
            h.remove()


def summarize():
    import numpy as np
    cases = [json.loads(p.read_text()) for p in sorted(ROOT.glob('case_*.json'))]
    assert len(cases) == N
    out = {}
    for task in ('all', 'niah_multivalue', 'vt'):
        group = [c for c in cases if task=='all' or task==c['task']]
        arrays = [np.asarray(c['steps'], dtype=np.float64) for c in group]
        pooled = np.concatenate(arrays)
        out[task] = dict(n=len(group), decode_steps=len(pooled), budgets={})
        for j,b in enumerate(BUDGETS):
            cov = np.stack([x[:, :, j].mean(axis=(0,1,2)) for x in arrays]).mean(axis=0)*100
            drops = -np.diff(pooled[:, :, j], axis=-1)*100
            out[task]['budgets'][b] = dict(
                mean_coverage_pct=dict(zip(METHODS, cov.tolist())),
                mean_loss_pp=dict(zip(LOSSES, (-np.diff(cov)).tolist())),
                p95_loss_pp=dict(zip(LOSSES, np.quantile(drops.reshape(-1,3), .95, axis=0).tolist())),
                layer_mean_loss_pp=drops.mean(axis=(0,2)).tolist())
    save('summary.json', out)


def launch():
    jobs = []
    started = time.monotonic()
    try:
        for gpu in range(4):
            with (ROOT/f'worker_{gpu}.log').open('a') as log:
                jobs.append(subprocess.Popen([sys.executable,'-u',__file__,str(gpu)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),
                    stdout=log,stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in jobs]
            if any(c not in (None,0) for c in codes):
                raise RuntimeError(f'Worker codes: {codes}')
            if all(c==0 for c in codes):
                break
            save('status.json',dict(stage='running',completed=len(list(ROOT.glob('case_*.json'))),target=N))
            time.sleep(3)
        summarize()
        save('status.json',dict(stage='complete',completed=N,target=N,seconds=time.monotonic()-started))
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for p in jobs:
            p.wait()


if __name__=='__main__':
    if sys.argv[1]=='launch':
        try:
            launch()
        except Exception:
            save('status.json',dict(stage='failed',traceback=traceback.format_exc()))
            raise
    else:
        work(int(sys.argv[1]))
