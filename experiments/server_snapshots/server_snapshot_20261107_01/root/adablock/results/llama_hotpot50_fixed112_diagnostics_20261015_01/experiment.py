"""Frozen HotpotQA test diagnostics; never use these labels for training."""
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import run
import runtime as rt

BASE = Path('/root/adablock/results/llama_hotpot50_adaptive_20261015_01')
ORIGINAL_ATTENTION = rt.selected_attention


class Diagnostic(rt.AdaptiveLlama):
    rows = None

    def choose(self, layer, query, scale):
        mode = self.mode
        self.mode = 'adaptive'
        indices, valid = super().choose(layer, query, scale)
        self.mode = mode
        h = self.model.config.num_key_value_heads
        group = self.model.config.num_attention_heads // h
        count = (self.position + 15) // 16
        sink, local, middle = rt.fixed_and_middle_blocks(count, 2, 32, count-1)
        ids = torch.tensor(sorted(middle), device=query.device, dtype=torch.long)
        digest = self.digests[layer]
        grouped = query[0, :, 0].float().view(h, group, -1) * scale
        scenes = [dict(query=grouped[head], mean=digest['mean'][head, ids],
                       basis=digest['basis'][head, ids], eigenvalues=digest['eigenvalues'][head, ids],
                       counts=digest['counts'][ids], block_ids=ids, block_size=16,
                       token_position=self.position, layer_id=layer, kv_head_id=head) for head in range(h)]
        scores = rt.forward_packed(self.ranker, rt.pack_scenes(scenes, query.device)).view(h,-1).float().cpu()
        features = torch.stack([rt.online_features(dict(s,query=s['query'].cpu()),scores[i]) for i,s in enumerate(scenes)])
        prob = self.budget(features.to(query.device)).softmax(-1).cpu()
        classes = rt.quantile_actions(prob,self.quantile).tolist()
        assert classes == self.step_rows[-1]['classes']
        self.probe = (layer, ids, scores, classes, prob.argmax(-1).tolist())
        if mode == 'all':
            indices = torch.arange(self.position+1,device=query.device)[None].expand(h,-1)
            valid = torch.ones_like(indices,dtype=torch.bool)
        return indices, valid

    def attention(self, query, keys, values, indices, valid, scale):
        layer, ids, scores, classes, raw = self.probe
        h = keys.shape[1]
        # Explicit FP32 teacher probabilities, averaged over the four Q heads per KV head.
        q = query[0,:,0].float().reshape(h,-1,query.shape[-1])
        mass = torch.einsum('hgd,htd->hgt',q,keys[0].float()).mul(scale).softmax(-1).mean(1)
        assert torch.allclose(mass.sum(-1),torch.ones(h,device=mass.device),atol=2e-6)
        pad = (-mass.shape[-1]) % 16
        blocks = torch.nn.functional.pad(mass,(0,pad)).reshape(h,-1,16).sum(-1)
        candidate = blocks[:,ids].cpu()
        fixed = 1-candidate.sum(-1)
        orders = scores.argsort(descending=True,stable=True)
        for head in range(h):
            ranked = candidate[head,orders[head]]
            curve = torch.cat([fixed[head,None],fixed[head]+ranked.cumsum(0)]).clamp(max=1)
            ideal = torch.cat([fixed[head,None],fixed[head]+candidate[head].sort(descending=True).values.cumsum(0)])
            n = len(ranked)
            required = int(torch.nonzero(curve >= .95)[0])
            oracle = int(torch.nonzero(ideal >= .95)[0])
            pred = n if rt.BUDGETS[classes[head]] == 'ALL' else min(n,rt.BUDGETS[classes[head]])
            argmax = n if rt.BUDGETS[raw[head]] == 'ALL' else min(n,rt.BUDGETS[raw[head]])
            minimum_class = next(i for i,b in enumerate(rt.BUDGETS) if (n if b=='ALL' else min(n,b)) >= required)
            minimum_bucket = n if rt.BUDGETS[minimum_class]=='ALL' else min(n,rt.BUDGETS[minimum_class])
            self.rows.append(dict(position=self.position,layer=layer,head=head,candidates=n,
                fixed_mass=float(fixed[head]),required=required,oracle_required=oracle,
                minimum_bucket=minimum_bucket,predicted=pred,argmax=argmax,
                mass_predicted=float(curve[pred]),mass_argmax=float(curve[argmax]),
                mass32=float(curve[min(32,n)]),mass64=float(curve[min(64,n)]),mass112=float(curve[min(112,n)])))
        return ORIGINAL_ATTENTION(query,keys,values,indices,valid,scale)

    def install(self):
        self.rows=[]
        super().install()
        rt.selected_attention=self.attention

    def uninstall(self):
        rt.selected_attention=ORIGINAL_ATTENTION
        super().uninstall()


@torch.inference_mode()
def main(shard):
    model,tok,ranker,budget,eos,config=run.load()
    selected=json.loads((BASE/'selected.json').read_text())
    old={r['source_index']:r for f in BASE.glob('worker_*.jsonl') for line in f.read_text().splitlines() if (r:=json.loads(line))}
    warm=run.prefill(model,selected[0]['input_ids'][:128]); del warm
    with (ROOT/f'worker_{shard}.jsonl').open('x') as journal:
        for index,row in enumerate(selected):
            if index%4 != shard: continue
            pairs,logits,prefill_seconds=run.prefill(model,row['input_ids'])
            run.sync(); start=time.perf_counter()
            digests=rt.build_prefix_digests(pairs)
            run.sync(); index_seconds=time.perf_counter()-start
            result=run.decode(model,pairs,logits,'fixed112',ranker,budget,digests,eos,quantile=config['budget_quantile'])
            result['prediction']=tok.decode(result['ids'],skip_special_tokens=True)
            result['f1']=max(run.token_f1(result['prediction'],a) for a in row['answers'])*100
            diagnostics={}
            for trajectory,mode in [('full','all'),('adaptive','adaptive')]:
                run.AdaptiveLlama=Diagnostic
                try:
                    trace=run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,
                                     quantile=config['budget_quantile'],capture=True)
                finally:
                    run.AdaptiveLlama=rt.AdaptiveLlama
                controller=trace.pop('_controller')
                assert trace['ids']==old[row['source_index']]['methods'][trajectory]['ids'], (row['source_index'],trajectory,trace['ids'])
                diagnostics[trajectory]=controller.rows
                del controller,trace
            record=dict(source_index=row['source_index'],input_ids_sha256=row['input_ids_sha256'],
                        truncated=row['truncated'],prefill_seconds=prefill_seconds,index_seconds=index_seconds,
                        fixed112=result,diagnostics=diagnostics,trajectory_ids_match=True)
            journal.write(json.dumps(record)+'\n'); journal.flush()
            print(json.dumps(dict(source_index=row['source_index'],f1=result['f1'],done=True)),flush=True)
            del pairs,logits,digests,diagnostics,record,result
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main(int(sys.argv[1]))
