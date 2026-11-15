"""128-step dynamic-shape and memory regression after removing per-build cleanup."""
import traceback
from common import *
from graph_ranker_v3 import GraphRanker


class ObservedGraph(GraphRanker):
    def __init__(self):
        super().__init__(ORIGINAL)
        self.memory = []

    def _build(self, model, packed, signature):
        super()._build(model, packed, signature)
        self.memory.append(dict(candidates=packed['mean'].shape[0],
            allocated_MiB=torch.cuda.memory_allocated()/2**20,
            reserved_MiB=torch.cuda.memory_reserved()/2**20))


@torch.inference_mode()
def main():
    owners = exclusive_context()
    status('long_decode_memory_measurement',steps=128)
    model,tok,models,eos = load()
    rank,budget = models['original_B']
    row=max(json.loads((SOURCE/'selected.json').read_text()),key=lambda r:len(r['input_ids']))
    pairs,logits,_=e.prefill(model,row['input_ids'])
    digests,_=e.build(pairs,'gpu_digest')
    graph=ObservedGraph()
    error=torch.zeros((),device='cuda')
    def exact_score(model,packed):
        actual=graph(model,packed)
        expected=ORIGINAL(model,packed)
        error.copy_(torch.maximum(error,(actual-expected).abs().max()))
        return actual
    results={}
    for version,scorer in [('eager',ORIGINAL),('graph',graph)]:
        e.baseline.forward_packed=scorer
        clear()
        torch.cuda.reset_peak_memory_stats()
        out=e.c.run.decode(model,pairs,logits,'adaptive',rank,budget,digests,eos,
                          max_new=129,ignore_eos=True,capture=True,quantile=.95)
        obj=out.pop('_controller')
        results[version]=dict(ids=out['ids'],decode_steps=out['decode_steps'],decode_seconds=out['decode_seconds'],
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            digest_updates=obj.audit['digest_updates'])
        del out,obj
        check_exclusive(owners)
    e.baseline.forward_packed=ORIGINAL
    assert results['eager']['ids']==results['graph']['ids']
    assert graph.builds>=8
    report=dict(passed=True,prompt_tokens=len(row['input_ids']),steps=128,
        exact_scores_previously_verified=True,exact_generated_tokens=True,
        graph=graph.metrics(),memory_by_build=graph.memory,results=results,
        note='Uninstrumented 128-step run: no extra reference score or selection hashing; peak differences are measured at equal output tokens')
    dump('long_decode_memory_only.json',report)
    verify()
    status('complete',completed=18,final_version='v3',long_decode_check=True,
           stages_profiled=True,old_files_unchanged=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        status('failed_long_decode_check',error=traceback.format_exc())
        raise
    finally:
        marker=SMALL/'PAUSE_KEEPALIVE'
        if marker.exists() and str(ROOT) in marker.read_text():
            marker.unlink()
