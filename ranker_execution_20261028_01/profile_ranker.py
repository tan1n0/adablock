"""Paired ranker-only measurements on exactly identical real 32K inputs."""
import json
import statistics
import time
import traceback
from contextlib import contextmanager
from common import *
from graph_ranker import GraphRanker


@contextmanager
def module_ranges(model):
    modules = {'query_projection': model.query_proj, 'mean_projection': model.mean_proj,
               'layer_embedding': model.layer_embedding, 'head_embedding': model.head_embedding,
               'output_linear': model.output}
    for i, module in enumerate(model.interaction):
        modules[f'interaction_{i}_{type(module).__name__}'] = module
    handles = []
    for name, module in modules.items():
        stack = []
        def pre(mod, args, name=name, stack=stack):
            ctx = torch.profiler.record_function('rankpart::' + name)
            ctx.__enter__()
            stack.append(ctx)
        def post(mod, args, result, stack=stack):
            stack.pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def measure(fn, repeats=5, iterations=20):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            out = fn()
        enqueue = time.perf_counter() - start
        torch.cuda.synchronize()
        times.append(dict(wall_ms=1000*(time.perf_counter()-start)/iterations,
                          cpu_enqueue_ms=1000*enqueue/iterations))
    return dict(repeats=times, median_wall_ms=statistics.median(t['wall_ms'] for t in times),
                median_cpu_enqueue_ms=statistics.median(t['cpu_enqueue_ms'] for t in times))


@torch.inference_mode()
def main():
    owners = exclusive_context()
    protect()
    status('capture_real_inputs')
    model, tok, models, eos = load()
    rows = json.loads((SOURCE / 'selected.json').read_text())
    row = next(r for r in rows if r['length'] == 32768 and r['task'] == 'niah_single_1')
    p, logits, _ = e.prefill(model, row['input_ids'])
    digests, _ = e.build(p, 'gpu_digest')
    captured = {}
    calls = 0
    def capture(rank, packed):
        nonlocal calls
        if calls in [0, 15, 31]:
            captured[calls] = {k: v.clone() for k, v in packed.items()}
        calls += 1
        return ORIGINAL(rank, packed)
    e.baseline.forward_packed = capture
    rank, budget = models['original_B']
    result = e.c.run.decode(model, p, logits, 'adaptive', rank, budget, digests, eos,
                           max_new=2, ignore_eos=True, quantile=.95)
    e.baseline.forward_packed = ORIGINAL
    assert set(captured) == {0, 15, 31}
    torch.save({k: {a: b.cpu() for a, b in v.items()} for k, v in captured.items()}, ROOT / 'rank_inputs.pt')
    dump('protocol.json', dict(prompt_tokens=len(row['input_ids']), layers=[0,15,31], names=NAMES,
         same_query_and_digests=True, fp32=True, ranker_chunk_unchanged=True,
         scope='Ranker-only wall time is not whole decode. GPU kernel sums are not wall-time shares.',
         graph='Exact-shape replay of unchanged operations; current query/summary copied every call; build cost separate here and included in end-to-end validation.',
         source_sha256={f.name:sha(f) for f in ROOT.glob('*.py')}))
    del model, tok, p, logits, digests, result
    e.reset_registry()
    clear()
    report = []
    for layer, packed in captured.items():
        for name in NAMES:
            check_exclusive(owners)
            status('ranker_microprofile', layer=layer, name=name, completed=len(report), total=18)
            rank, budget = models[name]
            reference = ORIGINAL(rank, packed).clone()
            eager = measure(lambda: ORIGINAL(rank, packed))
            graph = GraphRanker(ORIGINAL)
            value = graph(rank, packed)
            assert torch.equal(reference, value), ('Graph changed score', layer, name)
            replay = measure(lambda: graph(rank, packed))
            # Changing the query and identity must update output, not reuse old scores.
            changed = {k:v.clone() for k,v in packed.items()}
            changed['query'].mul_(1.01)
            changed['layer_id'].add_(1).remainder_(32)
            expected = ORIGINAL(rank, changed)
            actual = graph(rank, changed)
            assert torch.equal(expected, actual), ('Dynamic input parity', layer, name)
            assert not torch.equal(reference, actual), 'The dynamic-input check did not exercise a change'
            metrics = graph.metrics()
            del graph, changed, actual, expected, value
            clear()
            iterations = 5
            with module_ranges(rank):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(iterations):
                        with torch.profiler.record_function('ranker_total'):
                            value = ORIGINAL(rank, packed)
                    torch.cuda.synchronize()
            events = prof.events()
            kernels = [x for x in events if x.device_type == torch.autograd.DeviceType.CUDA
                       and not x.is_user_annotation]
            gpu_us = sum(x.device_time_total for x in kernels)
            parts = {x.key.removeprefix('rankpart::'): dict(gpu_ms=x.device_time_total/1000/iterations,
                     cpu_range_ms=x.cpu_time_total/1000/iterations, calls=x.count/iterations)
                     for x in prof.key_averages() if x.key.startswith('rankpart::')
                     and x.device_type == torch.autograd.DeviceType.CPU}
            parts['feature_processing_other'] = dict(gpu_ms=gpu_us/1000/iterations-
                                                     sum(x['gpu_ms'] for x in parts.values()))
            record = dict(name=name, layer=layer, candidates=packed['mean'].shape[0],
                          eager=eager, graph=replay, graph_setup=metrics,
                          graph_speedup=eager['median_wall_ms']/replay['median_wall_ms'],
                          kernel_count_per_call=len(kernels)/iterations,
                          kernel_sum_ms=gpu_us/1000/iterations, parts=parts, exact_scores=True)
            dump(f'micro/layer{layer}_{name}.json', record)
            dump(f'micro/operators_layer{layer}_{name}.json', [dict(name=x.key, calls=x.count/iterations,
                  self_cpu_ms=x.self_cpu_time_total/1000/iterations,
                  self_gpu_ms=x.self_device_time_total/1000/iterations) for x in prof.key_averages()])
            if layer == 15 and name.endswith('_B'):
                prof.export_chrome_trace(str(ROOT / f'micro/trace_{name}.json'))
            report.append(record)
            dump('micro_summary.json', dict(completed=len(report), expected=18, cases=report))
            print(json.dumps(record), flush=True)
            del prof, events, kernels, reference, value
            check_exclusive(owners)
    dump('micro_verification.json', verify())
    status('microprofile_complete', completed=len(report), next='Inspect measured bottleneck before full-decode validation')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
