"""Same-session whole-decode stage profile after exact-output regression passes."""
import importlib.util
import statistics
import traceback
from common import *
from graph_ranker_v2 import GraphRanker

SPEC = importlib.util.spec_from_file_location('old_profiler',
    '/root/adablock_decode_profile_20261022_03/profile_decode.py')
old_profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old_profiler)
old_profiler.torch = torch


def wrapped(fn, name):
    def call(*args, **kwargs):
        with torch.profiler.record_function('stage::'+name):
            return fn(*args, **kwargs)
    return call


@torch.inference_mode()
def main():
    assert json.loads((ROOT/'verification.json').read_text())['cases']==18
    marker = SMALL/'PAUSE_KEEPALIVE'
    assert not marker.exists(), 'Do not take over a pre-existing timing lock'
    marker.write_text(str(ROOT)+' stage profiling')
    time.sleep(8)
    owners = exclusive_context()
    model, tok, models, eos = load()
    rank, budget = models['original_B']
    row = next(r for r in json.loads((SOURCE/'selected.json').read_text())
               if r['length']==32768 and r['task']=='niah_single_1')
    p, logits, _ = e.prefill(model,row['input_ids'])
    d, _ = e.build(p,'gpu_digest')
    originals = dict(features=e.gpu.features_gpu, select=e.gpu.selected_gpu,
                     attention=e.baseline.selected_attention, budget=budget.forward)
    report = {}
    for version in ['full','eager','graph']:
        status('final_stage_profile',version=version)
        graph = GraphRanker(ORIGINAL) if version=='graph' else None
        scorer = graph if graph else ORIGINAL
        e.baseline.forward_packed = scorer
        def run():
            return e.c.run.decode(model,p,logits,'full' if version=='full' else 'adaptive',rank,budget,d,eos,
                                  max_new=2,ignore_eos=True,quantile=.95)
        # Same single step repeated: every model sees identical queries and positions.
        first=run()
        expected=first['ids']
        times=[]
        for _ in range(10):
            out=run()
            assert out['ids']==expected
            times.append(1000*out['decode_seconds'])
        e.baseline.forward_packed=wrapped(scorer,'ranker')
        e.gpu.features_gpu=wrapped(originals['features'],'budget_features')
        e.gpu.selected_gpu=wrapped(originals['select'],'block_selection_and_width_sync')
        budget.forward=wrapped(originals['budget'],'budget_network')
        old_profiler.ACTIVE=True
        e.baseline.selected_attention=old_profiler.selected_attention
        iterations=4
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(iterations):
                out=run()
                assert out['ids']==expected
        old_profiler.ACTIVE=False
        e.baseline.forward_packed=ORIGINAL
        e.gpu.features_gpu=originals['features']
        e.gpu.selected_gpu=originals['select']
        e.baseline.selected_attention=originals['attention']
        budget.forward=originals['budget']
        events=prof.events()
        kernels=[v for v in events if v.device_type==torch.autograd.DeviceType.CUDA and not v.is_user_annotation]
        total=sum(v.device_time_total for v in kernels)/1000/iterations
        stages={v.key.removeprefix('stage::'):dict(gpu_ms=v.device_time_total/1000/iterations,
                   cpu_range_ms=v.cpu_time_total/1000/iterations,calls=v.count/iterations)
                for v in prof.key_averages() if v.key.startswith('stage::')
                and v.device_type==torch.autograd.DeviceType.CPU}
        attributed=sum(v['gpu_ms'] for v in stages.values())
        # Graph replay attribution depends on the installed profiler. Preserve raw evidence.
        report[version]=dict(warm_wall_ms=times,median_warm_wall_ms=statistics.median(times),
            total_gpu_kernel_ms=total,kernel_count_per_token=len(kernels)/iterations,stages=stages,
            attributed_gpu_ms=attributed,other_gpu_ms=total-attributed,
            timing_scope='Warm graph, repeated same single token; separate from cold-build 16-step regression timing',
            graph_capture_excluded=True,exact_token_parity=True)
        prof.export_chrome_trace(str(ROOT/f'decode_trace_{version}.json'))
        dump('stage_profile.json',report)
        del graph,scorer,prof,events,kernels,out,first
        clear()
        check_exclusive(owners)
    verify()
    status('complete',completed=18,stages_profiled=True,old_files_unchanged=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        status('failed_stage_profile',error=traceback.format_exc())
        raise
    finally:
        marker=SMALL/'PAUSE_KEEPALIVE'
        if marker.exists() and str(ROOT) in marker.read_text():
            marker.unlink()
