"""Attribute graph nodes via cudaGraphLaunch correlation, preserving raw profiler data."""
import bisect
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    original = json.loads((ROOT/'stage_profile.json').read_text())
    report = {}
    for version in ['eager','graph']:
        events = json.loads((ROOT/f'decode_trace_{version}.json').read_text())['traceEvents']
        ranges = collections.defaultdict(list)
        for event in events:
            if event.get('cat')=='user_annotation' and event['name'].startswith('stage::'):
                ranges[event['pid'],event['tid']].append(event)
        for key in ranges:
            ranges[key].sort(key=lambda e:e['ts'])
        starts = {key:[e['ts'] for e in group] for key,group in ranges.items()}
        mapping = {}
        graph_correlations = set()
        for event in events:
            if event.get('cat')!='cuda_runtime':
                continue
            key = (event['pid'],event['tid'])
            if key not in ranges:
                continue
            idx = bisect.bisect_right(starts[key],event['ts'])-1
            if idx<0:
                continue
            parent = ranges[key][idx]
            if event['ts']+event.get('dur',0)>parent['ts']+parent['dur']+.01:
                continue
            correlation = event.get('args',{}).get('correlation')
            if correlation is None:
                continue
            mapping[correlation] = parent['name'].removeprefix('stage::')
            if event['name']=='cudaGraphLaunch':
                assert mapping[correlation]=='ranker'
                graph_correlations.add(correlation)
        totals = collections.Counter()
        counts = collections.Counter()
        graph_nodes = collections.Counter()
        for event in events:
            if event.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:
                continue
            correlation = event.get('args',{}).get('correlation')
            label = mapping.get(correlation,'other')
            totals[label] += event['dur']
            counts[label] += 1
            if correlation in graph_correlations:
                graph_nodes[correlation] += 1
        total = sum(totals.values())/4000
        assert abs(total-original[version]['total_gpu_kernel_ms'])<.01
        if version=='graph':
            assert len(graph_nodes)==128
            assert set(graph_nodes.values())=={337}, graph_nodes
        report[version] = dict(total_gpu_ms=total, stages={name:dict(gpu_ms=value/4000,
            gpu_percent=100*value/sum(totals.values()),kernel_calls_per_token=counts[name]/4)
            for name,value in totals.most_common()},graph_launches=len(graph_nodes),
            graph_nodes_per_launch=sorted(set(graph_nodes.values())),
            method='CUDA runtime correlation to enclosing non-overlapping CPU stage; graph nodes mapped to cudaGraphLaunch correlation',
            cpu_warning='CPU ranges overlap GPU work and may wait for earlier graph kernels; do not sum them with GPU time or interpret budget-feature CPU range as pure feature compute')
    (ROOT/'stage_profile_corrected.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
