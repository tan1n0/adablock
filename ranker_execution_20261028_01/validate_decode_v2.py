"""Paired whole-decode validation; graph creation remains inside measured decode."""
import hashlib
import statistics
import traceback
from common import *
from graph_ranker_v2 import GraphRanker


class AuditController(e.gpu.Controller):
    audit_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_trace = []

    def choose(self, layer, query, scale):
        indices, valid = super().choose(layer, query, scale)
        if self.audit_enabled:
            row = self.step_rows[-1]
            digest = hashlib.sha256()
            for value in [indices, valid, row['classes'], row['middle_blocks']]:
                digest.update(value.cpu().numpy().tobytes())
            self.selection_trace.append([self.position, layer, digest.hexdigest()])
        return indices, valid


@torch.inference_mode()
def main():
    owners = exclusive_context()
    model, tok, models, eos = load()
    e.c.run.AdaptiveLlama = AuditController
    rows = json.loads((SOURCE / 'selected.json').read_text())
    selection = json.loads((SMALL / 'selection.json').read_text())['smoke']
    # Start with the demanding 32K input to expose memory and dynamic-shape problems early.
    selection = sorted(selection, key=lambda i: (-rows[i]['length'], i))
    dump('decode_v2_protocol.json', dict(indices=selection, branches=['S','B'], repetitions=3,
        timing_steps=16, audit_steps=32, natural_generation_cap=128, budget_target=.90,
        budget_quantile=.95, graph_build_included=True, graph_cache_per_decode='one exact shape',
        unchanged='FP32 weights, original summaries, features, budgets, sorting, padded attention, prompts and official scoring',
        sources_sha256={f.name:sha(f) for f in ROOT.glob('*.py')}))
    records = []
    for i in selection:
        row = rows[i]
        check_exclusive(owners)
        status('decode_validation', index=i, length=row['length'], completed=len(records), total=18)
        pairs, logits, prefill = e.prefill(model, row['input_ids'])
        digests, build = e.build(pairs, 'gpu_digest')
        old = json.loads((SOURCE / f'partial_{i}.json').read_text())
        for branch in ['S', 'B']:
            path = f'decode_v2/case_{i}_{branch}.json'
            assert not (ROOT / path).exists(), 'Do not overwrite completed cases'
            rank, budget = models['original_' + branch]
            record = dict(index=i, task=row['task'], length=row['length'], branch=branch,
                          prefill_seconds=prefill, digest_seconds=build, timing={}, answers={}, audit={})

            def run(version, cap, ignore_eos, audit=False):
                scorer = GraphRanker(ORIGINAL) if version == 'graph' else None
                error = torch.zeros((), device='cuda')
                def score(model, packed):
                    value = scorer(model, packed)
                    if audit:
                        ref = ORIGINAL(model, packed)
                        error.copy_(torch.maximum(error, (value-ref).abs().max()))
                    return value
                e.baseline.forward_packed = score if scorer else ORIGINAL
                AuditController.audit_enabled = audit
                clear()
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                try:
                    result = e.c.run.decode(model, pairs, logits, 'adaptive', rank, budget, digests, eos,
                        max_new=cap, ignore_eos=ignore_eos, capture=True, quantile=.95)
                finally:
                    e.baseline.forward_packed = ORIGINAL
                    AuditController.audit_enabled = False
                obj = result.pop('_controller')
                denominator = sum(x['full'] for x in obj.costs)
                stats = dict(ids=result['ids'], decode_seconds=result['decode_seconds'],
                    decode_steps=result['decode_steps'], hit_generation_limit=result['hit_generation_limit'],
                    logical_fraction=float(sum(x['read'] for x in obj.costs))/denominator,
                    padded_fraction=float(sum(x['padded'] for x in obj.costs))/denominator,
                    resident_MiB=resident/2**20, peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                    graph=scorer.metrics() if scorer else None,
                    digest_updates=obj.audit['digest_updates'])
                if audit:
                    stats.update(selection_trace=obj.selection_trace, score_max_abs_error=float(error))
                del obj, result, scorer
                return stats

            # Warm unchanged kernels; graphs are deliberately rebuilt in each timed run.
            warm = run('eager', 3, True)
            del warm
            audits = {v: run(v, 33, True, True) for v in ['eager','graph']}
            assert audits['eager']['ids'] == audits['graph']['ids'], 'Forced continuation changed'
            assert audits['eager']['selection_trace'] == audits['graph']['selection_trace'], 'Selection/budget changed'
            assert audits['graph']['score_max_abs_error'] == 0, 'Graph altered ranker scores'
            assert audits['graph']['graph']['builds'] >= 2, 'Dynamic candidate growth was not exercised'
            assert audits['graph']['digest_updates'] >= 32, 'Digest update path not exercised'
            record['audit'] = dict(exact_scores=True, selected_indices_and_budgets_equal=True,
                forced_tokens_equal=True, layer_steps=len(audits['eager']['selection_trace']),
                graph_builds=audits['graph']['graph']['builds'], digest_updates=audits['graph']['digest_updates'])
            dump(f'decode_v2/audit_{i}_{branch}.json', audits)
            del audits
            record['timing'] = {'eager': [], 'graph': []}
            for repeat in range(3):
                order = ['eager','graph'] if repeat % 2 == 0 else ['graph','eager']
                for version in order:
                    check_exclusive(owners)
                    stats = run(version, 17, True)
                    assert stats['decode_steps'] == 16
                    record['timing'][version].append(stats)
            for version in ['eager','graph']:
                stats = run(version, row['max_new'], False)
                pred = e.official.normalize_prediction(tok.decode(stats['ids'], skip_special_tokens=True), row['task'])
                stats.update(prediction=pred, score=e.official.metric_for(row['task'])([pred],[row['answers']]))
                assert stats['ids'] == old['methods'][branch]['ids'], ('Original QA output mismatch',i,branch,version)
                assert not stats['hit_generation_limit']
                record['answers'][version] = stats
            assert record['answers']['eager']['logical_fraction'] == record['answers']['graph']['logical_fraction']
            assert record['answers']['eager']['padded_fraction'] == record['answers']['graph']['padded_fraction']
            record['median_ms'] = {v: statistics.median(1000*x['decode_seconds']/x['decode_steps']
                                                      for x in rr) for v,rr in record['timing'].items()}
            record['speedup'] = record['median_ms']['eager']/record['median_ms']['graph']
            dump(path, record)
            records.append(record)
            summarize(records)
            print(json.dumps(dict(index=i,branch=branch,ms=record['median_ms'],speedup=record['speedup'],audit=record['audit'])),flush=True)
            check_exclusive(owners)
            status('decode_validation', index=i, completed=len(records), total=18)
            clear()
        del pairs, logits, digests
        e.reset_registry()
        clear()
    verification = verify()
    verification.update(cases=len(records), exact_scores_and_selections=True, original_answers_verified=True)
    dump('verification.json', verification)
    status('complete', completed=len(records), old_files_unchanged=True)


def summarize(records):
    summary = dict(completed=len(records), expected=18, by_length={})
    for length in [8192,16384,32768]:
        rr = [r for r in records if r['length']==length]
        if not rr:
            continue
        original = statistics.mean(r['median_ms']['eager'] for r in rr)
        optimized = statistics.mean(r['median_ms']['graph'] for r in rr)
        summary['by_length'][length] = dict(cases=len(rr), eager_ms=original, graph_ms=optimized,
            speedup=original/optimized, time_reduction_pct=100*(1-optimized/original),
            score_eager=statistics.mean(r['answers']['eager']['score'] for r in rr),
            score_graph=statistics.mean(r['answers']['graph']['score'] for r in rr),
            graph_build_ms_per_decode=statistics.mean(statistics.median(x['graph']['build_seconds']*1000
                                          for x in r['timing']['graph']) for r in rr),
            peak_allocated_extra_MiB=statistics.mean(statistics.median(x['peak_allocated_MiB'] for x in r['timing']['graph'])-
                                  statistics.median(x['peak_allocated_MiB'] for x in r['timing']['eager']) for r in rr),
            peak_reserved_extra_MiB=statistics.mean(statistics.median(x['peak_reserved_MiB'] for x in r['timing']['graph'])-
                                  statistics.median(x['peak_reserved_MiB'] for x in r['timing']['eager']) for r in rr))
    summary['audited_layer_steps'] = sum(r['audit']['layer_steps'] for r in records)
    dump('decode_v2_summary.json', summary)
    return summary


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        marker = SMALL / 'PAUSE_KEEPALIVE'
        if marker.exists() and str(ROOT) in marker.read_text():
            marker.unlink()
