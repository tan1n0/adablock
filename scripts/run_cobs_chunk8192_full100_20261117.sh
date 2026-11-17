#!/usr/bin/env bash
set -euo pipefail
cd /root/adablock
out=results/value_aware_selector_20261116/cobs_chunk8192_full100_gpu
py=/root/miniconda3/envs/ada/bin/python
model=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb
mkdir -p "$out"
date -u '+started %Y-%m-%d %H:%M:%S UTC'
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py run \
    --model "$model" --methods cobs --fast-topk --native-gqa --graph-score --digest-chunk 8192 \
    --limit 100 --audit-count 0 --profile-count 10 --num-shards 4 --shard-id "$shard" \
    --output "$out" > "$out/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
for shard in 0 1 2 3; do
  for attempt in 1 2 3; do
    if CUDA_VISIBLE_DEVICES="$shard" "$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py run \
      --model "$model" --methods cobs --fast-topk --native-gqa --graph-score --digest-chunk 8192 \
      --limit 100 --audit-count 0 --profile-count 10 --num-shards 4 --shard-id "$shard" \
      --output "$out" >> "$out/shard_${shard}.log" 2>&1; then break; fi
    if [[ "$attempt" -eq 3 ]]; then echo "shard $shard failed three attempts" >&2; exit 1; fi
    sleep 20
  done
done
"$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py summarize \
  --model "$model" --methods cobs --fast-topk --native-gqa --graph-score --digest-chunk 8192 \
  --limit 100 --audit-count 0 --profile-count 10 --output "$out"
"$py" - <<'PY'
import json,statistics
from pathlib import Path
root=Path('results/value_aware_selector_20261116')
new=root/'cobs_chunk8192_full100_gpu';old=root/'cobs_persistent_full100_gpu'
a=[json.loads((new/f'case_{i:03d}.json').read_text())['methods']['cobs'] for i in range(100)]
b=[json.loads((old/f'case_{i:03d}.json').read_text())['methods']['cobs'] for i in range(100)]
diff=[i for i,(x,y) in enumerate(zip(a,b)) if x['ids']!=y['ids']]
warm=[x['index_build_seconds'] for i,x in enumerate(a) if i>3]
out={'n':100,'changed_ids':diff,'warm_index_median_seconds':statistics.median(warm),
     'warm_index_mean_seconds':statistics.mean(warm),
     'cold_capture_mean_seconds':statistics.mean(a[i]['index_build_seconds'] for i in range(4)),
     'score':statistics.mean(x['ruler_score'] for x in a),'all_hit':sum(x['all_hit'] for x in a)}
(new/'paired_original_graph.json').write_text(json.dumps(out,indent=2))
(new/'COMPLETE.json').write_text(json.dumps({'status':'complete','cases':100,'paired_original_graph':out},indent=2))
print(json.dumps(out,indent=2))
PY
date -u '+complete %Y-%m-%d %H:%M:%S UTC'
