#!/usr/bin/env bash
set -euo pipefail
cd /root/adablock
out=results/value_aware_selector_20261116/cobs_persistent_full100_gpu
py=/root/miniconda3/envs/ada/bin/python
model=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb
mkdir -p "$out"
date -u '+resume supervisor started %Y-%m-%d %H:%M:%S UTC'
for pid in 25828 25970 26114 26260; do
  while kill -0 "$pid" 2>/dev/null; do sleep 15; done
done
date -u '+original shards finished or disconnected %Y-%m-%d %H:%M:%S UTC'
for shard in 0 1 2 3; do
  for attempt in 1 2 3; do
    if CUDA_VISIBLE_DEVICES="$shard" "$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py run \
      --model "$model" --methods cobs --fast-topk --native-gqa --graph-score \
      --limit 100 --audit-count 0 --profile-count 10 --num-shards 4 --shard-id "$shard" \
      --output "$out"; then break; fi
    if [[ "$attempt" -eq 3 ]]; then echo "shard $shard failed three attempts" >&2; exit 1; fi
    sleep 20
  done
done
"$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py summarize \
  --limit 100 --methods cobs --fast-topk --native-gqa --graph-score --output "$out"
"$py" - <<'PY'
from pathlib import Path
import json
p=Path('results/value_aware_selector_20261116/cobs_persistent_full100_gpu')
rows=[json.loads((p/f'case_{i:03d}.json').read_text()) for i in range(100)]
assert all('cobs' in row['methods'] for row in rows)
(p/'COMPLETE.json').write_text(json.dumps({'cases':100,'status':'complete','summary':'summary_100.json'},indent=2))
PY
date -u '+resume supervisor complete %Y-%m-%d %H:%M:%S UTC'
