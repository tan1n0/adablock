#!/usr/bin/env bash
set -euo pipefail
cd /root/adablock
py=/root/miniconda3/envs/ada/bin/python
model=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb
root=results/matched_cobs_jssa_full_20261117/full100
mkdir -p "$root"/{full,jssa_512,cobs}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
date -u '+matched run started %Y-%m-%d %H:%M:%S UTC'
for block in 0 1 2 3 4 5 6 7 8 9; do
  start=$((block*10))
  end=$((start+10))
  case "$((block%3))" in
    0) order=(full jssa_512 cobs);;
    1) order=(jssa_512 cobs full);;
    2) order=(cobs full jssa_512);;
  esac
  for method in "${order[@]}"; do
    date -u "+block=$block cases=$start-$((end-1)) method=$method start %Y-%m-%d %H:%M:%S UTC"
    for attempt in 1 2 3; do
      if [[ "$method" == cobs ]]; then
        if "$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py run \
          --model "$model" --methods cobs --fast-topk --native-gqa --graph-score \
          --digest-chunk 8192 --start-index "$start" --limit "$end" \
          --audit-count 0 --profile-count 100 --output "$root/cobs" \
          >> "$root/cobs/run.log" 2>&1; then break; fi
      else
        if "$py" experiments/one_off/matched_backend_eval_20261117.py run \
          --method "$method" --model "$model" --start-index "$start" --limit "$end" \
          --output "$root/$method" >> "$root/$method/run.log" 2>&1; then break; fi
      fi
      if [[ "$attempt" -eq 3 ]]; then echo "FAILED block=$block method=$method" >&2; exit 1; fi
      sleep 20
    done
    date -u "+block=$block method=$method complete %Y-%m-%d %H:%M:%S UTC"
  done
done
"$py" experiments/one_off/matched_backend_eval_20261117.py summarize \
  --method full --model "$model" --limit 100 --output "$root/full"
"$py" experiments/one_off/matched_backend_eval_20261117.py summarize \
  --method jssa_512 --model "$model" --limit 100 --output "$root/jssa_512"
"$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py summarize \
  --model "$model" --methods cobs --fast-topk --native-gqa --graph-score \
  --digest-chunk 8192 --limit 100 --audit-count 0 --profile-count 100 --output "$root/cobs"
"$py" experiments/one_off/analyze_matched_cobs_jssa_full_20261117.py --root "$root"
date -u '+matched run complete %Y-%m-%d %H:%M:%S UTC'
