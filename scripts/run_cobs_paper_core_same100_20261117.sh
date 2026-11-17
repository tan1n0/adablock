#!/usr/bin/env bash
# Resume-safe, single-GPU paper-core COBS comparison on the fixed 100 RULER cases.
set -euo pipefail
cd /root/adablock
py=/root/miniconda3/envs/ada/bin/python
model=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb
root=results/cobs_paper_core_same100_20261117
out="$root/full100/cobs"
selector="$root/checkpoint"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="${COBS_GPU:-0}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$py" experiments/one_off/prepare_cobs_paper_core_20261117.py --output "$selector"
common=(--model "$model" --selector-dir "$selector" --methods cobs
        --fast-topk --native-gqa --graph-score --digest-chunk 8192
        --budget 512 --audit-count 0 --profile-count 100 --output "$out")
"$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py check "${common[@]}" --limit 100
if ! nvidia-smi -L | grep -q 'GPU'; then
  echo 'No CUDA GPU visible; inputs and checkpoint are ready, evaluation has not started.' >&2
  exit 2
fi

for block in {0..9}; do
  start=$((block*10))
  end=$((start+10))
  complete=1
  for ((i=start;i<end;i++)); do
    printf -v case_file '%s/case_%03d.json' "$out" "$i"
    if [[ ! -s "$case_file" ]]; then complete=0; break; fi
  done
  if [[ "$complete" -eq 1 ]]; then
    echo "skip completed cases $start-$((end-1))"
    continue
  fi
  for attempt in 1 2 3; do
    if "$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py run \
      "${common[@]}" --start-index "$start" --limit "$end" >> "$out/run.log" 2>&1; then
      break
    fi
    if [[ "$attempt" -eq 3 ]]; then
      echo "FAILED cases $start-$((end-1)); inspect $out/run.log" >&2
      exit 1
    fi
    sleep 20
  done
  echo "complete cases $start-$((end-1))"
done

"$py" experiments/one_off/cobs_persistent_graph_eval_20261116.py summarize \
  "${common[@]}" --limit 100 > "$out/summary_stdout.json"
"$py" experiments/one_off/analyze_cobs_paper_core_same100_20261117.py \
  --fixed "$out" --matched results/matched_cobs_jssa_full_20261117/full100 \
  --output "$root/full100/paired_summary.json"
echo "complete $root/full100/paired_summary.json"
