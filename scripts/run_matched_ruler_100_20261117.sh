#!/usr/bin/env bash
# One entry point for the seven-method, same-100-prompt RULER comparison.
# Full/JSSA/rank-4 COBS are alternated; Oracle, legacy MLP, and fixed COBS
# retain their original execution protocols and result directories.
set -euo pipefail
cd /root/adablock

py=/root/miniconda3/envs/ada/bin/python
model=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb
matched=results/matched_cobs_jssa_full_20261117/full100
oracle=results/value_aware_selector_20261116/oracle_pilot10_gpu
mlp=results/value_aware_selector_20261116/selector_full_100_gpu
fixed=results/cobs_paper_core_same100_20261117/full100
report=results/matched_ruler_100_20261117

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

usage() {
  cat <<'EOF'
Usage: bash scripts/run_matched_ruler_100_20261117.sh [--check|--summarize]

No option: resume incomplete evaluations, then write the combined summary.
--check: print input and result status without loading the model.
--summarize: combine existing complete results without loading the model.

Uses GPU 0, /root/miniconda3/envs/ada/bin/python, and the local Llama model.
Existing completed case files are preserved. The seven methods share the
same 100 prompts, but only Full/JSSA/rank-4 COBS alternate by 10-case block.
EOF
}

mode=run
case "${1:-}" in
  '') ;;
  --check) mode=check ;;
  --summarize) mode=summarize ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

stage_status() {
  "$py" - "$1" "$matched" "$oracle" "$mlp" "$fixed" <<'PY'
import json
import sys
from pathlib import Path

stage, matched, oracle, mlp, fixed = sys.argv[1:]
matched, oracle, mlp, fixed = map(Path, (matched, oracle, mlp, fixed))
if stage == 'matched':
    folders = [(matched / name, (name == 'cobs' and 'cobs' or None))
               for name in ('full', 'jssa_512', 'cobs')]
    summary = matched / 'summary.json'
elif stage == 'oracle':
    folders = [(oracle, ('mass', 'value'))]
    summary = oracle / 'summary_100.json'
elif stage == 'mlp':
    folders = [(mlp, ('mlp', 'cobs'))]
    summary = mlp / 'summary_100.json'
elif stage == 'fixed':
    folders = [(fixed / 'cobs', ('cobs',))]
    summary = fixed / 'paired_summary.json'
else:
    raise ValueError(stage)
for folder, required in folders:
    for i in range(100):
        path = folder / f'case_{i:03d}.json'
        if not path.is_file():
            print(f'incomplete: {path}')
            raise SystemExit(1)
        if required:
            try:
                methods = json.loads(path.read_text())['methods']
            except (ValueError, KeyError) as exc:
                print(f'invalid: {path}: {exc}')
                raise SystemExit(1)
            if any(name not in methods for name in ((required,) if isinstance(required, str) else required)):
                print(f'incomplete methods: {path}')
                raise SystemExit(1)
if not summary.is_file():
    print(f'missing summary: {summary}')
    raise SystemExit(1)
print(f'complete: {stage} (100 cases)')
PY
}

input_status() {
  local missing=0
  local required=(
    "$py"
    "$model/config.json"
    experiments/runs/adablock_ruler_3090_free_budget_20261113_full_01/selected.json
    experiments/runs/adablock_jssa_20261023_01/basis.pt
    results/s_no_fixed_20261116/ranker/best.pt
    results/value_aware_selector_20261116/pilot_cpu_1536/cobs_best.pt
    experiments/runs/adablock_ruler_harder_20261022_01/RULER/scripts/synthetic.yaml
    experiments/server_snapshots/ruler_harder_20261022_01/official.py
  )
  for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "missing input: $path" >&2
      missing=1
    fi
  done
  return "$missing"
}

summarize_all() {
  mkdir -p "$report"
  "$py" experiments/one_off/summarize_matched_ruler_100_20261117.py \
    --matched "$matched" --oracle "$oracle" --mlp "$mlp" --fixed "$fixed" \
    --output "$report"
}

if [[ "$mode" == check ]]; then
  if ! input_status; then exit 1; fi
  echo 'all required local inputs are present'
  for stage in matched oracle mlp fixed; do stage_status "$stage" || true; done
  exit 0
fi

if [[ "$mode" == summarize ]]; then
  for stage in matched oracle mlp fixed; do stage_status "$stage"; done
  summarize_all
  exit 0
fi

needs_gpu=0
for stage in matched oracle mlp fixed; do
  if ! stage_status "$stage" >/dev/null; then needs_gpu=1; fi
done
if (( needs_gpu )); then
  input_status
  if ! nvidia-smi -L | grep -q '^GPU 0:'; then
    echo 'GPU 0 is unavailable; completed results were left untouched.' >&2
    exit 2
  fi
fi

if ! stage_status matched >/dev/null; then
  bash scripts/run_matched_cobs_jssa_full_20261117.sh
fi

if ! stage_status oracle >/dev/null; then
  mkdir -p "$oracle"
  for attempt in 1 2 3; do
    if "$py" experiments/one_off/value_aware_free_eval_20261116.py run \
      --model "$model" --limit 100 --max-swaps 4 --candidate-pool 256 \
      --output "$oracle" >> "$oracle/run.log" 2>&1; then break; fi
    if (( attempt == 3 )); then echo "Oracle failed; see $oracle/run.log" >&2; exit 1; fi
    sleep 20
  done
  "$py" experiments/one_off/value_aware_free_eval_20261116.py summarize \
    --limit 100 --output "$oracle"
fi

if ! stage_status mlp >/dev/null; then
  mkdir -p "$mlp"
  for attempt in 1 2 3; do
    if "$py" experiments/one_off/selector_free_eval_20261116.py run \
      --model "$model" --methods mlp cobs --budget 512 --limit 100 \
      --audit-count 20 --profile-count 10 --output "$mlp" >> "$mlp/run.log" 2>&1; then break; fi
    if (( attempt == 3 )); then echo "MLP failed; see $mlp/run.log" >&2; exit 1; fi
    sleep 20
  done
  "$py" experiments/one_off/selector_free_eval_20261116.py summarize \
    --methods mlp cobs --budget 512 --limit 100 --output "$mlp"
fi

if ! stage_status fixed >/dev/null; then
  COBS_GPU=0 bash scripts/run_cobs_paper_core_same100_20261117.sh
fi

for stage in matched oracle mlp fixed; do stage_status "$stage"; done
summarize_all
