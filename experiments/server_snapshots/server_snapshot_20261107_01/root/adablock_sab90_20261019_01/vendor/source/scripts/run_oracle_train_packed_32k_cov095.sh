#!/usr/bin/env bash
set -euo pipefail

# Rebuild the packed 32K causal Oracle with a 95% attention-mass coverage
# target.  The original 90% Oracle is kept intact.  Re-running this script
# resumes from the last completely written document.

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSONL="${INPUT_JSONL:-data/train_strict_packed_tokens.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/oracle_train_packed_32k_causal_cov095.jsonl}"
LOG_FILE="${LOG_FILE:-logs/build_oracle_train_packed_32k_causal_cov095.log}"
SCORE_SELECTOR="${SCORE_SELECTOR:-cosine}"
STATS_JSON="${STATS_JSON:-results/oracle_train_packed_32k_causal_cov095_length_stats.json}"
BUDGET_BUCKETS="${BUDGET_BUCKETS:-0 4 8 16 32 64 128}"
BUDGET_BUCKET_MAX="${BUDGET_BUCKET_MAX:-}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${LOG_FILE}")"

command=(
  "${PYTHON_BIN}" -u scripts/build_oracle_dataset.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --model-name Qwen/Qwen2.5-7B-Instruct
  --max-length 32768
  --block-size 16
  --sink-tokens 32
  --local-window-blocks 32
  --coverage 0.95
  --reuse-coverage 0.9
  --score-selector "${SCORE_SELECTOR}"
  --sample-stride 64
  --transition-mode causal_next_token
  --anchor-offset-mode document_hash
  --oracle-layer-indices 12 20 27
  --attention-mode sampled_recompute
  --attn-implementation sdpa
  --attention-query-chunk-size 8
  --dtype bfloat16
  --device-map balanced_low_0
  --resume
  --log-every-doc
)

if [[ -n "${BUDGET_BUCKET_MAX}" ]]; then
  command+=(--budget-bucket-max "${BUDGET_BUCKET_MAX}")
else
  command+=(--budget-buckets ${BUDGET_BUCKETS})
fi

printf 'COMMAND: '
printf '%q ' "${command[@]}"
printf '\n\n'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${command[@]}" 2>&1 | tee -a "${LOG_FILE}"

if [[ -n "${BUDGET_BUCKET_MAX}" ]]; then
  # The summary script accepts the same compact max-bucket form.
  summary_command=(
    "${PYTHON_BIN}" scripts/summarize_oracle_budget_by_length.py
    --input-jsonl "${OUTPUT_JSONL}"
    --expected-coverage 0.95
    --budget-bucket-max "${BUDGET_BUCKET_MAX}"
    --output-json "${STATS_JSON}"
  )
else
  summary_command=(
    "${PYTHON_BIN}" scripts/summarize_oracle_budget_by_length.py
    --input-jsonl "${OUTPUT_JSONL}"
    --expected-coverage 0.95
    --budget-buckets ${BUDGET_BUCKETS}
    --output-json "${STATS_JSON}"
  )
fi
"${summary_command[@]}"
