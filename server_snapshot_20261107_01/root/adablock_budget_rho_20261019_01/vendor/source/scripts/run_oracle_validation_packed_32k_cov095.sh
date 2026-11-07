#!/usr/bin/env bash
set -euo pipefail

# Build the validation counterpart of the 95% packed 32K causal Oracle.
# Parameters intentionally match run_oracle_train_packed_32k_cov095.sh.

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSONL="${INPUT_JSONL:-data/validation_strict_packed_tokens.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/oracle_validation_packed_32k_causal_cov095.jsonl}"
LOG_FILE="${LOG_FILE:-logs/build_oracle_validation_packed_32k_causal_cov095.log}"
STATS_JSON="${STATS_JSON:-results/oracle_validation_packed_32k_causal_cov095_length_stats.json}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${LOG_FILE}")" "$(dirname "${STATS_JSON}")"

command=(
  "${PYTHON_BIN}" -u scripts/build_oracle_dataset.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --model-name Qwen/Qwen2.5-7B-Instruct
  --max-length 32768
  --block-size 16
  --sink-tokens 32
  --local-window-blocks 32
  --budget-buckets 0 4 8 16 32
  --coverage 0.95
  --reuse-coverage 0.9
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

printf 'COMMAND: '
printf '%q ' "${command[@]}"
printf '\n\n'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${command[@]}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" scripts/summarize_oracle_budget_by_length.py \
  --input-jsonl "${OUTPUT_JSONL}" \
  --expected-coverage 0.95 \
  --budget-buckets 0 4 8 16 32 \
  --output-json "${STATS_JSON}"
