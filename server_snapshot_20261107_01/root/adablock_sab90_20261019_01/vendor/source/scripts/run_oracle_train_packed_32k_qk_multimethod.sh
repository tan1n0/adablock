#!/usr/bin/env bash
set -euo pipefail

# Build one all-layer post-RoPE Q/K Oracle containing targets for all six
# retrieval methods.  Model forward, Q/K projection and sampled full-attention
# recomputation are shared; method-specific statistics are stored under
# budget_targets in each JSONL row.

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ada/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-data/train_strict_packed_tokens.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/oracle_train_packed_32k_qk_multimethod.jsonl}"
LOG_FILE="${LOG_FILE:-logs/build_oracle_train_packed_32k_qk_multimethod.log}"
STATS_DIR="${STATS_DIR:-results/oracle_train_packed_32k_qk_multimethod_stats}"
MAX_DOCS="${MAX_DOCS:-}"
COBS_RANK="${COBS_RANK:-4}"
PHSA_LAMBDA="${PHSA_LAMBDA:-0.5}"

selectors=(mean_pool quest_aabb arkvale_sphere spla cobs phsa)
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${LOG_FILE}")" "${STATS_DIR}"

command=(
  "${PYTHON_BIN}" -u scripts/build_oracle_dataset.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --model-name Qwen/Qwen2.5-7B-Instruct
  --max-length 32768
  --block-size 16
  --sink-tokens 32
  --local-window-blocks 32
  --budget-buckets 0 4 8 16 32 64 128
  --coverage 0.95
  --reuse-coverage 0.9
  --sample-stride 64
  --transition-mode causal_next_token
  --anchor-offset-mode document_hash
  --oracle-layer-indices "${layers[@]}"
  --attention-mode sampled_recompute
  --attn-implementation sdpa
  --attention-query-chunk-size 8
  --retrieval-space post_rope_qk
  --qk-score-selectors "${selectors[@]}"
  --primary-qk-selector mean_pool
  --cobs-rank "${COBS_RANK}"
  --phsa-lambda "${PHSA_LAMBDA}"
  --dtype bfloat16
  --device-map balanced_low_0
  --resume
  --log-every-doc
)

if [[ -n "${MAX_DOCS}" ]]; then
  command+=(--max-docs "${MAX_DOCS}")
fi

printf 'COMMAND: '
printf '%q ' "${command[@]}"
printf '\n\n'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${command[@]}" 2>&1 | tee -a "${LOG_FILE}"

for selector in "${selectors[@]}"; do
  "${PYTHON_BIN}" scripts/summarize_oracle_budget_by_length.py \
    --input-jsonl "${OUTPUT_JSONL}" \
    --target-selector "${selector}" \
    --expected-coverage 0.95 \
    --budget-buckets 0 4 8 16 32 64 128 \
    --output-json "${STATS_DIR}/${selector}.json" \
    | tee "${STATS_DIR}/${selector}.tsv"
done

echo "Finished all-layer Q/K multi-method Oracle."
echo "Dataset: ${OUTPUT_JSONL}"
echo "Statistics: ${STATS_DIR}"
