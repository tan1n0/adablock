#!/usr/bin/env bash
set -euo pipefail

# Compare the existing global hidden-centroid selector with current-query,
# per-layer/per-KV-head Q/K centroid selection on exactly the same samples.
# This is an accuracy experiment; the layer/head prototype is not optimized.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-checkpoints/policy_mlp_packed_32k_causal/best.pt}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-data/longbench}"
TASKS=(${TASKS:-hotpotqa})
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-32768}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-32}"
BUDGET_TOKENS="${BUDGET_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SINK_TOKENS="${SINK_TOKENS:-32}"
LOCAL_WINDOW_BLOCKS="${LOCAL_WINDOW_BLOCKS:-4}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESULT_ROOT="${RESULT_ROOT:-results/layer_head_qk_compare_b${BUDGET_TOKENS}_s${MAX_SAMPLES}}"
LOG_DIR="${LOG_DIR:-logs/layer_head_qk_compare_b${BUDGET_TOKENS}_s${MAX_SAMPLES}}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

common_args=(
  --model-name "${MODEL_NAME}"
  --tasks "${TASKS[@]}"
  --local-data-dir "${LOCAL_DATA_DIR}"
  --max-input-length "${MAX_INPUT_LENGTH}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens-override "${MAX_NEW_TOKENS_OVERRIDE}"
  --budget-tokens "${BUDGET_TOKENS}"
  --block-size "${BLOCK_SIZE}"
  --sink-tokens "${SINK_TOKENS}"
  --local-window-blocks "${LOCAL_WINDOW_BLOCKS}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPL}"
  --device-map "${DEVICE_MAP}"
)

run_mode() {
  local tag="$1"
  local mode="$2"
  shift 2
  local out_dir="${RESULT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/${tag}.log"
  {
    printf 'COMMAND: '
    printf '%q ' TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${common_args[@]}" --output-dir "${out_dir}" --selection-mode "${mode}" "$@"
    echo
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${common_args[@]}" --output-dir "${out_dir}" --selection-mode "${mode}" "$@"
  } 2>&1 | tee "${log_file}"
}

# Fixed 512-token comparison: isolates block location/ranking.
run_mode global_fixed score_topk --fill-budget-with-score
run_mode layer_head_fixed layer_head_qk_topk

# Adaptive-budget comparison: same checkpoint and legacy controller inputs.
# A threshold above one disables old-path reuse so both arms reselect each step.
run_mode global_policy policy \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --reuse-threshold 1.1
run_mode layer_head_policy policy_layer_head_qk \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --reuse-threshold 1.1

python scripts/summarize_aligned_longbench.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/comparison.json" \
  --output-tsv "${RESULT_ROOT}/comparison.tsv"

echo "Results: ${RESULT_ROOT}/comparison.tsv"
