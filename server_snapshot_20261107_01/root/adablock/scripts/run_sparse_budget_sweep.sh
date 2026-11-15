#!/usr/bin/env bash
set -euo pipefail

# LongBench sparse budget sweep runner.
# Usage:
#   bash scripts/run_sparse_budget_sweep.sh
#
# Edit the variables below before running if needed.

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
POLICY_CHECKPOINT="checkpoints/policy_mlp/best.pt"
TASKS=("hotpotqa" "musique")
LOCAL_DATA_DIR="data/longbench"
MAX_INPUT_LENGTH=2048
MAX_SAMPLES=20
BLOCK_SIZE=16
SINK_TOKENS=32
LOCAL_WINDOW_BLOCKS=4
SAMPLE_STRIDE=64
MAX_NEW_TOKENS=32
DTYPE="float32"
ATTN_IMPL="sdpa"
DEVICE_MAP="auto"

# Choose which methods to run.
# Supported: full, score_topk, quest_topk, policy
METHODS=("full" "score_topk" "quest_topk" "policy")

# Budgets are ignored for METHOD=full.
BUDGETS=(1024 768 512 384 256)

LOG_DIR="logs"
RESULT_ROOT="results/budget_sweep_2048_s20"

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"

TASKS_JOINED="${TASKS[*]}"

run_full() {
  local out_dir="${RESULT_ROOT}/full"
  local log_file="${LOG_DIR}/run_full_2048_s20.log"
  {
    echo "COMMAND:"
    echo "TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py --model-name ${MODEL_NAME} --tasks ${TASKS_JOINED} --local-data-dir ${LOCAL_DATA_DIR} --output-dir ${out_dir} --max-input-length ${MAX_INPUT_LENGTH} --max-samples ${MAX_SAMPLES} --max-new-tokens-override ${MAX_NEW_TOKENS} --dtype ${DTYPE} --device-map ${DEVICE_MAP}"
    echo
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens-override "${MAX_NEW_TOKENS}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  } 2>&1 | tee "${log_file}"
}

run_sparse() {
  local method="$1"
  local budget="$2"
  local out_dir="${RESULT_ROOT}/${method}_b${budget}"
  local log_file="${LOG_DIR}/run_${method}_b${budget}_2048_s20.log"

  {
    echo "COMMAND:"
    echo "TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py --policy-checkpoint ${POLICY_CHECKPOINT} --model-name ${MODEL_NAME} --tasks ${TASKS_JOINED} --local-data-dir ${LOCAL_DATA_DIR} --output-dir ${out_dir} --max-input-length ${MAX_INPUT_LENGTH} --max-samples ${MAX_SAMPLES} --block-size ${BLOCK_SIZE} --local-window-blocks ${LOCAL_WINDOW_BLOCKS} --sample-stride ${SAMPLE_STRIDE} --budget-tokens ${budget} --selection-mode ${method} --fill-budget-with-score --max-new-tokens-override ${MAX_NEW_TOKENS} --dtype ${DTYPE} --attn-implementation ${ATTN_IMPL} --device-map ${DEVICE_MAP}"
    echo
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      --policy-checkpoint "${POLICY_CHECKPOINT}" \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      --max-samples "${MAX_SAMPLES}" \
      --block-size "${BLOCK_SIZE}" \
      --sink-tokens "${SINK_TOKENS}" \
      --local-window-blocks "${LOCAL_WINDOW_BLOCKS}" \
      --sample-stride "${SAMPLE_STRIDE}" \
      --budget-tokens "${budget}" \
      --selection-mode "${method}" \
      --fill-budget-with-score \
      --max-new-tokens-override "${MAX_NEW_TOKENS}" \
      --dtype "${DTYPE}" \
      --attn-implementation "${ATTN_IMPL}" \
      --device-map "${DEVICE_MAP}"
  } 2>&1 | tee "${log_file}"
}

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "full" ]]; then
    run_full
    continue
  fi

  for budget in "${BUDGETS[@]}"; do
    run_sparse "${method}" "${budget}"
  done
done

echo
echo "Finished budget sweep."
echo "Results root: ${RESULT_ROOT}"
