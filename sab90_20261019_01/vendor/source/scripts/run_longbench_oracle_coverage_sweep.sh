#!/usr/bin/env bash
set -euo pipefail

# Measure LongBench quality loss when an Oracle selector retains the minimum
# number of blocks needed for 95/90/85/80% dense attention-mass coverage.
# The dense pass provides selection signals only; generation state is sparse.

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ada/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-data/longbench}"
TASKS=(${TASKS:-hotpotqa musique multifieldqa_en})
COVERAGE_THRESHOLDS=(${COVERAGE_THRESHOLDS:-0.95 0.90 0.85 0.80})

MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-32768}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SINK_TOKENS="${SINK_TOKENS:-32}"
LOCAL_WINDOW_BLOCKS="${LOCAL_WINDOW_BLOCKS:-32}"
ORACLE_MAX_MIDDLE_BLOCKS="${ORACLE_MAX_MIDDLE_BLOCKS:-}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-balanced_low_0}"
RUN_FULL="${RUN_FULL:-1}"

RESULT_ROOT="${RESULT_ROOT:-results/longbench_oracle_coverage_sweep}"
LOG_DIR="${LOG_DIR:-logs/longbench_oracle_coverage_sweep}"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

optional_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  optional_args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NEW_TOKENS_OVERRIDE}" ]]; then
  optional_args+=(--max-new-tokens-override "${MAX_NEW_TOKENS_OVERRIDE}")
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  full_dir="${RESULT_ROOT}/full"
  if [[ -f "${full_dir}/summary.json" ]]; then
    echo "Full baseline already complete: ${full_dir}/summary.json"
  else
    mkdir -p "${full_dir}"
    "${PYTHON_BIN}" -u scripts/eval_longbench_full.py \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${full_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      "${optional_args[@]}" \
      --dtype "${DTYPE}" \
      --local-files-only \
      --device-map "${DEVICE_MAP}" \
      2>&1 | tee "${LOG_DIR}/full.log"
  fi
fi

for threshold in "${COVERAGE_THRESHOLDS[@]}"; do
  tag="cov${threshold/./}"
  out_dir="${RESULT_ROOT}/${tag}"
  log_file="${LOG_DIR}/${tag}.log"
  command=(
    "${PYTHON_BIN}" -u scripts/eval_longbench_adablock_sparse.py
    --model-name "${MODEL_NAME}"
    --tasks "${TASKS[@]}"
    --local-data-dir "${LOCAL_DATA_DIR}"
    --output-dir "${out_dir}"
    --max-input-length "${MAX_INPUT_LENGTH}"
    --block-size "${BLOCK_SIZE}"
    --sink-tokens "${SINK_TOKENS}"
    --local-window-blocks "${LOCAL_WINDOW_BLOCKS}"
    --selection-mode oracle_coverage
    --attention-coverage-threshold "${threshold}"
    --dtype "${DTYPE}"
    --attn-implementation sdpa
    --local-files-only
    --device-map "${DEVICE_MAP}"
    --resume
    "${optional_args[@]}"
  )
  if [[ -n "${ORACLE_MAX_MIDDLE_BLOCKS}" ]]; then
    command+=(--oracle-max-middle-blocks "${ORACLE_MAX_MIDDLE_BLOCKS}")
  fi

  printf 'COMMAND: '
  printf '%q ' "${command[@]}"
  printf '\n\n'
  "${command[@]}" 2>&1 | tee -a "${log_file}"
done

summary_args=(
  "${PYTHON_BIN}" scripts/summarize_aligned_longbench.py
  --result-root "${RESULT_ROOT}"
  --output-json "${RESULT_ROOT}/coverage_summary.json"
  --output-tsv "${RESULT_ROOT}/coverage_summary.tsv"
)
if [[ "${RUN_FULL}" == "1" || -d "${RESULT_ROOT}/full" ]]; then
  summary_args+=(--reference-method full)
fi
"${summary_args[@]}"

echo "Finished Oracle attention-coverage sweep."
echo "Results: ${RESULT_ROOT}/coverage_summary.tsv"
echo "Logs: ${LOG_DIR}"
