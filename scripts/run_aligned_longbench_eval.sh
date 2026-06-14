#!/usr/bin/env bash
set -euo pipefail

# Run full / policy / score_topk / quest_topk under one aligned LongBench setup.
#
# Defaults target a quick paper-aligned check:
#   model: Qwen/Qwen2.5-7B-Instruct
#   tasks: shorter LongBench long-input tasks first
#   budget: 512 retrieved tokens for sparse methods
#   sink/local retained tokens: S=32, W=512, matching LouisKV long-input settings
#   max input length: 32768
#   max new tokens: per-task defaults from utils/longbench_eval.py unless overridden
#
# Example:
#   bash scripts/run_aligned_longbench_eval.sh
#
# Presets:
#   quick: hotpotqa musique multifieldqa_en
#   tiny: hotpotqa musique
#   paper: narrativeqa qasper multifieldqa_en hotpotqa musique gov_report
#
# Common overrides:
#   PRESET=paper bash scripts/run_aligned_longbench_eval.sh
#   MAX_INPUT_LENGTH=8192 MAX_SAMPLES=20 RESULT_ROOT=results/aligned_debug \
#     bash scripts/run_aligned_longbench_eval.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-checkpoints/adablock_policy_longbench_qa_1024/best.pt}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-data/longbench}"

PRESET="${PRESET:-quick}"
if [[ -n "${TASKS:-}" ]]; then
  TASKS=(${TASKS})
else
  case "${PRESET}" in
    tiny)
      TASKS=(hotpotqa musique)
      ;;
    quick)
      TASKS=(hotpotqa musique multifieldqa_en)
      ;;
    paper)
      TASKS=(narrativeqa qasper multifieldqa_en hotpotqa musique gov_report)
      ;;
    *)
      echo "Unknown PRESET=${PRESET}. Use tiny, quick, paper, or set TASKS explicitly." >&2
      exit 2
      ;;
  esac
fi
METHODS=(${METHODS:-full policy score_topk quest_topk})

MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-32768}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"

BUDGET_TOKENS="${BUDGET_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SINK_TOKENS="${SINK_TOKENS:-32}"
LOCAL_WINDOW_TOKENS="${LOCAL_WINDOW_TOKENS:-512}"
LOCAL_WINDOW_BLOCKS="${LOCAL_WINDOW_BLOCKS:-$(((LOCAL_WINDOW_TOKENS + BLOCK_SIZE - 1) / BLOCK_SIZE))}"
REQUIRED_BLOCKS_OUTSIDE_BUDGET="${REQUIRED_BLOCKS_OUTSIDE_BUDGET:-1}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-64}"
REUSE_THRESHOLD="${REUSE_THRESHOLD:-0.8}"
FILL_BUDGET_WITH_SCORE="${FILL_BUDGET_WITH_SCORE:-1}"

DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

RESULT_ROOT="${RESULT_ROOT:-results/aligned_longbench_${PRESET}_${MAX_INPUT_LENGTH}_b${BUDGET_TOKENS}}"
LOG_DIR="${LOG_DIR:-logs/aligned_longbench_${PRESET}_${MAX_INPUT_LENGTH}_b${BUDGET_TOKENS}}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

optional_common_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  optional_common_args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NEW_TOKENS_OVERRIDE}" ]]; then
  optional_common_args+=(--max-new-tokens-override "${MAX_NEW_TOKENS_OVERRIDE}")
fi

sparse_fill_args=()
if [[ "${FILL_BUDGET_WITH_SCORE}" == "1" ]]; then
  sparse_fill_args+=(--fill-budget-with-score)
fi

required_budget_args=()
if [[ "${REQUIRED_BLOCKS_OUTSIDE_BUDGET}" == "1" ]]; then
  required_budget_args+=(--required-blocks-outside-budget)
fi

run_full() {
  local out_dir="${RESULT_ROOT}/full"
  local log_file="${LOG_DIR}/full.log"
  {
    echo "COMMAND:"
    printf '%q ' TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      "${optional_common_args[@]}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
    echo
    echo
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      "${optional_common_args[@]}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  } 2>&1 | tee "${log_file}"
}

run_sparse() {
  local method="$1"
  local out_dir="${RESULT_ROOT}/${method}_b${BUDGET_TOKENS}"
  local log_file="${LOG_DIR}/${method}_b${BUDGET_TOKENS}.log"
  local policy_args=()
  if [[ "${method}" == "policy" ]]; then
    policy_args+=(--policy-checkpoint "${POLICY_CHECKPOINT}")
  fi

  {
    echo "COMMAND:"
    printf '%q ' TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${policy_args[@]}" \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      "${optional_common_args[@]}" \
      --block-size "${BLOCK_SIZE}" \
      --local-window-blocks "${LOCAL_WINDOW_BLOCKS}" \
      --sink-tokens "${SINK_TOKENS}" \
      "${required_budget_args[@]}" \
      --sample-stride "${SAMPLE_STRIDE}" \
      --budget-tokens "${BUDGET_TOKENS}" \
      --reuse-threshold "${REUSE_THRESHOLD}" \
      --selection-mode "${method}" \
      "${sparse_fill_args[@]}" \
      --dtype "${DTYPE}" \
      --attn-implementation "${ATTN_IMPL}" \
      --device-map "${DEVICE_MAP}"
    echo
    echo
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${policy_args[@]}" \
      --model-name "${MODEL_NAME}" \
      --tasks "${TASKS[@]}" \
      --local-data-dir "${LOCAL_DATA_DIR}" \
      --output-dir "${out_dir}" \
      --max-input-length "${MAX_INPUT_LENGTH}" \
      "${optional_common_args[@]}" \
      --block-size "${BLOCK_SIZE}" \
      --local-window-blocks "${LOCAL_WINDOW_BLOCKS}" \
      --sink-tokens "${SINK_TOKENS}" \
      "${required_budget_args[@]}" \
      --sample-stride "${SAMPLE_STRIDE}" \
      --budget-tokens "${BUDGET_TOKENS}" \
      --reuse-threshold "${REUSE_THRESHOLD}" \
      --selection-mode "${method}" \
      "${sparse_fill_args[@]}" \
      --dtype "${DTYPE}" \
      --attn-implementation "${ATTN_IMPL}" \
      --device-map "${DEVICE_MAP}"
  } 2>&1 | tee "${log_file}"
}

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "full" ]]; then
    run_full
  else
    run_sparse "${method}"
  fi
done

python scripts/summarize_aligned_longbench.py \
  --result-root "${RESULT_ROOT}" \
  --target-profile louiskv-qwen25-512 \
  --output-json "${RESULT_ROOT}/aligned_summary.json" \
  --output-tsv "${RESULT_ROOT}/aligned_summary.tsv"

echo
echo "Finished aligned LongBench evaluation."
echo "Results root: ${RESULT_ROOT}"
echo "Logs: ${LOG_DIR}"
