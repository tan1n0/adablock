#!/usr/bin/env bash
set -euo pipefail

# Quick LongBench quality-vs-number-of-blocks sweep.
#
# The sparse points are the number of *dynamic middle blocks*.  sink/local
# blocks remain fixed, as they do in eval_longbench_adablock_sparse.py.  A
# separate full-attention run is added as the final point.
#
# Example (a tiny, local smoke run):
#   MAX_SAMPLES=3 MAX_INPUT_LENGTH=4096 TASKS="hotpotqa musique" \
#     bash scripts/run_longbench_block_trend.sh
#
# Useful overrides:
#   METHOD=policy POLICY_CHECKPOINT=...   # default is deterministic score_topk
#   BLOCK_POINTS="0 4 8 16 32 64"        # explicit sparse points
#   MAX_SAMPLES=10 DTYPE=bfloat16 DEVICE_MAP=auto

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-checkpoints/policy_mlp_packed_32k_causal/best.pt}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-data/longbench}"
TASKS=(${TASKS:-hotpotqa musique})
METHOD="${METHOD:-score_topk}"

MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-32768}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SINK_TOKENS="${SINK_TOKENS:-32}"
LOCAL_WINDOW_BLOCKS="${LOCAL_WINDOW_BLOCKS:-4}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-64}"
REUSE_THRESHOLD="${REUSE_THRESHOLD:-0.8}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESULT_ROOT="${RESULT_ROOT:-results/longbench_block_trend_${MAX_INPUT_LENGTH}_s${MAX_SAMPLES}}"
LOG_DIR="${LOG_DIR:-logs/longbench_block_trend_${MAX_INPUT_LENGTH}_s${MAX_SAMPLES}}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${METHOD}" != "score_topk" && "${METHOD}" != "quest_topk" && "${METHOD}" != "arkvale_topk" && "${METHOD}" != "policy" ]]; then
  echo "METHOD must be score_topk, quest_topk, arkvale_topk, or policy (got ${METHOD})" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

# Each sparse budget is expressed in total blocks by the evaluator.  Reserve
# the maximum fixed-block footprint so that a requested point means roughly
# the requested number of dynamic middle blocks (at short prefixes fewer may
# be available).  If BLOCK_POINTS is omitted, scan powers of two to the
# largest block count implied by MAX_INPUT_LENGTH.
fixed_blocks=$(( (SINK_TOKENS + BLOCK_SIZE - 1) / BLOCK_SIZE + LOCAL_WINDOW_BLOCKS ))
max_blocks=$(( (MAX_INPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE ))
if [[ -n "${BLOCK_POINTS:-}" ]]; then
  read -r -a points <<< "${BLOCK_POINTS}"
else
  points=(0)
  point=4
  while (( point < max_blocks )); do
    points+=("${point}")
    point=$((point * 2))
  done
  points+=("${max_blocks}")
fi

common_args=(
  --model-name "${MODEL_NAME}"
  --tasks "${TASKS[@]}"
  --local-data-dir "${LOCAL_DATA_DIR}"
  --max-input-length "${MAX_INPUT_LENGTH}"
  --max-samples "${MAX_SAMPLES}"
  --block-size "${BLOCK_SIZE}"
  --sink-tokens "${SINK_TOKENS}"
  --local-window-blocks "${LOCAL_WINDOW_BLOCKS}"
  --sample-stride "${SAMPLE_STRIDE}"
  --reuse-threshold "${REUSE_THRESHOLD}"
  --max-new-tokens-override "${MAX_NEW_TOKENS_OVERRIDE}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPL}"
  --device-map "${DEVICE_MAP}"
)

run_sparse() {
  local middle_blocks="$1"
  local total_blocks=$((middle_blocks + fixed_blocks))
  local budget_tokens=$((total_blocks * BLOCK_SIZE))
  local tag="${METHOD}_m${middle_blocks}"
  local out_dir="${RESULT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/${tag}.log"
  local policy_args=()
  if [[ "${METHOD}" == "policy" ]]; then
    policy_args+=(--policy-checkpoint "${POLICY_CHECKPOINT}")
  fi

  {
    printf 'COMMAND: '
    printf '%q ' TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${policy_args[@]}" "${common_args[@]}" \
      --output-dir "${out_dir}" \
      --budget-tokens "${budget_tokens}" \
      --selection-mode "${METHOD}" \
      --fill-budget-with-score
    echo
    if [[ "${DRY_RUN}" == "1" ]]; then
      return
    fi
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_adablock_sparse.py \
      "${policy_args[@]}" "${common_args[@]}" \
      --output-dir "${out_dir}" \
      --budget-tokens "${budget_tokens}" \
      --selection-mode "${METHOD}" \
      --fill-budget-with-score
  } 2>&1 | tee "${log_file}"
}

run_full() {
  local out_dir="${RESULT_ROOT}/full"
  local log_file="${LOG_DIR}/full.log"
  local full_args=(
    --model-name "${MODEL_NAME}"
    --tasks "${TASKS[@]}"
    --local-data-dir "${LOCAL_DATA_DIR}"
    --output-dir "${out_dir}"
    --max-input-length "${MAX_INPUT_LENGTH}"
    --max-samples "${MAX_SAMPLES}"
    --max-new-tokens-override "${MAX_NEW_TOKENS_OVERRIDE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  {
    printf 'COMMAND: '
    printf '%q ' TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py "${full_args[@]}"
    echo
    if [[ "${DRY_RUN}" == "1" ]]; then
      return
    fi
    TRANSFORMERS_VERBOSITY=error python scripts/eval_longbench_full.py "${full_args[@]}"
  } 2>&1 | tee "${log_file}"
}

echo "Sparse points (dynamic middle blocks): ${points[*]}"
echo "Fixed-block reservation: ${fixed_blocks} blocks; full-attention is a separate baseline."
for middle_blocks in "${points[@]}"; do
  if ! [[ "${middle_blocks}" =~ ^[0-9]+$ ]]; then
    echo "Invalid block point: ${middle_blocks}" >&2
    exit 2
  fi
  run_sparse "${middle_blocks}"
done
run_full

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: no model evaluation or summary was executed."
  exit 0
fi

python scripts/summarize_aligned_longbench.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/trend_summary.json" \
  --output-tsv "${RESULT_ROOT}/trend_summary.tsv"

echo
echo "Finished LongBench block trend sweep."
echo "Results: ${RESULT_ROOT}"
echo "Summary: ${RESULT_ROOT}/trend_summary.tsv"
