#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSONL="${INPUT_JSONL:-data/validation_strict_packed_tokens.jsonl}" \
OUTPUT_JSONL="${OUTPUT_JSONL:-data/oracle_validation_packed_32k_qk_multimethod.jsonl}" \
LOG_FILE="${LOG_FILE:-logs/build_oracle_validation_packed_32k_qk_multimethod.log}" \
STATS_DIR="${STATS_DIR:-results/oracle_validation_packed_32k_qk_multimethod_stats}" \
  bash "${SCRIPT_DIR}/run_oracle_train_packed_32k_qk_multimethod.sh"
