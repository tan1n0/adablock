#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-data/strict_sources}"
CACHE_DIR="${CACHE_DIR:-.cache/adablock_training_sources}"
LONGBENCH_DIR="${LONGBENCH_DIR:-data/longbench}"
SEED="${SEED:-42}"

mkdir -p "${SOURCE_DIR}"

build_source() {
  local name="$1"
  shift
  python -u scripts/build_strict_training_corpus.py     --cache-dir "${CACHE_DIR}"     --longbench-dir "${LONGBENCH_DIR}"     --seed "${SEED}"     --train-output "${SOURCE_DIR}/${name}_train.jsonl"     --validation-output "${SOURCE_DIR}/${name}_validation.jsonl"     --metadata-output "${SOURCE_DIR}/${name}_metadata.json"     "$@"
}

build_source pg19   --pg19-count 750 --booksum-count 0 --arxiv-count 0   --pubmed-count 0 --wikihop-count 0 --multi-needle-count 0

build_source booksum   --pg19-count 0 --booksum-count 750 --arxiv-count 0   --pubmed-count 0 --wikihop-count 0 --multi-needle-count 0

build_source arxiv   --pg19-count 0 --booksum-count 0 --arxiv-count 1000   --pubmed-count 0 --wikihop-count 0 --multi-needle-count 0

build_source pubmed   --pg19-count 0 --booksum-count 0 --arxiv-count 0   --pubmed-count 1000 --wikihop-count 0 --multi-needle-count 0

build_source wikihop   --pg19-count 0 --booksum-count 0 --arxiv-count 0   --pubmed-count 0 --wikihop-count 1000 --multi-needle-count 0

build_source multi_needle   --pg19-count 0 --booksum-count 0 --arxiv-count 0   --pubmed-count 0 --wikihop-count 0 --multi-needle-count 1500

python -u scripts/merge_strict_training_corpus.py   --source-dir "${SOURCE_DIR}"   --longbench-dir "${LONGBENCH_DIR}"   --seed "${SEED}"   --train-output data/train_strict_clean.jsonl   --validation-output data/validation_strict_clean.jsonl   --metadata-output data/train_strict_clean_metadata.json
