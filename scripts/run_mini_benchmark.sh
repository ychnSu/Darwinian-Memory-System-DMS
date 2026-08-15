#!/usr/bin/env bash
set -euo pipefail

AGENT_NAME="${1:-zero_shot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HOME/android_world_results/mini_benchmark_${AGENT_NAME}_${STAMP}}"

mkdir -p "$OUT"

python -u run.py \
  --agent_name="$AGENT_NAME" \
  --benchmark_config=configs/mini_benchmark.json \
  --output_path="$OUT" \
  --results_csv="$OUT/results.csv" \
  2>&1 | tee "$OUT/run.log"
