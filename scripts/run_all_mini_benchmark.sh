#!/usr/bin/env bash
# Run Baseline A, Baseline B, and DMS on the mini-benchmark in order.
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/android_world_results/all_mini_benchmark_${STAMP}}"

mkdir -p "$OUT_ROOT"
exec > >(tee "$OUT_ROOT/run_all.log") 2>&1

cd "$ROOT_DIR"

echo "Mini-benchmark full comparison"
echo "Output root: $OUT_ROOT"
echo

echo "== Baseline A: PA-Lite =="
OUT="$OUT_ROOT/baseline_a" bash scripts/run_mini_benchmark.sh zero_shot
echo

echo "== Baseline B: Static Memory =="
OUT="$OUT_ROOT/baseline_b" bash scripts/run_baseline_b_mini_benchmark.sh
echo

echo "== DMS =="
OUT="$OUT_ROOT/dms" \
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}" \
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$HOME/projects/android_world/embedding_model}" \
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}" \
bash scripts/run_dms_mini_benchmark.sh
echo

echo "All mini-benchmark runs finished."
echo "Baseline A: $OUT_ROOT/baseline_a"
echo "Baseline B: $OUT_ROOT/baseline_b"
echo "DMS:        $OUT_ROOT/dms"
echo "Master log: $OUT_ROOT/run_all.log"
