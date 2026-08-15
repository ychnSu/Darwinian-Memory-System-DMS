#!/usr/bin/env bash
# Run Baseline B (static memory) on the configured mini-benchmark.
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG="${CONFIG:-configs/mini_benchmark.json}"
OUT="${OUT:-$HOME/android_world_results/baseline_b_mini_benchmark_${STAMP}}"
MEMORY_DIR="${MEMORY_DIR:-$OUT/memory}"
TRANSITION_PAUSE="${TRANSITION_PAUSE:-1.25}"
RESULTS_CSV="${RESULTS_CSV:-$OUT/results.csv}"

mkdir -p "$OUT" "$MEMORY_DIR"

CMD=(
  python -u run.py
  --agent_name=static_memory
  --benchmark_config="$CONFIG"
  --dms_memory_dir="$MEMORY_DIR"
  --dms_transition_pause="$TRANSITION_PAUSE"
  --output_path="$OUT"
  --results_csv="$RESULTS_CSV"
)

if [[ -n "${TASKS:-}" ]]; then
  CMD+=(--tasks="$TASKS")
fi

echo "Baseline B mini-benchmark"
echo "Config: $CONFIG"
echo "Output: $OUT"
echo "Results CSV: $RESULTS_CSV"
echo "Run log: $OUT/run.log"
echo "Static memory: $MEMORY_DIR/static_memory.json"
echo "Transition pause: $TRANSITION_PAUSE"
if [[ -n "${TASKS:-}" ]]; then
  echo "Task override: $TASKS"
fi
echo

"${CMD[@]}" 2>&1 | tee "$OUT/run.log"

echo "Baseline B mini-benchmark finished."
echo "Output: $OUT"
echo "Results CSV: $RESULTS_CSV"
echo "Run log: $OUT/run.log"
echo "Static memory: $MEMORY_DIR/static_memory.json"
if [[ -f "$OUT/summary.json" ]]; then
  echo "Summary JSON: $OUT/summary.json"
fi
