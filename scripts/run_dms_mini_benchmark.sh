#!/usr/bin/env bash
# Run DMS on the configured mini-benchmark and generate per-task/overall charts.
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG="${CONFIG:-configs/mini_benchmark.json}"
OUT="${OUT:-$HOME/android_world_results/dms_mini_benchmark_${STAMP}}"
MEMORY_DIR="${MEMORY_DIR:-$OUT/memory}"
RESULTS_CSV="${RESULTS_CSV:-$OUT/results.csv}"
TRANSITION_PAUSE="${TRANSITION_PAUSE:-1.5}"
PRUNE_INTERVAL="${PRUNE_INTERVAL:-5}"
CAPACITY_MIN="${CAPACITY_MIN:-18}"
CAPACITY_MAX="${CAPACITY_MAX:-72}"
CAPACITY_STEP="${CAPACITY_STEP:-2}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$HOME/projects/android_world/embedding_model}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:18001/embed}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:18000/v1}"
VLM_MODEL="${VLM_MODEL:-/root/autodl-tmp/model}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/android_world/bin/python}"

mkdir -p "$OUT" "$MEMORY_DIR"
exec > >(tee "$OUT/run.log") 2>&1

echo "DMS mini-benchmark"
echo "Config: $CONFIG"
echo "Output: $OUT"
echo "Results CSV: $RESULTS_CSV"
echo "Run log: $OUT/run.log"
echo "DMS memory metadata: $MEMORY_DIR/dms_memory.json"
echo "DMS trajectories: $MEMORY_DIR/memory_trajectories/"
echo "Metrics report: $OUT/dms_metrics_report"
echo "Embedding provider: $EMBEDDING_PROVIDER"
echo "Embedding model path: $EMBEDDING_MODEL_PATH"
echo "Embedding device: $EMBEDDING_DEVICE"
if [[ "$EMBEDDING_PROVIDER" == "remote" ]]; then
  echo "Embedding URL: $EMBEDDING_URL"
fi
echo "VLM base URL: $VLM_BASE_URL"
echo "VLM model: $VLM_MODEL"
echo "Python: $PYTHON_BIN"
echo "Transition pause: $TRANSITION_PAUSE"
echo "Prune interval: $PRUNE_INTERVAL"
echo "Capacity: min=$CAPACITY_MIN max=$CAPACITY_MAX step=$CAPACITY_STEP"
if [[ -n "${TASKS:-}" ]]; then
  echo "Task override: $TASKS"
fi
echo

"$PYTHON_BIN" - <<PY
import json
import urllib.request

base_url = "$VLM_BASE_URL".rstrip("/")
request = urllib.request.Request(f"{base_url}/models")
try:
  with urllib.request.urlopen(request, timeout=10) as response:
    body = json.load(response)
  models = body.get("data") if isinstance(body, dict) else None
  model_ids = [
      str(item.get("id", "")) for item in models
      if isinstance(item, dict) and item.get("id")
  ] if isinstance(models, list) else []
  print(f"VLM health: ok; models={model_ids or 'unknown'}")
except Exception as error:
  raise SystemExit(
      "VLM health: failed; "
      f"{type(error).__name__}: {error}. "
      "Start/restart the remote vLLM service and SSH tunnel before running."
  )
PY
echo

"$PYTHON_BIN" - <<PY
import json
import urllib.request

provider = "$EMBEDDING_PROVIDER"
if provider == "local":
  from dms.embeddings import LocalEmbeddingProvider
  embedder = LocalEmbeddingProvider("$EMBEDDING_MODEL_PATH", device="$EMBEDDING_DEVICE")
  vectors = embedder.encode(["health check"])
  print(f"Embedding health: ok; provider=local; vectors={len(vectors)}; dim={len(vectors[0]) if vectors else None}")
elif provider == "remote":
  url = "$EMBEDDING_URL"
  payload = json.dumps({"texts": ["health check"]}).encode("utf-8")
  request = urllib.request.Request(
      url, data=payload, headers={"Content-Type": "application/json"}
  )
  with urllib.request.urlopen(request, timeout=5) as response:
    body = json.load(response)
  embeddings = body.get("embeddings")
  dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None
  print(f"Embedding health: ok; provider=remote; vectors={len(embeddings) if isinstance(embeddings, list) else None}; dim={dim}")
else:
  raise SystemExit(f"Unsupported EMBEDDING_PROVIDER={provider!r}; use local or remote.")
PY
echo

CMD=(
  "$PYTHON_BIN" -u run.py
  --agent_name=dms
  --benchmark_config="$CONFIG"
  --dms_memory_dir="$MEMORY_DIR"
  --dms_embedding_provider="$EMBEDDING_PROVIDER"
  --dms_embedding_model_path="$EMBEDDING_MODEL_PATH"
  --dms_embedding_device="$EMBEDDING_DEVICE"
  --dms_embedding_url="$EMBEDDING_URL"
  --dms_vlm_base_url="$VLM_BASE_URL"
  --dms_vlm_model="$VLM_MODEL"
  --dms_prune_interval_tasks="$PRUNE_INTERVAL"
  --dms_transition_pause="$TRANSITION_PAUSE"
  --dms_memory_capacity_min="$CAPACITY_MIN"
  --dms_memory_capacity_max="$CAPACITY_MAX"
  --dms_memory_capacity_step="$CAPACITY_STEP"
  --output_path="$OUT"
  --results_csv="$RESULTS_CSV"
)

if [[ -n "${TASKS:-}" ]]; then
  CMD+=(--tasks="$TASKS")
fi

"${CMD[@]}"

"$PYTHON_BIN" scripts/dms_metrics_report.py "$RESULTS_CSV" --output_dir "$OUT/dms_metrics_report"

echo
echo "DMS mini-benchmark finished."
echo "Output: $OUT"
echo "Results CSV: $RESULTS_CSV"
echo "Run log: $OUT/run.log"
echo "Summary JSON: $OUT/summary.json"
echo "Metrics report: $OUT/dms_metrics_report"
echo "Overall dashboard: $OUT/dms_metrics_report/overall_dashboard.svg"
echo "Per-task reports: $OUT/dms_metrics_report/per_task/"
echo "Task report index: $OUT/dms_metrics_report/per_task/task_report_index.json"
echo "DMS memory metadata: $MEMORY_DIR/dms_memory.json"
echo "DMS trajectories: $MEMORY_DIR/memory_trajectories/"
