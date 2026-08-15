#!/usr/bin/env bash
# Build and verify the DMS reproduction environment from WSL.
# Keep credentials out of this file. Start the remote VLM/tunnel separately when needed.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-android_world}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DEVICE_SERIAL="${DEVICE_SERIAL:-emulator-5554}"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"

EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$ROOT_DIR/embedding_model}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:18000/v1}"
VLM_MODEL="${VLM_MODEL:-/root/autodl-tmp/model}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-16384}"

section() {
  printf '\n== %s ==\n' "$1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

section "Base commands"
require_command bash
require_command curl
require_command adb
require_command conda

section "Conda environment"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python -V

section "Python package setup"
cd "$ROOT_DIR"
python -m pip install -e .
python -m pip install \
  "numpy==1.26.3" \
  "pandas==2.1.4" \
  "fastapi" \
  "uvicorn" \
  "sentence-transformers==2.7.0" \
  "transformers<5" \
  --upgrade-strategy only-if-needed
python -m pip check

section "Android SDK and emulator"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export ANDROID_SDK_ROOT="$ANDROID_SDK_ROOT"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
adb devices
adb -s "$DEVICE_SERIAL" shell getprop sys.boot_completed | grep -q 1
adb -s "$DEVICE_SERIAL" shell pm list packages | grep -E "dialer|camera|contacts|deskclock" || true

section "Intent resolution"
adb -s "$DEVICE_SERIAL" shell cmd package resolve-activity --brief -a android.intent.action.DIAL || true
adb -s "$DEVICE_SERIAL" shell cmd package resolve-activity --brief -a android.media.action.IMAGE_CAPTURE || true

section "Local embedding"
EMBEDDING_PROVIDER="$EMBEDDING_PROVIDER" \
EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH" \
EMBEDDING_DEVICE="$EMBEDDING_DEVICE" \
python - <<'PY'
import os
from dms.embeddings import LocalEmbeddingProvider, RemoteEmbeddingProvider

if os.environ["EMBEDDING_PROVIDER"] == "local":
    embedder = LocalEmbeddingProvider(
        os.environ["EMBEDDING_MODEL_PATH"],
        device=os.environ["EMBEDDING_DEVICE"],
    )
else:
    embedder = RemoteEmbeddingProvider()
vectors = embedder.encode(["DMS environment check"])
print(f"embedding_count={len(vectors)} dim={len(vectors[0])}")
PY

section "Remote VLM endpoint"
curl -fsS --max-time 15 "$VLM_BASE_URL/models" >/dev/null
printf 'vlm_base_url=%s\nvlm_model=%s\nmax_model_len=%s\n' \
  "$VLM_BASE_URL" "$VLM_MODEL" "$VLM_MAX_MODEL_LEN"

section "AndroidWorld smoke task"
python -u minimal_task_runner.py \
  --agent_name=zero_shot \
  --task="${SMOKE_TASK:-ClockStopWatchRunning}" \
  --task_random_seed="${TASK_RANDOM_SEED:-30}"

section "Done"
