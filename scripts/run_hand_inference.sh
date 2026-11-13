#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${EGO_CONDA_ENV:-ego-hand}"
ARGS=()
while (($#)); do
  case "$1" in
    --conda-env)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo 'Missing value for --conda-env' >&2; exit 2; }
      CONDA_ENV="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
for ARG in "${ARGS[@]}"; do
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then
    exec python3 "$ROOT/scripts/run_hand_inference.py" "${ARGS[@]}"
  fi
done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/python_deps/wilor_runtime:$ROOT/scripts"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ego-hand-matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ego-hand-ultralytics}"
export TORCHINDUCTOR_CACHE_DIR="${EGO_TORCHINDUCTOR_CACHE_DIR:-/tmp/ego-hand-torchinductor}"
if [[ -n "${EGO_HAND_PYTHON:-}" ]]; then
  exec "$EGO_HAND_PYTHON" "$ROOT/scripts/run_hand_inference.py" "${ARGS[@]}"
fi
exec conda run --no-capture-output -n "$CONDA_ENV" python "$ROOT/scripts/run_hand_inference.py" "${ARGS[@]}"
