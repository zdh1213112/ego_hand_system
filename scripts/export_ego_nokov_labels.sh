#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_PYTHON=${EGO_NATIVE_PYTHON:-}
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == "ego-hand" ]]; then
    TASK_PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x /home/p3/miniforge3/envs/ego-hand/bin/python ]]; then
    TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
  elif command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n ego-hand python -B "$SCRIPT_DIR/export_ego_nokov_labels.py" "$@"
  fi
fi
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "Set EGO_NATIVE_PYTHON to the Python executable in the ego-hand environment." >&2
  exit 1
fi
exec "$TASK_PYTHON" -B "$SCRIPT_DIR/export_ego_nokov_labels.py" "$@"
