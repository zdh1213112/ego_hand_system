#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_PYTHON=${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "Set EGO_NATIVE_PYTHON to the Python executable in the ego-hand environment." >&2
  exit 1
fi
exec "$TASK_PYTHON" -B "$SCRIPT_DIR/export_ego_nokov_labels.py" "$@"
