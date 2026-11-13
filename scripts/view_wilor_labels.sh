#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_PYTHON="${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
exec "$VIEWER_PYTHON" -B "$SCRIPT_DIR/view_wilor_labels.py" "$@"
