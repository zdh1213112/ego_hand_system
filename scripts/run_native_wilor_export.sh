#!/usr/bin/env bash
# Full export: bash scripts/run_native_wilor_export.sh --output /path/to/new_output
# Sample export: append --sample-stride 80 --max-samples 32
set -Eeuo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$TASK_PYTHON" "$TASK_ROOT/scripts/run_native_wilor_export.py" "$@"
