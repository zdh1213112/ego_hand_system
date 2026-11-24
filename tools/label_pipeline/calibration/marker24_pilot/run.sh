#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$TOOL_DIR/../.." && pwd)"
PILOT_PYTHON="${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
PILOT_OUTPUT="${1:-$ROOT_DIR/output/native_kb_training_20261111/marker24_pilot_$(date +%Y%m%d_%H%M%S)}"
"$PILOT_PYTHON" "$TOOL_DIR/run_pilot.py" --enable-marker24 --output "$PILOT_OUTPUT"
"$PILOT_PYTHON" "$TOOL_DIR/build_review.py" --pilot "$PILOT_OUTPUT"
echo "实验结果：$PILOT_OUTPUT"
echo "查看器：$PILOT_OUTPUT/review/index.html"
