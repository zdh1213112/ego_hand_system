#!/usr/bin/env bash
# Full VIO comparison with partial NOKOV coverage; write to a new directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BASALT_COMPARISON_PYTHON:-/home/lff/miniforge3/envs/wilor/bin/python}"
OUTPUT="${1:?Usage: $0 NEW_OUTPUT_DIRECTORY [EGO_SESSION] [NOKOV_SESSION]}"
SESSION="${2:-/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO-4B_AZLCA610091_20261104_195732}"
REFERENCE="${3:-/home/lff/xense_project/NOKOV/nokov-rigid-capture/sessions/session_20261104_195743_694430}"
CACHE="${BASALT_BUILD_CACHE:-$ROOT/output/basalt_comparison_20261104_195844}"
cd "$ROOT"
[[ ! -e "$OUTPUT" ]] || { echo 'Choose a new output directory.' >&2; exit 2; }
[[ -x "$CACHE/build_original/basalt_vio" ]]
[[ -x "$ROOT/third_party/basalt_monado/build/basalt_vio" ]]
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
ln -s "$(realpath "$CACHE/build_original")" "$OUTPUT/build_original"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" scripts/compare_basalt_ego4.py prepare --session "$SESSION" --output "$OUTPUT" > "$OUTPUT/prepare.log" 2>&1
"$PYTHON" scripts/calibrate_nokov_ego4.py --output "$OUTPUT" --reference "$REFERENCE" --calibration-seconds 20 > "$OUTPUT/nokov_calibration.log" 2>&1
"$PYTHON" scripts/compare_basalt_ego4.py run --output "$OUTPUT" --threads 2 --repeats 3 > "$OUTPUT/benchmark.log" 2>&1
"$PYTHON" scripts/analyze_basalt_nokov_overlap.py --output "$OUTPUT" > "$OUTPUT/evaluation.log" 2>&1
"$PYTHON" scripts/visualize_vio_nokov_trajectories.py --comparison "$OUTPUT" --output "$OUTPUT" > "$OUTPUT/viewer.log" 2>&1
printf 'Report: %s/comparison_report.html\n' "$OUTPUT"
printf 'Viewer: %s/viewer.html\n' "$OUTPUT"
