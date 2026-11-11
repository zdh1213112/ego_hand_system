#!/usr/bin/env bash
# Re-run in a NEW output directory; original recordings remain read-only.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BASALT_COMPARISON_PYTHON:-/home/lff/miniforge3/envs/wilor/bin/python}"
OUTPUT="${1:?Usage: $0 NEW_OUTPUT_DIRECTORY}"
SESSION="${2:-/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO-4B_AZLCA610091_20261104_195844}"
REFERENCE="${3:-/home/lff/xense_project/NOKOV/nokov-rigid-capture/sessions/session_20261104_195852_461949}"
cd "$ROOT"
[[ ! -e "$OUTPUT" ]] || { echo 'Choose a new output directory.' >&2; exit 2; }
mkdir -p "$OUTPUT/deps"
OUTPUT="$(realpath "$OUTPUT")"
git clone --filter=blob:none --no-checkout https://gitlab.com/VladyslavUsenko/basalt-headers.git "$OUTPUT/deps/basalt-headers"
git -C "$OUTPUT/deps/basalt-headers" checkout aa441ba3e51050c47ba1902537792a2e4db7e43d
"$PYTHON" scripts/build_basalt_comparison.py --output "$OUTPUT"
"$PYTHON" scripts/compare_basalt_ego4.py prepare --session "$SESSION" --output "$OUTPUT"
"$PYTHON" scripts/compare_basalt_ego4.py run --output "$OUTPUT" --threads 2 --repeats 3 --save-marg
"$PYTHON" scripts/compare_basalt_ego4.py map --output "$OUTPUT"
"$PYTHON" scripts/compare_basalt_ego4.py map --output "$OUTPUT" --mapping-solver lm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" scripts/calibrate_nokov_ego4.py --output "$OUTPUT" --reference "$REFERENCE"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" scripts/analyze_basalt_ego4_comparison.py --output "$OUTPUT" --video
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" scripts/visualize_vio_nokov_trajectories.py --comparison "$OUTPUT" --output "$OUTPUT"
printf 'Report: %s/comparison_report.html\n' "$OUTPUT"
printf 'Viewer: %s/viewer.html\n' "$OUTPUT"
