#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEGO_HAND_BUILD_LIVE=ON
cmake --build "${BUILD_DIR}" -j"$(nproc)" --target ego_live_bridge

exec conda run --no-capture-output -n ego-hand \
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
    QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu \
    python "${PROJECT_DIR}/scripts/ego_live_stereo.py" \
    --bridge "${BUILD_DIR}/ego_live_bridge" \
    --sdk-config "${PROJECT_DIR}/third_party/orbbec_sdk/OrbbecSDKConfig.xml" \
    --model "${PROJECT_DIR}/models/hand_landmarker.task" \
    --output "${PROJECT_DIR}/output/ego_live" \
    --mano \
    --mano-source "${PROJECT_DIR}/third_party/MANO" \
    --mano-model-dir "${PROJECT_DIR}/models/mano" \
    --mano-profile "${PROJECT_DIR}/output/mano_fit_refined" \
    "$@"
