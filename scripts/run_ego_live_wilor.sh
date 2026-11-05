#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDK_ROOT="${ORBBEC_SDK_ROOT:-${PROJECT_DIR}/third_party/orbbec_sdk}"
if [[ ! -f "${SDK_ROOT}/lib/OrbbecSDKConfig.cmake" ]]; then
    echo "Orbbec SDK not found at ${SDK_ROOT}; set ORBBEC_SDK_ROOT." >&2
    exit 1
fi
SDK_CONFIG="${SDK_ROOT}/shared/OrbbecSDKConfig.xml"
if [[ ! -f "${SDK_CONFIG}" ]]; then
    SDK_CONFIG="${SDK_ROOT}/lib/OrbbecSDKConfig.xml"
fi
cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release -DEGO_HAND_BUILD_LIVE=ON \
    -DORBBEC_SDK_ROOT="${SDK_ROOT}"
cmake --build "${PROJECT_DIR}/build" -j"$(nproc)" --target ego_live_bridge
exec conda run --no-capture-output -n ego-hand \
    env PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_DIR}/scripts" \
    python "${PROJECT_DIR}/scripts/ego_live_wilor.py" \
    --bridge "${PROJECT_DIR}/build/ego_live_bridge" \
    --sdk-config "${SDK_CONFIG}" "$@"
