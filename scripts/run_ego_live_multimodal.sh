#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDK_ROOT="${ORBBEC_SDK_ROOT:-${PROJECT_DIR}/third_party/orbbec_sdk}"
if [[ -z "${ORBBEC_SDK_ROOT:-}" && ! -f "${SDK_ROOT}/lib/OrbbecSDKConfig.cmake" ]]; then
    for SDK_CANDIDATE in "${PROJECT_DIR}"/../OrbbecSDK* "${PROJECT_DIR}"/../EGO_glove/OrbbecSDK*; do
        if [[ -f "${SDK_CANDIDATE}/lib/OrbbecSDKConfig.cmake" ]]; then
            SDK_ROOT="${SDK_CANDIDATE}"
            break
        fi
    done
fi
EGO_PYTHON="${EGO_HAND_PYTHON:-${HOME}/miniforge3/envs/ego-hand/bin/python}"
if [[ ! -x "${EGO_PYTHON}" ]]; then
    echo "Set EGO_HAND_PYTHON to the ego-hand environment's Python." >&2
    exit 1
fi
if [[ ! -f "${SDK_ROOT}/lib/OrbbecSDKConfig.cmake" ]]; then
    echo "Orbbec SDK not found at ${SDK_ROOT}; set ORBBEC_SDK_ROOT." >&2
    exit 1
fi
SDK_CONFIG="${SDK_ROOT}/shared/OrbbecSDKConfig.xml"
if [[ ! -f "${SDK_CONFIG}" ]]; then
    SDK_CONFIG="${SDK_ROOT}/lib/OrbbecSDKConfig.xml"
fi
env -u PYTHONPATH PYTHONNOUSERSITE=1 "${EGO_PYTHON}" \
    "${SCRIPT_DIR}/install_basalt_live_headers.py"
cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release -DEGO_HAND_BUILD_LIVE=ON \
    -DEGO_HAND_BUILD_BASALT_LIVE=ON -DORBBEC_SDK_ROOT="${SDK_ROOT}"
cmake --build "${PROJECT_DIR}/build" -j"${EGO_BUILD_JOBS:-4}" \
    --target ego_live_bridge ego_basalt_live
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 "${EGO_PYTHON}" -u \
    "${SCRIPT_DIR}/run_ego_live_multimodal.py" --sdk-config "${SDK_CONFIG}" "$@"
