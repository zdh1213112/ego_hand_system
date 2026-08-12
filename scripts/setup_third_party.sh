#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${PROJECT_DIR}/.gitmodules" ]]; then
  echo "error: .gitmodules is missing; use a complete project checkout." >&2
  exit 2
fi

echo "Initializing public source dependencies..."
git -C "${PROJECT_DIR}" submodule sync --recursive
git -C "${PROJECT_DIR}" submodule update --init third_party/MANO third_party/basalt third_party/WiLoR

echo
echo "Installing public MediaPipe model..."
"${PROJECT_DIR}/scripts/install_mediapipe_model.sh"

echo
echo "Installing Basalt ${BASALT_VERSION:-0.1.7} runtime..."
"${PROJECT_DIR}/scripts/install_basalt_runtime.sh"

echo
"${PYTHON:-python3}" "${PROJECT_DIR}/scripts/check_third_party.py"

echo
echo "Source dependencies are initialized."
echo "Licensed/vendor assets shown as MISSING must be copied manually for their workflow."
