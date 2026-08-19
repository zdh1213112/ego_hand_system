#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EGO_CONDA_ENV:-ego-hand}"
PYPI_INDEX="${EGO_PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_INDEX="${EGO_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CHUMPY_REVISION="580566eafc9ac68b2614b64d6f7aaa84eebb70da"
PIP_CACHE_DIR="${EGO_PIP_CACHE_DIR:-/tmp/ego-hand-pip-cache}"

if ! command -v conda >/dev/null 2>&1; then
    printf 'error: conda is not available on PATH\n' >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    printf 'error: git is required to install the pinned chumpy source\n' >&2
    exit 2
fi

if conda run -n "${ENV_NAME}" python -c 'import sys' >/dev/null 2>&1; then
    printf '[python-env] updating Conda bootstrap: %s\n' "${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f "${PROJECT_DIR}/environment.yml"
else
    printf '[python-env] creating Conda bootstrap: %s\n' "${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f "${PROJECT_DIR}/environment.yml"
fi

PIP=(
    conda run --no-capture-output -n "${ENV_NAME}"
    env PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib
    PIP_CACHE_DIR="${PIP_CACHE_DIR}" PIP_DISABLE_PIP_VERSION_CHECK=1
    python -m pip
)
NETWORK_OPTIONS=(--retries 20 --resume-retries 20 --timeout 120)
mkdir -p "${PIP_CACHE_DIR}"

printf '[python-env] installing CUDA 12.8 PyTorch wheels\n'
"${PIP[@]}" install --upgrade "${NETWORK_OPTIONS[@]}" \
    --index-url "${PYTORCH_INDEX}" --extra-index-url "${PYPI_INDEX}" \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128

printf '[python-env] installing core runtime from PyPI\n'
REPAIR_OPENCV=0
if "${PIP[@]}" show opencv-python >/dev/null 2>&1 || \
   "${PIP[@]}" show opencv-python-headless >/dev/null 2>&1; then
    REPAIR_OPENCV=1
    printf '[python-env] removing duplicate OpenCV distributions\n'
    "${PIP[@]}" uninstall -y opencv-python opencv-python-headless || true
fi
"${PIP[@]}" install --upgrade "${NETWORK_OPTIONS[@]}" \
    --index-url "${PYPI_INDEX}" \
    -r "${PROJECT_DIR}/requirements/core.txt"
if ((REPAIR_OPENCV)); then
    # OpenCV wheel distributions share the same cv2 files. Reinstall contrib
    # after removing a duplicate distribution so no shared files remain missing.
    "${PIP[@]}" install --force-reinstall --no-deps "${NETWORK_OPTIONS[@]}" \
        --index-url "${PYPI_INDEX}" opencv-contrib-python==5.0.0.93
fi

if conda run -n "${ENV_NAME}" python -c \
   'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("chumpy") else 1)' \
   >/dev/null 2>&1; then
    printf '[python-env] keeping the installed chumpy package\n'
else
    printf '[python-env] installing legacy chumpy with build isolation disabled\n'
    printf '[python-env] chumpy is required to unpickle official MANO v1.2 models\n'
    printf '[python-env] the project applies its NumPy 2 compatibility shim at runtime\n'
    "${PIP[@]}" install --upgrade "${NETWORK_OPTIONS[@]}" \
        --index-url "${PYPI_INDEX}" --no-build-isolation \
        "git+https://github.com/mattloper/chumpy.git@${CHUMPY_REVISION}"
fi

printf '[python-env] installing WiLoR runtime from PyPI\n'
# Ultralytics 8.4 uses the maintained ultralytics-thop distribution. The old
# thop distribution installs into the same Python package directory, so remove
# it before upgrading an environment created by an earlier project revision.
if "${PIP[@]}" show thop >/dev/null 2>&1; then
    printf '[python-env] replacing legacy thop with ultralytics-thop\n'
    "${PIP[@]}" uninstall -y thop
fi
"${PIP[@]}" install --upgrade "${NETWORK_OPTIONS[@]}" \
    --index-url "${PYPI_INDEX}" \
    -r "${PROJECT_DIR}/requirements/wilor.txt"

printf '[python-env] installing ultralytics without its duplicate OpenCV distribution\n'
"${PIP[@]}" install --upgrade "${NETWORK_OPTIONS[@]}" \
    --index-url "${PYPI_INDEX}" --no-deps ultralytics==8.4.56

printf '[python-env] validating imports and package invariants\n'
# Import validation does not perform network requests. Do not let a shell-level
# generic SOCKS proxy (notably the unsupported "socks://" spelling) prevent
# httpx from being imported by Gradio. HTTP(S) proxy settings remain available
# to every preceding download step.
conda run --no-capture-output -n "${ENV_NAME}" \
    env -u ALL_PROXY -u all_proxy \
    PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib \
    python "${PROJECT_DIR}/scripts/check_python_environment.py"

printf '\n[python-env] ready. Activate with: conda activate %s\n' "${ENV_NAME}"
