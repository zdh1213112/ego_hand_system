#!/usr/bin/env bash
# Extend the project's validated ego-hand environment with GEM-X hand support.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EGO_CONDA_ENV:-ego-hand}"
GEM_ROOT="${GEM_X_ROOT:-/home/lff/xense_project/GEM-X}"
[[ -d "$GEM_ROOT/gem" ]] || { echo "GEM-X checkout missing: $GEM_ROOT" >&2; exit 2; }
if ! conda run -n "$ENV_NAME" python -c 'import torch,ultralytics' >/dev/null 2>&1; then
  "$ROOT/scripts/setup_python_environment.sh"
fi
# GEM-X declares opencv-python-headless, whereas this project uses the
# mutually exclusive opencv-contrib distribution. Install the remaining GEM-X
# requirements explicitly and expose the checkout with a .pth file.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  'transformers==5.3.0' 'py-soma-x==0.3.3' 'onnxruntime>=1.20,<2' 'open3d==0.19.0' 'pytest>=8,<10'
conda run --no-capture-output -n "$ENV_NAME" python - "$GEM_ROOT" <<'PY'
import pathlib, site, sys
root = pathlib.Path(sys.argv[1]).resolve()
(pathlib.Path(site.getsitepackages()[0]) / 'ego_hand_gem_x.pth').write_text(str(root) + '\n')
PY
conda run --no-capture-output -n "$ENV_NAME" env -u ALL_PROXY -u all_proxy \
  PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib \
  python "$ROOT/scripts/check_python_environment.py"
conda run --no-capture-output -n "$ENV_NAME" python -c 'import gem,soma,onnxruntime,open3d; print("GEM-X and SOMA available", soma.__version__)'
