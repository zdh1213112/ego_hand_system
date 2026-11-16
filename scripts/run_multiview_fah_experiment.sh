#!/usr/bin/env bash
# Keep the existing multiview experiment and replace only its pose/mesh backend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAH_ROOT="${EGO_FAH_ROOT:-/mnt/LFF/xense_project/FAH}"
FAH_PYTHON="${EGO_FAH_PYTHON:-}"
FAH_MODEL="${EGO_FAH_MODEL:-fah-s}"
FAH_WEIGHTS="${EGO_FAH_WEIGHTS:-}"
FAH_MANO_DIR="${EGO_FAH_MANO_DIR:-}"
FAH_SAM3D="${EGO_FAH_SAM3D:-}"
ARGS=()
while (($#)); do
  case "$1" in
    --fah-root|--fah-python|--fah-model|--fah-weights|--fah-mano-dir|--fah-sam3d|--fah-mode)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --fah-root) FAH_ROOT="$2" ;;
        --fah-python) FAH_PYTHON="$2" ;;
        --fah-model) FAH_MODEL="$2" ;;
        --fah-weights) FAH_WEIGHTS="$2" ;;
        --fah-mano-dir) FAH_MANO_DIR="$2" ;;
        --fah-sam3d) FAH_SAM3D="$2" ;;
        --fah-mode) ARGS+=(--wilor-mode "$2") ;;
      esac
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 (--mcap FILE | --session EGO4_DIR) --output DIR [original experiment options]"
      echo "Replace WiLoR pose/mesh inference with FAH; keep synchronization, fusion and video layout."
      echo "FAH: --fah-root DIR --fah-python FILE --fah-model fah-s|fah-b|fah|fah-h"
      echo "     --fah-weights DIR --fah-mano-dir DIR --fah-sam3d DIR"
      echo "     --fah-mode shared|legacy|both (alias of --wilor-mode; default shared)"
      echo "Default FAH Python: FAH_ROOT/.venv/bin/python when available, otherwise --conda-env."
      echo "The original detector, input pixels, calibration, MP4 layout and flags are retained."
      echo "FAH runs framewise dual-handedness hypotheses; temporal processing remains in the existing pipeline."
      bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" --help
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$FAH_MODEL" in fah-s|fah-b|fah|fah-h) ;; *) echo "Unknown FAH model: $FAH_MODEL" >&2; exit 2 ;; esac
FAH_WEIGHTS="${FAH_WEIGHTS:-$FAH_ROOT/weights}"
FAH_MANO_DIR="${FAH_MANO_DIR:-$FAH_ROOT/data/mano}"
FAH_SAM3D="${FAH_SAM3D:-$FAH_ROOT/data/sam-3d-body-dinov3}"
if [[ -z "$FAH_PYTHON" && -x "$FAH_ROOT/.venv/bin/python" ]]; then
  FAH_PYTHON="$FAH_ROOT/.venv/bin/python"
fi
[[ -f "$FAH_ROOT/fah/api.py" ]] || { echo "FAH checkout not found: $FAH_ROOT" >&2; exit 2; }
for FILE in "$FAH_WEIGHTS/$FAH_MODEL/config.json" "$FAH_WEIGHTS/$FAH_MODEL/model.pt" \
    "$FAH_WEIGHTS/shared/mesh_fitter.pt" "$FAH_MANO_DIR/MANO_RIGHT.pkl" \
    "$FAH_SAM3D/model_config.yaml" "$FAH_SAM3D/assets/mhr_model.pt"; do
  [[ -f "$FILE" ]] || { echo "FAH asset not found: $FILE" >&2; exit 2; }
done
[[ -z "$FAH_PYTHON" || -x "$FAH_PYTHON" ]] || { echo "FAH Python is not executable: $FAH_PYTHON" >&2; exit 2; }
export EGO_EXPERIMENT_POSE_BACKEND=fah
export EGO_FAH_ROOT="$FAH_ROOT" EGO_FAH_PYTHON="$FAH_PYTHON" EGO_FAH_MODEL="$FAH_MODEL"
export EGO_FAH_WEIGHTS="$FAH_WEIGHTS" EGO_FAH_MANO_DIR="$FAH_MANO_DIR" EGO_FAH_SAM3D="$FAH_SAM3D"
exec bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" "${ARGS[@]}"
