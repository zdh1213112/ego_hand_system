#!/usr/bin/env bash
# Use the exact WiLoR experiment pipeline with GEM-X as its pose backend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEM_ROOT="${GEM_X_ROOT:-/mnt/LFF/xense_project/GEM-X}"
if [[ ! -d "$GEM_ROOT" && -d /home/lff/xense_project/GEM-X ]]; then
  GEM_ROOT=/home/lff/xense_project/GEM-X
fi
CHECKPOINT="${EGO_GEM_CHECKPOINT:-}"
SOMA_ASSETS="${EGO_GEM_SOMA_ASSETS:-}"
GEM_PYTHON="${EGO_GEM_PYTHON:-}"
GEM_MODE="${EGO_GEM_MODE:-shared}"
SHARED_STEPS="${EGO_GEM_SHARED_STEPS:-100}"
SHARED_BATCH_SIZE="${EGO_GEM_SHARED_BATCH_SIZE:-32}"
ARGS=()
while (($#)); do
  case "$1" in
    --gem-root|--checkpoint|--soma-assets|--gem-python|--gem-mode|--shared-steps|--shared-batch-size)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --gem-root) GEM_ROOT="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --soma-assets) SOMA_ASSETS="$2" ;;
        --gem-python) GEM_PYTHON="$2" ;;
        --gem-mode) GEM_MODE="$2" ;;
        --shared-steps) SHARED_STEPS="$2" ;;
        --shared-batch-size) SHARED_BATCH_SIZE="$2" ;;
      esac
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 (--mcap FILE | --session EGO4_DIR) --output DIR [WiLoR experiment options]"
      echo "Run the WiLoR experiment pipeline on original frames with GEM-X hand pose/mesh."
      echo "GEM options: --gem-root DIR --checkpoint FILE --soma-assets DIR --gem-python FILE"
      echo "Detector, camera selection, fusion, assists, video and all other flags follow the WiLoR experiment."
      echo "--gem-mode shared|independent|both (default shared): choose shared SOMA fitting or per-camera geometry."
      echo "Shared mode projects one fitted skeleton/mesh into every raw view, without native-2D fallback."
      echo "--shared-steps N (100), --shared-batch-size N (32) configure shared SOMA fitting."
      echo "--final-renderer controls only independent-mode video; --final-video enables the chosen final video(s)."
      echo "GEM uses pinhole intrinsics as an approximation on raw fisheye images; fusion uses the original camera model."
      bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" --help
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$GEM_MODE" in shared|independent|both) ;; *) echo "--gem-mode must be shared, independent, or both" >&2; exit 2 ;; esac
[[ "$SHARED_STEPS" =~ ^[1-9][0-9]*$ && "$SHARED_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "Shared steps/batch size must be positive integers" >&2; exit 2; }
CHECKPOINT="${CHECKPOINT:-$GEM_ROOT/weights/egohand_s060000.ckpt}"
SOMA_ASSETS="${SOMA_ASSETS:-$GEM_ROOT/assets/soma}"
[[ -d "$GEM_ROOT/gem" ]] || { echo "GEM-X checkout not found: $GEM_ROOT" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "EgoHand checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$SOMA_ASSETS/SOMAHand.npz" ]] || { echo "SOMAHand.npz not found in: $SOMA_ASSETS" >&2; exit 2; }
[[ -z "$GEM_PYTHON" || -x "$GEM_PYTHON" ]] || { echo "GEM Python is not executable: $GEM_PYTHON" >&2; exit 2; }
export EGO_EXPERIMENT_POSE_BACKEND=gem
export EGO_GEM_ROOT="$GEM_ROOT" EGO_GEM_CHECKPOINT="$CHECKPOINT" EGO_GEM_SOMA_ASSETS="$SOMA_ASSETS" EGO_GEM_PYTHON="$GEM_PYTHON"
export EGO_GEM_MODE="$GEM_MODE" EGO_GEM_SHARED_STEPS="$SHARED_STEPS" EGO_GEM_SHARED_BATCH_SIZE="$SHARED_BATCH_SIZE"
# This entry point always uses raw source images and original calibration.
unset EGO_WILOR_UNDISTORT_FOV_DEG EGO_WILOR_UNDISTORT_WIDTH EGO_WILOR_UNDISTORT_HEIGHT EGO_WILOR_ORIGINAL_VIDEO
exec bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" "${ARGS[@]}"
