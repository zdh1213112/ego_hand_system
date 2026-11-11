#!/usr/bin/env bash
# Keep detector, fusion, temporal processing and render flags shared with the
# raw baseline. Only the image preprocessing stage differs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EGO_WILOR_UNDISTORT_FOV_DEG=120
export EGO_WILOR_UNDISTORT_WIDTH=0
export EGO_WILOR_UNDISTORT_HEIGHT=0
export EGO_WILOR_ORIGINAL_VIDEO=0
ARGS=()
while (($#)); do
  case "$1" in
    --original-video)
      export EGO_WILOR_ORIGINAL_VIDEO=1
      shift ;;
    --undistort-fov-deg|--undistort-width|--undistort-height)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --undistort-fov-deg) export EGO_WILOR_UNDISTORT_FOV_DEG="$2" ;;
        --undistort-width) export EGO_WILOR_UNDISTORT_WIDTH="$2" ;;
        --undistort-height) export EGO_WILOR_UNDISTORT_HEIGHT="$2" ;;
      esac
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 (--mcap FILE | --session EGO4_DIR) --output NEW_DIR [options]"
      echo "Undistort KB/GEN-DS views before detection, WiLoR pose estimation and multiview fusion."
      echo "  --undistort-fov-deg N   Horizontal pinhole FOV (default 120; 0 < N < 175)."
      echo "  --undistort-width W --undistort-height H   Optional even output resolution."
      echo "  --original-video      Also render a hybrid final video on the original fisheye frames."
      echo "                        Maps 3D, native-2D fallback and boxes; disabled by --no-video/--stop-after-inference."
      echo "Default resolution matches input; FFV1 storage is lossless. Extrinsics/timestamps are preserved."
      echo "All baseline options below are forwarded unchanged. Use a separate output directory for each FOV."
      bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" --help
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec bash "$ROOT/scripts/run_multiview_wilor_experiment.sh" "${ARGS[@]}"
