#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCAP=""
INPUT_KIND="mcap"
CAMERAS_EXPLICIT=0
REFERENCE_EXPLICIT=0
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
DETECTOR="${EGO_WILOR_DETECTOR:-$ROOT/models/wilor/detector.pt}"
CAMERAS=(camera0 camera1 camera2 camera3 camera4 camera5)
REFERENCE_CAMERA="camera2"
MAX_FRAMES=60
BATCH_SIZE=""
FRAME_BATCH_SIZE=""
PREPROCESS_WORKERS=""
MAX_DETECTIONS_PER_CLASS=""
COMPILE_BACKBONE=""
FUSION_WORKERS=""
GPU_PROFILE="compatible"
NO_VIDEO=0
STOP_AFTER_INFERENCE=0
FINAL_VIDEO_ENABLED=0
FINAL_2D_RESIDUAL_FALLBACK_PX="${EGO_FINAL_2D_RESIDUAL_FALLBACK_PX:-35}"
FINAL_2D_RESIDUAL_PERCENTILE="${EGO_FINAL_2D_RESIDUAL_PERCENTILE:-95}"
FINAL_MIN_JOINT_INLIER_VIEWS="${EGO_FINAL_MIN_JOINT_INLIER_VIEWS:-4}"
FINAL_WRIST_RESIDUAL_FALLBACK_PX="${EGO_FINAL_WRIST_RESIDUAL_FALLBACK_PX:-24}"
FINAL_REQUIRE_LOCAL_VIEW="${EGO_FINAL_REQUIRE_LOCAL_VIEW:-1}"
FINAL_PREDICTED_2D_FALLBACK="${EGO_FINAL_PREDICTED_2D_FALLBACK:-0}"
FINAL_RENDERER="${EGO_FINAL_RENDERER:-hybrid}"
FINAL_HYBRID_RESIDUAL_THRESHOLD_PX="${EGO_FINAL_HYBRID_RESIDUAL_THRESHOLD_PX:-12}"
FINAL_HYBRID_WRIST_THRESHOLD_PX="${EGO_FINAL_HYBRID_WRIST_THRESHOLD_PX:-15}"
FINAL_HYBRID_MAX_CONDITION="${EGO_FINAL_HYBRID_MAX_CONDITION:-100}"
FINAL_HYBRID_MAX_STEREO_DIFFERENCE_MM="${EGO_FINAL_HYBRID_MAX_STEREO_DIFFERENCE_MM:-20}"
DETECTOR_HANDEDNESS=""
ALLOW_SINGLE_HAND_ANCHORS="${EGO_ALLOW_SINGLE_HAND_ANCHORS:-0}"
GLOVE_MARKER_ASSIST="${EGO_GLOVE_MARKER_ASSIST:-${EGO_NOKOV_WILOR_ASSIST:-0}}"
MARKER_SATURATION_MAX="${EGO_MARKER_SATURATION_MAX:-100}"
MARKER_VALUE_MIN="${EGO_MARKER_VALUE_MIN:-160}"
MARKER_MIN_MATCHES="${EGO_MARKER_MIN_MATCHES:-3}"
MARKER_MIN_FINGER_GROUPS="${EGO_MARKER_MIN_FINGER_GROUPS:-2}"
MARKER_GLOBAL_MIN_MATCHES="${EGO_MARKER_GLOBAL_MIN_MATCHES:-8}"
MARKER_GLOBAL_MIN_FINGER_GROUPS="${EGO_MARKER_GLOBAL_MIN_FINGER_GROUPS:-4}"
MARKER_SEARCH_PADDING_PX="${EGO_MARKER_SEARCH_PADDING_PX:-45}"
MARKER_BBOX_PADDING_PX="${EGO_MARKER_BBOX_PADDING_PX:-12}"
MARKER_SEED_DISTANCE_PX="${EGO_MARKER_SEED_DISTANCE_PX:-35}"
MARKER_MATCH_DISTANCE_PX="${EGO_MARKER_MATCH_DISTANCE_PX:-13}"
MARKER_MAX_SHIFT_PX="${EGO_MARKER_MAX_SHIFT_PX:-20}"
MARKER_MAX_APPLIED_SHIFT_PX="${EGO_MARKER_MAX_APPLIED_SHIFT_PX:-10}"
MARKER_BLEND="${EGO_MARKER_BLEND:-0.15}"
MARKER_MAX_LOCAL_ADJUSTMENT_PX="${EGO_MARKER_MAX_LOCAL_ADJUSTMENT_PX:-3}"
SAM3_GLOVE_ASSIST="${EGO_SAM3_GLOVE_ASSIST:-0}"
SAM3_CONDA_ENV="${EGO_SAM3_CONDA_ENV:-annotator}"
SAM3_TOOL_ROOT="${EGO_SAM3_TOOL_ROOT:-/home/p3/data_sda1/tool/auto-obb-annotator}"
SAM3_MODEL="${EGO_SAM3_MODEL:-$SAM3_TOOL_ROOT/models/sam3.pt}"
SAM3_PROMPT="${EGO_SAM3_PROMPT:-hand wearing a black protective work glove}"
SAM3_DEVICE="${EGO_SAM3_DEVICE:-cuda}"
SAM3_CONF="${EGO_SAM3_CONF:-0.25}"
SAM3_MIN_SCORE="${EGO_SAM3_MIN_SCORE:-0.25}"
SAM3_MIN_BOX_IOU="${EGO_SAM3_MIN_BOX_IOU:-0.20}"
SAM3_MIN_JOINT_SUPPORT="${EGO_SAM3_MIN_JOINT_SUPPORT:-0.55}"
SAM3_HARD_JOINT_SUPPORT="${EGO_SAM3_HARD_JOINT_SUPPORT:-0.35}"
SAM3_TEXT_ONLY=0
SAM3_REFINE_2D="${EGO_SAM3_REFINE_2D:-0}"
SAM3_MAX_2D_CORRECTION_PX="${EGO_SAM3_MAX_2D_CORRECTION_PX:-24.0}"
SAM3_2D_CORRECTION_BLEND="${EGO_SAM3_2D_CORRECTION_BLEND:-0.75}"
SAM3_MAX_2D_BONE_CHANGE_RATIO="${EGO_SAM3_MAX_2D_BONE_CHANGE_RATIO:-0.55}"
CONDITION_HARD_THRESHOLD="${EGO_CONDITION_HARD_THRESHOLD:-150}"
ANATOMY_REFINEMENT="${EGO_ANATOMY_REFINEMENT:-0}"
JOINT_TEMPORAL_MIN_HAND_JOINTS="${EGO_JOINT_TEMPORAL_MIN_HAND_JOINTS:-18}"
JOINT_TEMPORAL_EXCESSIVE_JUMP_THRESHOLD_M="${EGO_JOINT_TEMPORAL_EXCESSIVE_JUMP_THRESHOLD_M:-0.060}"
JOINT_TEMPORAL_VIEW_WRIST_JUMP_PX="${EGO_JOINT_TEMPORAL_VIEW_WRIST_JUMP_PX:-80}"
JOINT_TEMPORAL_VIEW_EXPANDED_BOX_RATIO="${EGO_JOINT_TEMPORAL_VIEW_EXPANDED_BOX_RATIO:-2.0}"
JOINT_TEMPORAL_VIEW_EXPANDED_BOX_WRIST_PX="${EGO_JOINT_TEMPORAL_VIEW_EXPANDED_BOX_WRIST_PX:-80}"
POST_TEMPORAL_BRANCH_THRESHOLD_M="${EGO_POST_TEMPORAL_BRANCH_THRESHOLD_M:-0.030}"
POST_TEMPORAL_JOINT_REPASS="${EGO_POST_TEMPORAL_JOINT_REPASS:-1}"
TEMPORAL_HAND_IDENTITY_MAX_GAP="${EGO_TEMPORAL_HAND_IDENTITY_MAX_GAP:-60}"
TORCHINDUCTOR_CACHE="${EGO_TORCHINDUCTOR_CACHE_DIR:-/tmp/ego-hand-torchinductor}"

usage() {
  echo "Optional: --final-video also generates a final delivery video (hybrid trusted-3D/native-2D by default; disabled by default)."
  echo "Usage: $0 (--mcap FILE | --session EGO4_DIR) --output DIR [--detector FILE] [--cameras camera0 camera1 ...] [--reference-camera CAMERA] [--max-frames N] [--device cuda] [--conda-env NAME] [--gpu-profile compatible|rtx5090d] [--batch-size N] [--frame-batch-size N] [--preprocess-workers N] [--max-detections-per-class N] [--compile-backbone 0|1] [--fusion-workers N] [--glove-marker-assist 0|1] [--sam3-glove-assist 0|1] [--sam3-conda-env NAME] [--sam3-model FILE] [--sam3-prompt TEXT] [--sam3-text-only 0|1] [--sam3-refine-2d 0|1] [--sam3-max-2d-correction-px N] [--sam3-2d-correction-blend N] [--sam3-max-2d-bone-change-ratio N] [--condition-hard-threshold N] [--anatomy-refinement 0|1] [--marker-value-min 160] [--marker-saturation-max 100] [--marker-min-matches 3] [--marker-global-min-matches 8] [--marker-blend 0.15] [--detector-handedness strict|ignore|adaptive] [--final-renderer hybrid|fused] [--no-video] [--stop-after-inference]"
  echo "Defaults for glove stability: --sam3-refine-2d=0, --condition-hard-threshold=150, --anatomy-refinement=0. Override with EGO_* or the explicit flags."
  echo "Six-camera default anchors: camera0+camera5 (widest baseline); camera2+camera3 is used only when the wide pair is unavailable."
  echo "EGO4 --session uses the GEN WiLoR/fusion/temporal pipeline with four calibrated cameras; preferred anchor pair is camera0+camera1."
  echo "Optional --allow-single-hand-anchors 1 retains geometrically valid single-hand frames."
  echo "Final renderer: EGO_FINAL_RENDERER=hybrid (default) or fused; hybrid falls back to native 2D when depth is not trustworthy."
  echo "Final video env: EGO_FINAL_RENDERER (default hybrid), EGO_FINAL_HYBRID_RESIDUAL_THRESHOLD_PX (12), EGO_FINAL_HYBRID_WRIST_THRESHOLD_PX (15), EGO_FINAL_HYBRID_MAX_CONDITION (100), EGO_FINAL_HYBRID_MAX_STEREO_DIFFERENCE_MM (20), EGO_FINAL_PREDICTED_2D_FALLBACK (default 0), EGO_FINAL_REQUIRE_LOCAL_VIEW (default 1)."
}

while (($#)); do
  case "$1" in
    --no-video|--stop-after-inference|--final-video|-h|--help) ;;
    --*)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      ;;
  esac
  case "$1" in
    --mcap) MCAP="$2"; INPUT_KIND="mcap"; shift 2 ;;
    --session|--experimental-session-3d) MCAP="$2"; INPUT_KIND="session"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --detector) DETECTOR="$2"; shift 2 ;;
    --cameras)
      CAMERAS_EXPLICIT=1
      CAMERAS=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        CAMERAS+=("$1")
        shift
      done
      ;;
    --reference-camera) REFERENCE_CAMERA="$2"; REFERENCE_EXPLICIT=1; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --frame-batch-size) FRAME_BATCH_SIZE="$2"; shift 2 ;;
    --preprocess-workers) PREPROCESS_WORKERS="$2"; shift 2 ;;
    --max-detections-per-class) MAX_DETECTIONS_PER_CLASS="$2"; shift 2 ;;
    --compile-backbone) COMPILE_BACKBONE="$2"; shift 2 ;;
    --fusion-workers) FUSION_WORKERS="$2"; shift 2 ;;
    --allow-single-hand-anchors) ALLOW_SINGLE_HAND_ANCHORS="$2"; shift 2 ;;
    --glove-marker-assist|--nokov-wilor-assist) GLOVE_MARKER_ASSIST="$2"; shift 2 ;;
    --marker-saturation-max) MARKER_SATURATION_MAX="$2"; shift 2 ;;
    --marker-value-min) MARKER_VALUE_MIN="$2"; shift 2 ;;
    --marker-min-matches) MARKER_MIN_MATCHES="$2"; shift 2 ;;
    --marker-min-finger-groups) MARKER_MIN_FINGER_GROUPS="$2"; shift 2 ;;
    --marker-global-min-matches) MARKER_GLOBAL_MIN_MATCHES="$2"; shift 2 ;;
    --marker-global-min-finger-groups) MARKER_GLOBAL_MIN_FINGER_GROUPS="$2"; shift 2 ;;
    --marker-search-padding-px) MARKER_SEARCH_PADDING_PX="$2"; shift 2 ;;
    --marker-bbox-padding-px) MARKER_BBOX_PADDING_PX="$2"; shift 2 ;;
    --marker-seed-distance-px) MARKER_SEED_DISTANCE_PX="$2"; shift 2 ;;
    --marker-match-distance-px) MARKER_MATCH_DISTANCE_PX="$2"; shift 2 ;;
    --marker-max-shift-px) MARKER_MAX_SHIFT_PX="$2"; shift 2 ;;
    --marker-max-applied-shift-px) MARKER_MAX_APPLIED_SHIFT_PX="$2"; shift 2 ;;
    --marker-blend) MARKER_BLEND="$2"; shift 2 ;;
    --marker-max-local-adjustment-px) MARKER_MAX_LOCAL_ADJUSTMENT_PX="$2"; shift 2 ;;
    --sam3-glove-assist|--sam3-assist) SAM3_GLOVE_ASSIST="$2"; shift 2 ;;
    --sam3-conda-env) SAM3_CONDA_ENV="$2"; shift 2 ;;
    --sam3-tool-root) SAM3_TOOL_ROOT="$2"; shift 2 ;;
    --sam3-model) SAM3_MODEL="$2"; shift 2 ;;
    --sam3-prompt) SAM3_PROMPT="$2"; shift 2 ;;
    --sam3-device) SAM3_DEVICE="$2"; shift 2 ;;
    --sam3-conf) SAM3_CONF="$2"; shift 2 ;;
    --sam3-min-score) SAM3_MIN_SCORE="$2"; shift 2 ;;
    --sam3-min-box-iou) SAM3_MIN_BOX_IOU="$2"; shift 2 ;;
    --sam3-min-joint-support) SAM3_MIN_JOINT_SUPPORT="$2"; shift 2 ;;
    --sam3-hard-joint-support) SAM3_HARD_JOINT_SUPPORT="$2"; shift 2 ;;
    --sam3-text-only) SAM3_TEXT_ONLY="$2"; shift 2 ;;
    --sam3-refine-2d) SAM3_REFINE_2D="$2"; shift 2 ;;
    --sam3-max-2d-correction-px) SAM3_MAX_2D_CORRECTION_PX="$2"; shift 2 ;;
    --sam3-2d-correction-blend) SAM3_2D_CORRECTION_BLEND="$2"; shift 2 ;;
    --sam3-max-2d-bone-change-ratio) SAM3_MAX_2D_BONE_CHANGE_RATIO="$2"; shift 2 ;;
    --condition-hard-threshold) CONDITION_HARD_THRESHOLD="$2"; shift 2 ;;
    --anatomy-refinement) ANATOMY_REFINEMENT="$2"; shift 2 ;;
    --final-renderer) FINAL_RENDERER="$2"; shift 2 ;;
    --detector-handedness) DETECTOR_HANDEDNESS="$2"; shift 2 ;;
    --gpu-profile) GPU_PROFILE="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    --stop-after-inference) STOP_AFTER_INFERENCE=1; shift ;;
    --final-video) FINAL_VIDEO_ENABLED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$INPUT_KIND" == "session" ]]; then
  if ((CAMERAS_EXPLICIT == 0)); then CAMERAS=(camera0 camera1 camera2 camera3); fi
  if ((REFERENCE_EXPLICIT == 0)); then REFERENCE_CAMERA="camera0"; fi
  if [[ "${CAMERAS[*]}" != "camera0 camera1 camera2 camera3" ]]; then
    echo "EGO4 MP4 exports require camera0 camera1 camera2 camera3 in calibration order" >&2
    exit 2
  fi
fi

case "$GPU_PROFILE" in
  compatible)
    BATCH_SIZE="${BATCH_SIZE:-4}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-1}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-1}"
    MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-0}"
    COMPILE_BACKBONE="${COMPILE_BACKBONE:-0}"
    FUSION_WORKERS="${FUSION_WORKERS:-1}"
    ;;
  rtx5090d)
    BATCH_SIZE="${BATCH_SIZE:-64}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-16}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
    MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-1}"
    COMPILE_BACKBONE="${COMPILE_BACKBONE:-1}"
    FUSION_WORKERS="${FUSION_WORKERS:-8}"
    ;;
  *)
    echo "--gpu-profile must be compatible or rtx5090d: $GPU_PROFILE" >&2
    exit 2
    ;;
esac

if [[ -z "$MCAP" || -z "$OUTPUT" ]]; then
  usage >&2
  exit 2
fi
if ((${#CAMERAS[@]} < 2)); then
  echo "at least two cameras are required" >&2
  exit 2
fi
declare -A CAMERA_SEEN=()
for CAMERA in "${CAMERAS[@]}"; do
  [[ "$CAMERA" =~ ^camera[0-9]+$ ]] || {
    echo "invalid camera id: $CAMERA" >&2
    exit 2
  }
  [[ -z "${CAMERA_SEEN[$CAMERA]:-}" ]] || {
    echo "duplicate camera id: $CAMERA" >&2
    exit 2
  }
  CAMERA_SEEN[$CAMERA]=1
done
[[ -n "${CAMERA_SEEN[$REFERENCE_CAMERA]:-}" ]] || {
  echo "reference camera is not in the selected camera set: $REFERENCE_CAMERA" >&2
  exit 2
}
if [[ "$INPUT_KIND" == "session" && "${CAMERA_SEEN[camera1]:-}" == 1 && "${CAMERA_SEEN[camera0]:-}" == 1 ]]; then
  ANCHOR_CAMERAS=(camera0 camera1)
  ANATOMY_PROTECTED_CAMERAS=(camera0 camera1)
elif [[ "${CAMERA_SEEN[camera0]:-}" == 1 && "${CAMERA_SEEN[camera5]:-}" == 1 ]]; then
  # The six cameras are nearly collinear.  camera0/camera5 provide the
  # widest baseline (~20 cm), whereas camera2/camera3 are adjacent
  # (~5.9 cm) and make depth extremely ill-conditioned.  Prefer the wide
  # pair whenever both cameras are available; fall back to the old middle
  # pair only for reduced camera selections.
  ANCHOR_CAMERAS=(camera0 camera5)
  ANATOMY_PROTECTED_CAMERAS=(camera0 camera5)
elif [[ "${CAMERA_SEEN[camera2]:-}" == 1 && "${CAMERA_SEEN[camera3]:-}" == 1 ]]; then
  ANCHOR_CAMERAS=(camera2 camera3)
  ANATOMY_PROTECTED_CAMERAS=(camera2 camera3)
else
  ANCHOR_CAMERAS=("${CAMERAS[0]}" "${CAMERAS[1]}")
  ANATOMY_PROTECTED_CAMERAS=("$REFERENCE_CAMERA")
fi
if [[ "$INPUT_KIND" == "session" ]]; then
  [[ -d "$MCAP" ]] || { echo "EGO4 session does not exist: $MCAP" >&2; exit 2; }
else
  [[ -f "$MCAP" ]] || { echo "MCAP does not exist: $MCAP" >&2; exit 2; }
fi
if [[ ! -f "$DETECTOR" ]]; then
  echo "detector checkpoint does not exist: $DETECTOR" >&2
  exit 2
fi
if ! [[ "$GLOVE_MARKER_ASSIST" =~ ^[01]$ ]]; then
  echo "--glove-marker-assist must be 0 or 1: $GLOVE_MARKER_ASSIST" >&2
  exit 2
fi
if ! [[ "$ALLOW_SINGLE_HAND_ANCHORS" =~ ^[01]$ ]]; then
  echo "--allow-single-hand-anchors must be 0 or 1: $ALLOW_SINGLE_HAND_ANCHORS" >&2
  exit 2
fi
if ! [[ "$SAM3_GLOVE_ASSIST" =~ ^[01]$ && "$SAM3_TEXT_ONLY" =~ ^[01]$ && "$SAM3_REFINE_2D" =~ ^[01]$ && "$ANATOMY_REFINEMENT" =~ ^[01]$ ]]; then
  echo "--sam3-glove-assist/--sam3-text-only/--sam3-refine-2d/--anatomy-refinement must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$FINAL_PREDICTED_2D_FALLBACK" =~ ^[01]$ && "$FINAL_REQUIRE_LOCAL_VIEW" =~ ^[01]$ ]]; then
  echo "EGO_FINAL_PREDICTED_2D_FALLBACK/EGO_FINAL_REQUIRE_LOCAL_VIEW must be 0 or 1" >&2
  exit 2
fi
case "$FINAL_RENDERER" in
  hybrid|fused) ;;
  *)
    echo "--final-renderer/EGO_FINAL_RENDERER must be hybrid or fused: $FINAL_RENDERER" >&2
    exit 2
    ;;
esac
if ! [[ "$FINAL_HYBRID_RESIDUAL_THRESHOLD_PX" =~ ^[0-9]+([.][0-9]+)?$ && "$FINAL_HYBRID_WRIST_THRESHOLD_PX" =~ ^[0-9]+([.][0-9]+)?$ && "$FINAL_HYBRID_MAX_CONDITION" =~ ^[0-9]+([.][0-9]+)?$ && "$FINAL_HYBRID_MAX_STEREO_DIFFERENCE_MM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "hybrid final trust thresholds must be positive numbers" >&2
  exit 2
fi
for VALUE in "$SAM3_CONF" "$SAM3_MIN_SCORE" "$SAM3_MIN_BOX_IOU" "$SAM3_MIN_JOINT_SUPPORT" "$SAM3_HARD_JOINT_SUPPORT" "$SAM3_MAX_2D_CORRECTION_PX" "$SAM3_2D_CORRECTION_BLEND" "$SAM3_MAX_2D_BONE_CHANGE_RATIO"; do
  [[ "$VALUE" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "SAM3 thresholds must be non-negative decimal numbers: $VALUE" >&2
    exit 2
  }
done
python3 - "$SAM3_CONF" "$SAM3_MIN_SCORE" "$SAM3_MIN_BOX_IOU" "$SAM3_MIN_JOINT_SUPPORT" "$SAM3_HARD_JOINT_SUPPORT" "$SAM3_MAX_2D_CORRECTION_PX" "$SAM3_2D_CORRECTION_BLEND" "$SAM3_MAX_2D_BONE_CHANGE_RATIO" <<'PY'
import sys
values = [float(value) for value in sys.argv[1:]]
if any(value <= 0.0 or value > 1.0 for value in values[:1]):
    raise SystemExit("SAM3 confidence must be in (0, 1]")
if any(value < 0.0 or value > 1.0 for value in values[1:5]):
    raise SystemExit("SAM3 thresholds must be in [0, 1]")
if values[4] > values[3]:
    raise SystemExit("SAM3 hard joint support cannot exceed minimum joint support")
if values[5] < 0.0:
    raise SystemExit("SAM3 max 2D correction must be non-negative")
if not 0.0 <= values[6] <= 1.0:
    raise SystemExit("SAM3 2D correction blend must be in [0, 1]")
if values[7] < 0.0:
    raise SystemExit("SAM3 max 2D bone change ratio must be non-negative")
PY
if ((SAM3_GLOVE_ASSIST)); then
  [[ -f "$SAM3_MODEL" ]] || {
    echo "SAM3 model does not exist: $SAM3_MODEL" >&2
    exit 2
  }
  if [[ "$SAM3_DEVICE" == "auto" ]]; then
    SAM3_DEVICE="cuda"
  fi
fi
if [[ -z "$DETECTOR_HANDEDNESS" ]]; then
  if ((GLOVE_MARKER_ASSIST)); then
    DETECTOR_HANDEDNESS="adaptive"
  else
    DETECTOR_HANDEDNESS="strict"
  fi
fi
case "$DETECTOR_HANDEDNESS" in
  strict|ignore|adaptive) ;;
  *)
    echo "--detector-handedness must be strict, ignore, or adaptive: $DETECTOR_HANDEDNESS" >&2
    exit 2
    ;;
esac
if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$FRAME_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PREPROCESS_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_DETECTIONS_PER_CLASS" =~ ^[0-9]+$ && "$COMPILE_BACKBONE" =~ ^[01]$ && "$FUSION_WORKERS" =~ ^[1-9][0-9]*$ && "$MARKER_SATURATION_MAX" =~ ^[0-9]+$ && "$MARKER_VALUE_MIN" =~ ^[0-9]+$ && "$MARKER_MIN_MATCHES" =~ ^[1-9][0-9]*$ && "$MARKER_MIN_FINGER_GROUPS" =~ ^[1-9][0-9]*$ && "$MARKER_GLOBAL_MIN_MATCHES" =~ ^[1-9][0-9]*$ && "$MARKER_GLOBAL_MIN_FINGER_GROUPS" =~ ^[1-9][0-9]*$ && "$POST_TEMPORAL_JOINT_REPASS" =~ ^[01]$ ]]; then
  echo "frames/detection limit must be non-negative; batch sizes/workers must be positive" >&2
  exit 2
fi
for VALUE in "$MARKER_SEARCH_PADDING_PX" "$MARKER_BBOX_PADDING_PX" "$MARKER_SEED_DISTANCE_PX" "$MARKER_MATCH_DISTANCE_PX" "$MARKER_MAX_SHIFT_PX" "$MARKER_MAX_APPLIED_SHIFT_PX" "$MARKER_BLEND" "$MARKER_MAX_LOCAL_ADJUSTMENT_PX"; do
  [[ "$VALUE" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "marker distances/blend must be non-negative decimal numbers: $VALUE" >&2
    exit 2
  }
done

OUTPUT="$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd)"
NORMALIZED="$OUTPUT/normalized_multiview"
RAW_PREDICTIONS="$OUTPUT/wilor_multiview"
PREDICTIONS="$RAW_PREDICTIONS"
FUSION="$OUTPUT/fusion_multiview"
if ((GLOVE_MARKER_ASSIST)); then
  PREDICTIONS="$OUTPUT/wilor_multiview_glove_marker_assisted"
  FUSION="$OUTPUT/fusion_multiview_glove_marker_assisted"
elif [[ "$DETECTOR_HANDEDNESS" != "strict" ]]; then
  FUSION="$OUTPUT/fusion_multiview_geometric"
fi
MARKER_PREDICTIONS="$PREDICTIONS"
MARKER_FUSION="$FUSION"
SAM3_SOURCE_PREDICTIONS="$PREDICTIONS"
SAM3_PREDICTIONS="$OUTPUT/wilor_multiview_sam3_assisted"
SAM3_FUSION="$OUTPUT/fusion_multiview_sam3_assisted"
if ((SAM3_GLOVE_ASSIST)); then
  PREDICTIONS="$SAM3_PREDICTIONS"
  FUSION="$SAM3_FUSION"
fi
VIDEO="$FUSION/diagnostic_${#CAMERAS[@]}view.mp4"
FINAL_VIDEO="$FUSION/final_only_${#CAMERAS[@]}view.mp4"
CAMERA_CSV="$(IFS=,; echo "${CAMERAS[*]}")"
CAMERA_CONFIDENCE_ARGS=()
for CAMERA in "${CAMERAS[@]}"; do
  case "$CAMERA" in
    camera0) CONFIDENCE="0.2" ;;
    camera4|camera5) CONFIDENCE="0.1" ;;
    *) CONFIDENCE="0.3" ;;
  esac
  CAMERA_CONFIDENCE_ARGS+=(--camera-confidence "${CAMERA}=${CONFIDENCE}")
done

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE" \
    python "$@"
}

run_sam3_python() {
  conda run --no-capture-output -n "$SAM3_CONDA_ENV" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    python "$@"
}

run_python - "$OUTPUT/run_config.json" "$MCAP" "$MAX_FRAMES" "$DEVICE" "$DETECTOR" "$BATCH_SIZE" "$GPU_PROFILE" "$FRAME_BATCH_SIZE" "$PREPROCESS_WORKERS" "$MAX_DETECTIONS_PER_CLASS" "$COMPILE_BACKBONE" "$FUSION_WORKERS" "$CAMERA_CSV" "$REFERENCE_CAMERA" "${ANCHOR_CAMERAS[*]}" "$DETECTOR_HANDEDNESS" "$ALLOW_SINGLE_HAND_ANCHORS" "$GLOVE_MARKER_ASSIST" "$MARKER_SATURATION_MAX" "$MARKER_VALUE_MIN" "$MARKER_MIN_MATCHES" "$MARKER_MIN_FINGER_GROUPS" "$MARKER_GLOBAL_MIN_MATCHES" "$MARKER_GLOBAL_MIN_FINGER_GROUPS" "$MARKER_SEARCH_PADDING_PX" "$MARKER_BBOX_PADDING_PX" "$MARKER_SEED_DISTANCE_PX" "$MARKER_MATCH_DISTANCE_PX" "$MARKER_MAX_SHIFT_PX" "$MARKER_MAX_APPLIED_SHIFT_PX" "$MARKER_BLEND" "$MARKER_MAX_LOCAL_ADJUSTMENT_PX" "$SAM3_GLOVE_ASSIST" "$SAM3_CONDA_ENV" "$SAM3_TOOL_ROOT" "$SAM3_MODEL" "$SAM3_PROMPT" "$SAM3_DEVICE" "$SAM3_CONF" "$SAM3_MIN_SCORE" "$SAM3_MIN_BOX_IOU" "$SAM3_MIN_JOINT_SUPPORT" "$SAM3_HARD_JOINT_SUPPORT" "$SAM3_TEXT_ONLY" "$SAM3_REFINE_2D" "$SAM3_MAX_2D_CORRECTION_PX" "$SAM3_2D_CORRECTION_BLEND" "$SAM3_MAX_2D_BONE_CHANGE_RATIO" "$CONDITION_HARD_THRESHOLD" "$ANATOMY_REFINEMENT" "$JOINT_TEMPORAL_MIN_HAND_JOINTS" "$JOINT_TEMPORAL_EXCESSIVE_JUMP_THRESHOLD_M" "$JOINT_TEMPORAL_VIEW_WRIST_JUMP_PX" "$JOINT_TEMPORAL_VIEW_EXPANDED_BOX_RATIO" "$JOINT_TEMPORAL_VIEW_EXPANDED_BOX_WRIST_PX" "$TEMPORAL_HAND_IDENTITY_MAX_GAP" <<'PY'
import json
from pathlib import Path
import sys

from glove_marker_assist import MarkerAssistConfig

config_path, source_text, max_frames, device, detector_text, batch_size, gpu_profile, frame_batch_size, preprocess_workers, max_detections_per_class, compile_backbone, fusion_workers, camera_csv, reference_camera, anchor_cameras_text, detector_handedness, allow_single_hand_anchors, marker_enabled, marker_saturation_max, marker_value_min, marker_min_matches, marker_min_finger_groups, marker_global_min_matches, marker_global_min_finger_groups, marker_search_padding_px, marker_bbox_padding_px, marker_seed_distance_px, marker_match_distance_px, marker_max_shift_px, marker_max_applied_shift_px, marker_blend, marker_max_local_adjustment_px, sam3_enabled, sam3_conda_env, sam3_tool_root, sam3_model, sam3_prompt, sam3_device, sam3_conf, sam3_min_score, sam3_min_box_iou, sam3_min_joint_support, sam3_hard_joint_support, sam3_text_only, sam3_refine_2d, sam3_max_2d_correction_px, sam3_2d_correction_blend, sam3_max_2d_bone_change_ratio, condition_hard_threshold, anatomy_refinement, joint_temporal_min_hand_joints, joint_temporal_excessive_jump_threshold_m, joint_temporal_view_wrist_jump_px, joint_temporal_view_expanded_box_ratio, joint_temporal_view_expanded_box_wrist_px, temporal_hand_identity_max_gap = sys.argv[1:]
source = Path(source_text).resolve()
detector = Path(detector_text).resolve()
stat = source.stat()
camera_ids = camera_csv.split(",")
marker_config = MarkerAssistConfig(
    saturation_max=int(marker_saturation_max),
    value_min=int(marker_value_min),
    min_matches=int(marker_min_matches),
    min_finger_groups=int(marker_min_finger_groups),
    global_min_matches=int(marker_global_min_matches),
    global_min_finger_groups=int(marker_global_min_finger_groups),
    search_padding_px=float(marker_search_padding_px),
    bbox_padding_px=float(marker_bbox_padding_px),
    seed_distance_px=float(marker_seed_distance_px),
    match_distance_px=float(marker_match_distance_px),
    max_shift_px=float(marker_max_shift_px),
    max_applied_shift_px=float(marker_max_applied_shift_px),
    marker_blend=float(marker_blend),
    max_local_adjustment_px=float(marker_max_local_adjustment_px),
)
marker_config.validate()
config = {
    "mcap": {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    "cameras": camera_ids,
    "reference_camera": reference_camera,
    "anchor_cameras": anchor_cameras_text.split(),
    "max_frames": int(max_frames),
    "device": device,
    "detector": str(detector),
    "batch_size": int(batch_size),
    "gpu_profile": gpu_profile,
    "frame_batch_size": int(frame_batch_size),
    "preprocess_workers": int(preprocess_workers),
    "max_detections_per_class": int(max_detections_per_class),
    "compile_backbone": bool(int(compile_backbone)),
    "fusion_workers": int(fusion_workers),
    "detector_handedness": detector_handedness,
    "allow_single_hand_anchors": bool(int(allow_single_hand_anchors)),
    "glove_marker_assist": {"enabled": bool(int(marker_enabled)), **marker_config.to_dict()},
    "sam3_glove_assist": {
        "enabled": bool(int(sam3_enabled)),
        "conda_env": sam3_conda_env,
        "tool_root": str(Path(sam3_tool_root).resolve()),
        "model": str(Path(sam3_model).resolve()),
        "prompt": sam3_prompt,
        "device": sam3_device,
        "conf": float(sam3_conf),
        "min_score": float(sam3_min_score),
        "min_box_iou": float(sam3_min_box_iou),
        "min_joint_support": float(sam3_min_joint_support),
        "hard_joint_support": float(sam3_hard_joint_support),
        "text_only": bool(int(sam3_text_only)),
        "refine_2d": bool(int(sam3_refine_2d)),
        "max_2d_correction_px": float(sam3_max_2d_correction_px),
        "two_d_correction_blend": float(sam3_2d_correction_blend),
        "max_2d_bone_change_ratio": float(sam3_max_2d_bone_change_ratio),
    },
    "camera_confidences": {"camera0": 0.2, "camera1": 0.3, "camera2": 0.3,
                           "camera3": 0.3, "camera4": 0.1, "camera5": 0.1},
    "fusion_algorithm": "anchor_guided_condition_rejected_joint_temporal_v28_branch_guard_cam2_sam3_angle_weighted",
    "fusion_stability": {
        "suppress_duplicate_detector_boxes": True,
        "duplicate_detector_iou": 0.95,
        "duplicate_detector_containment": 0.95,
        "temporal_3d_guard": False,
        "temporal_3d_guard_radius": 24,
        "temporal_3d_guard_threshold_m": 0.020,
        "temporal_3d_guard_min_anchor_cameras": 5,
        "temporal_3d_guard_min_palm_inliers": 3,
        "temporal_3d_guard_max_anchor_quality_px": 12.0,
        "temporal_3d_guard_max_anchor_speed_m": 0.050,
        "temporal_3d_guard_max_local_reprojection_ratio": 1.5,
        "temporal_3d_guard_clear_endpoint": True,
        "joint_temporal_guard": True,
        "joint_temporal_ray_prior_weight": 3.0,
        "joint_temporal_ray_blend": 0.80,
        "joint_temporal_max_step_m": 0.025,
        "joint_temporal_max_radius_m": 0.55,
        "joint_temporal_max_missing_gap": 2,
        "joint_temporal_local_radius": 3,
        "joint_temporal_local_outlier_threshold_m": 0.025,
        "joint_temporal_local_candidate_ratio": 0.82,
        "joint_temporal_smoothing_sigma": 1.30,
        "joint_temporal_bone_iterations": 15,
        "joint_temporal_bone_strength": 0.80,
        "condition_hard_threshold": float(condition_hard_threshold),
        "anatomy_refinement": bool(int(anatomy_refinement)),
        "min_joint_inlier_views": 3,
        "allow_stereo_final": False,
        "temporal_hand_identity": True,
        "temporal_hand_identity_max_gap": int(temporal_hand_identity_max_gap),
        "temporal_hand_identity_swap_margin": 0.25,
        "temporal_hand_identity_swap_min_cost": 0.90,
        "joint_temporal_ray_conditioning_adaptation": True,
        "joint_temporal_ray_conditioning_hard_threshold": 1000.0,
        "joint_temporal_ray_angle_weighting": True,
        "joint_temporal_ray_angle_reference_degrees": 25.0,
        "joint_temporal_ray_angle_weight_floor": 0.20,
        "joint_temporal_sam3_ray_quality_weighting": True,
        "joint_temporal_isolated_step_spike_guard": True,
        "joint_temporal_final_continuity_branch_repair_threshold_m": 0.025,
        "joint_temporal_final_continuity_hard_cap": False,
        "joint_temporal_gap_aware_runs": True,
        "joint_temporal_boundary_tail_guard": True,
        "joint_temporal_boundary_tail_threshold_m": 0.080,
        "joint_temporal_excessive_jump_visibility_threshold_m": float(joint_temporal_excessive_jump_threshold_m),
        "joint_temporal_min_hand_joints": int(joint_temporal_min_hand_joints),
        "joint_temporal_view_wrist_jump_px": float(joint_temporal_view_wrist_jump_px),
        "joint_temporal_view_expanded_box_ratio": float(joint_temporal_view_expanded_box_ratio),
        "joint_temporal_view_expanded_box_wrist_px": float(joint_temporal_view_expanded_box_wrist_px),
        "anatomy_protected_cameras": [reference_camera],
        "anatomy_protected_camera_shift_px": 35.0,
        "post_temporal_target_camera_branch_guard": True,
        "post_temporal_target_camera_branch_threshold_m": 0.050,
        "post_temporal_target_camera_branch_view_motion_threshold_px": 60.0,
        "post_temporal_target_camera_branch_max_checked_gap_frames": 3,
    },
}
if source.is_dir():
    config["session"] = config.pop("mcap")
    config["camera_confidences"] = {
        camera: config["camera_confidences"][camera] for camera in camera_ids
    }
path = Path(config_path)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    # Outputs made before GPU profiles were introduced used this exact path.
    previous.setdefault("gpu_profile", "compatible")
    previous.setdefault("frame_batch_size", 1)
    previous.setdefault(
        "preprocess_workers", 8 if previous["gpu_profile"] == "rtx5090d" else 1
    )
    previous.setdefault("max_detections_per_class", 0)
    previous.setdefault("compile_backbone", False)
    previous.setdefault(
        "fusion_workers", 8 if previous["gpu_profile"] == "rtx5090d" else 1
    )
    previous.setdefault(
        "anchor_cameras",
        ["camera2", "camera3"]
        if {"camera2", "camera3"}.issubset(previous["cameras"])
        else previous["cameras"][:2],
    )
    previous.setdefault("detector_handedness", "strict")
    previous.setdefault("allow_single_hand_anchors", False)
    previous.setdefault("detector", str(detector))
    previous.setdefault(
        "glove_marker_assist",
        {"enabled": False, **MarkerAssistConfig().to_dict()},
    )
    previous.setdefault(
        "sam3_glove_assist",
        {
            "enabled": False,
            "conda_env": sam3_conda_env,
            "tool_root": str(Path(sam3_tool_root).resolve()),
            "model": str(Path(sam3_model).resolve()),
            "prompt": sam3_prompt,
            "device": sam3_device,
            "conf": float(sam3_conf),
            "min_score": float(sam3_min_score),
            "min_box_iou": float(sam3_min_box_iou),
            "min_joint_support": float(sam3_min_joint_support),
            "hard_joint_support": float(sam3_hard_joint_support),
            "text_only": bool(int(sam3_text_only)),
            "refine_2d": bool(int(sam3_refine_2d)),
            "max_2d_correction_px": float(sam3_max_2d_correction_px),
            "two_d_correction_blend": float(sam3_2d_correction_blend),
            "max_2d_bone_change_ratio": float(sam3_max_2d_bone_change_ratio),
        },
    )
    if previous.get("fusion_algorithm") == "anchor_guided_dynamic_temporal_handedness_v3":
        previous["fusion_algorithm"] = "anchor_guided_dynamic_temporal_marker_v4"
    if previous != config:
        changes = []
        for key in sorted(set(previous) | set(config)):
            old, new = previous.get(key), config.get(key)
            if old == new:
                continue
            if isinstance(old, dict) and isinstance(new, dict):
                fields = sorted(
                    field for field in set(old) | set(new)
                    if old.get(field) != new.get(field)
                )
                changes.append(f"{key}: {', '.join(fields)}")
            else:
                changes.append(f"{key}: previous={old!r}, current={new!r}")
        raise SystemExit(
            "run configuration differs; use a new --output directory to avoid "
            "mixing results from different settings:\n  " + "\n  ".join(changes)
        )
else:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

require_complete_or_absent() {
  local directory="$1"
  local marker="$2"
  if [[ -d "$directory" && ! -f "$directory/$marker" ]]; then
    echo "Incomplete stage exists: $directory (missing $marker). Keep it for diagnosis and use a new output directory." >&2
    exit 2
  fi
}

require_complete_or_absent "$NORMALIZED" manifest.json
require_complete_or_absent "$RAW_PREDICTIONS" summary.json
if ((GLOVE_MARKER_ASSIST)); then
  require_complete_or_absent "$MARKER_PREDICTIONS" summary.json
fi
if ((SAM3_GLOVE_ASSIST)); then
  require_complete_or_absent "$SAM3_PREDICTIONS" summary.json
fi
require_complete_or_absent "$FUSION" summary.json

if [[ ! -f "$NORMALIZED/manifest.json" ]]; then
  echo "[multiview] normalize and synchronize ${CAMERAS[*]}"
  if [[ "$INPUT_KIND" == "session" ]]; then
    run_python "$ROOT/scripts/normalize_orbbec_ego4_session.py" \
      --session "$MCAP" --output "$NORMALIZED" \
      --reference-camera "$REFERENCE_CAMERA" --max-delta-us 1500 --max-frames "$MAX_FRAMES"
  else
    run_python "$ROOT/scripts/normalize_multiview_recording.py" \
      --input "$MCAP" --output "$NORMALIZED" \
      --cameras "${CAMERAS[@]}" \
      --reference-camera "$REFERENCE_CAMERA" --max-delta-us 1500 --max-frames "$MAX_FRAMES"
  fi
else
  echo "[multiview] reuse normalized dataset"
fi

if [[ ! -f "$RAW_PREDICTIONS/summary.json" ]]; then
  echo "[multiview] run ${#CAMERAS[@]}-view dual-hypothesis WiLoR"
  INFERENCE_ARGS=(
    --dataset "$NORMALIZED" --output "$RAW_PREDICTIONS"
    --detector "$DETECTOR"
    --device "$DEVICE" --gpu-profile "$GPU_PROFILE"
    --batch-size "$BATCH_SIZE" --frame-batch-size "$FRAME_BATCH_SIZE"
    --preprocess-workers "$PREPROCESS_WORKERS"
    --max-detections-per-class "$MAX_DETECTIONS_PER_CLASS"
    --compile-backbone "$COMPILE_BACKBONE"
    --max-frames "$MAX_FRAMES"
    "${CAMERA_CONFIDENCE_ARGS[@]}"
  )
  run_python "$ROOT/scripts/wilor_multiview_inference.py" "${INFERENCE_ARGS[@]}"
else
  echo "[multiview] reuse WiLoR predictions"
fi

if ((STOP_AFTER_INFERENCE)); then
  echo "[multiview] inference checkpoint complete: $RAW_PREDICTIONS"
  exit 0
fi

if ((GLOVE_MARKER_ASSIST)) && [[ ! -f "$PREDICTIONS/summary.json" ]]; then
  echo "[multiview] refine WiLoR image joints with visible glove markers"
  run_python "$ROOT/scripts/assist_wilor_with_glove_markers.py" \
    --dataset "$NORMALIZED" --predictions "$RAW_PREDICTIONS" --output "$PREDICTIONS" \
    --cameras "${CAMERAS[@]}" --max-frames "$MAX_FRAMES" \
    --saturation-max "$MARKER_SATURATION_MAX" --value-min "$MARKER_VALUE_MIN" \
    --min-matches "$MARKER_MIN_MATCHES" \
    --min-finger-groups "$MARKER_MIN_FINGER_GROUPS" \
    --global-min-matches "$MARKER_GLOBAL_MIN_MATCHES" \
    --global-min-finger-groups "$MARKER_GLOBAL_MIN_FINGER_GROUPS" \
    --search-padding-px "$MARKER_SEARCH_PADDING_PX" \
    --bbox-padding-px "$MARKER_BBOX_PADDING_PX" \
    --seed-distance-px "$MARKER_SEED_DISTANCE_PX" \
    --match-distance-px "$MARKER_MATCH_DISTANCE_PX" \
    --max-shift-px "$MARKER_MAX_SHIFT_PX" \
    --max-applied-shift-px "$MARKER_MAX_APPLIED_SHIFT_PX" \
    --marker-blend "$MARKER_BLEND" \
    --max-local-adjustment-px "$MARKER_MAX_LOCAL_ADJUSTMENT_PX"
elif ((GLOVE_MARKER_ASSIST)); then
  echo "[multiview] reuse glove-marker-assisted WiLoR predictions"
fi

if ((SAM3_GLOVE_ASSIST)) && [[ ! -f "$SAM3_PREDICTIONS/summary.json" ]]; then
  echo "[multiview] validate WiLoR detections and 2D hypotheses with SAM3 glove masks"
  SAM3_ARGS=(
    "$ROOT/scripts/sam3_glove_assist.py"
    --dataset "$NORMALIZED"
    --predictions "$SAM3_SOURCE_PREDICTIONS"
    --output "$SAM3_PREDICTIONS"
    --cameras "${CAMERAS[@]}"
    --tool-root "$SAM3_TOOL_ROOT"
    --model "$SAM3_MODEL"
    --prompt "$SAM3_PROMPT"
    --device "$SAM3_DEVICE"
    --conf "$SAM3_CONF"
    --min-score "$SAM3_MIN_SCORE"
    --min-box-iou "$SAM3_MIN_BOX_IOU"
    --min-joint-support "$SAM3_MIN_JOINT_SUPPORT"
    --refine-2d "$SAM3_REFINE_2D"
    --max-2d-correction-px "$SAM3_MAX_2D_CORRECTION_PX"
    --2d-correction-blend "$SAM3_2D_CORRECTION_BLEND"
    --max-2d-bone-change-ratio "$SAM3_MAX_2D_BONE_CHANGE_RATIO"
    --max-frames "$MAX_FRAMES"
  )
  ((SAM3_TEXT_ONLY)) && SAM3_ARGS+=(--text-only)
  run_sam3_python "${SAM3_ARGS[@]}"
elif ((SAM3_GLOVE_ASSIST)); then
  echo "[multiview] reuse SAM3-assisted WiLoR predictions"
fi

if [[ ! -f "$FUSION/summary.json" ]]; then
  if [[ "$INPUT_KIND" == "session" ]]; then
    echo "[multiview] fuse calibrated KB rays with RANSAC"
  else
    echo "[multiview] fuse native Double-Sphere rays with RANSAC"
  fi
  FUSION_ARGS=(
    --dataset "$NORMALIZED"
    --predictions "$PREDICTIONS"
    --output "$FUSION"
    --cameras "${CAMERAS[@]}"
    --anchor-cameras "${ANCHOR_CAMERAS[@]}"
    --detector-handedness "$DETECTOR_HANDEDNESS"
    --workers "$FUSION_WORKERS"
    --max-frames "$MAX_FRAMES"
    --sam3-gate "$SAM3_GLOVE_ASSIST"
    --sam3-min-score "$SAM3_MIN_SCORE"
    --sam3-min-box-iou "$SAM3_MIN_BOX_IOU"
    --sam3-min-joint-support "$SAM3_MIN_JOINT_SUPPORT"
    --sam3-hard-joint-support "$SAM3_HARD_JOINT_SUPPORT"
    --condition-hard-threshold "$CONDITION_HARD_THRESHOLD"
    --anatomy-refinement "$ANATOMY_REFINEMENT"
    --min-joint-inlier-views 3
    --allow-stereo-final 0
    --allow-single-hand-anchors "$ALLOW_SINGLE_HAND_ANCHORS"
    --suppress-duplicate-detector-boxes 1
    --duplicate-detector-iou 0.95
    --duplicate-detector-containment 0.95
    --temporal-3d-guard 0
    --temporal-3d-guard-radius 24
    --temporal-3d-guard-threshold-m 0.020
    --temporal-3d-guard-min-anchor-cameras 5
    --temporal-3d-guard-min-palm-inliers 3
    --temporal-3d-guard-max-anchor-quality-px 12.0
    --temporal-3d-guard-max-anchor-speed-m 0.050
    --temporal-3d-guard-max-local-reprojection-ratio 1.5
    --temporal-3d-guard-clear-endpoint 1
    --joint-temporal-guard 1
    --joint-temporal-ray-prior-weight 3.0
    --joint-temporal-ray-blend 0.80
    --joint-temporal-max-step-m 0.025
    --joint-temporal-max-radius-m 0.55
    --joint-temporal-max-missing-gap 2
    --joint-temporal-local-radius 3
    --joint-temporal-local-outlier-threshold-m 0.025
    --joint-temporal-local-candidate-ratio 0.82
    --joint-temporal-smoothing-sigma 1.30
    --joint-temporal-bone-iterations 15
    --joint-temporal-bone-strength 0.80
    --joint-temporal-min-hand-joints "$JOINT_TEMPORAL_MIN_HAND_JOINTS"
    --joint-temporal-excessive-jump-threshold-m "$JOINT_TEMPORAL_EXCESSIVE_JUMP_THRESHOLD_M"
    --joint-temporal-view-wrist-jump-px "$JOINT_TEMPORAL_VIEW_WRIST_JUMP_PX"
    --joint-temporal-view-expanded-box-ratio "$JOINT_TEMPORAL_VIEW_EXPANDED_BOX_RATIO"
    --joint-temporal-view-expanded-box-wrist-px "$JOINT_TEMPORAL_VIEW_EXPANDED_BOX_WRIST_PX"
    --anatomy-protected-cameras "${ANATOMY_PROTECTED_CAMERAS[@]}"
    --anatomy-protected-camera-shift-px 35.0
    --post-temporal-branch-threshold-m "$POST_TEMPORAL_BRANCH_THRESHOLD_M"
    --post-temporal-joint-repass "$POST_TEMPORAL_JOINT_REPASS"
    --temporal-hand-identity 1
    --temporal-hand-identity-max-gap "$TEMPORAL_HAND_IDENTITY_MAX_GAP"
    --temporal-hand-identity-swap-margin 0.25
    --temporal-hand-identity-swap-min-cost 0.90
  )
  run_python "$ROOT/scripts/fuse_multiview_wilor_guided.py" "${FUSION_ARGS[@]}"
else
  echo "[multiview] reuse fused result"
fi

if [[ "$NO_VIDEO" == 0 && ! -f "$VIDEO" ]]; then
  echo "[multiview] render ${#CAMERAS[@]}-view diagnostic video"
  run_python "$ROOT/scripts/render_multiview_wilor.py" \
    --dataset "$NORMALIZED" --fusion "$FUSION" \
    --output "$VIDEO" --cameras "${CAMERAS[@]}" \
    --video-codec auto --max-frames "$MAX_FRAMES"
fi

if [[ "$NO_VIDEO" == 0 && "$FINAL_VIDEO_ENABLED" == 1 && ! -f "$FINAL_VIDEO" ]]; then
  if [[ "$FINAL_RENDERER" == "hybrid" ]]; then
    echo "[multiview] render conservative trusted-3D/native-2D final video"
    run_python "$ROOT/scripts/render_hybrid_multiview.py" \
      --dataset "$NORMALIZED" --fusion "$FUSION" --predictions "$PREDICTIONS" \
      --output "$FINAL_VIDEO" --cameras "${CAMERAS[@]}" --video-codec auto \
      --max-frames "$MAX_FRAMES" --physical-side-swap 0 \
      --min-joint-inlier-views "$FINAL_MIN_JOINT_INLIER_VIEWS" \
      --residual-percentile 75 --residual-threshold-px "$FINAL_HYBRID_RESIDUAL_THRESHOLD_PX" \
      --wrist-threshold-px "$FINAL_HYBRID_WRIST_THRESHOLD_PX" \
      --max-condition-number "$FINAL_HYBRID_MAX_CONDITION" \
      --max-stereo-difference-mm "$FINAL_HYBRID_MAX_STEREO_DIFFERENCE_MM" \
      --smoothing-radius 1 --smoothing-strength 0.25 \
      --wrist-outlier-threshold-px 150 --max-hold-frames 2
  else
    echo "[multiview] render final fused 3-D video"
    run_python "$ROOT/scripts/render_multiview_wilor.py" \
      --dataset "$NORMALIZED" --fusion "$FUSION" \
      --output "$FINAL_VIDEO" --cameras "${CAMERAS[@]}" \
      --overlay-mode final-only --video-codec auto --max-frames "$MAX_FRAMES" \
      --final-predicted-2d-fallback "$FINAL_PREDICTED_2D_FALLBACK" \
      --final-2d-residual-fallback-px "$FINAL_2D_RESIDUAL_FALLBACK_PX" \
      --final-2d-residual-percentile "$FINAL_2D_RESIDUAL_PERCENTILE" \
      --final-wrist-residual-fallback-px "$FINAL_WRIST_RESIDUAL_FALLBACK_PX" \
      --final-min-joint-inlier-views "$FINAL_MIN_JOINT_INLIER_VIEWS" \
      --final-require-local-view "$FINAL_REQUIRE_LOCAL_VIEW"
  fi
fi

echo "[multiview] finished: $FUSION"
cat "$FUSION/summary.json"
