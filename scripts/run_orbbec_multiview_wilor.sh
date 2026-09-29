#!/usr/bin/env bash
set -euo pipefail

# EGO Pro four-camera MCAP or four-MP4 export -> raw WiLoR 2-D review.  This baseline does
# not run SAM3, triangulation, 3-D fusion, temporal correction, or MANO fit.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
GPU_PROFILE="rtx5090d"
DETECTOR="${EGO_WILOR_DETECTOR:-$ROOT/models/wilor/detector.pt}"
CHECKPOINT="${EGO_WILOR_CHECKPOINT:-$ROOT/models/wilor/wilor_final.ckpt}"
MODEL_CONFIG="${EGO_WILOR_MODEL_CONFIG:-$ROOT/models/wilor/model_config.yaml}"
MANO_DIR="${EGO_WILOR_MANO_DIR:-$ROOT/models/mano}"
CAMERAS=(camera0 camera1 camera2 camera3)
REFERENCE_CAMERA="camera0"
MAX_FRAMES=0
BATCH_SIZE=""
FRAME_BATCH_SIZE=""
PREPROCESS_WORKERS=""
MAX_DETECTIONS_PER_CLASS=""
COMPILE_BACKBONE=""
CONFIDENCE="0.3"
IOU="0.5"
RESCALE_FACTOR="2.0"
MAX_DELTA_US="1500"
CALIBRATION_DIR=""
VIDEO_CODEC="auto"
TILE_WIDTH="480"
NO_VIDEO=0
OPTIMIZED_VIDEO=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_orbbec_multiview_wilor.sh --input MCAP_OR_EGO4_EXPORT_DIR --output DIR [options]

The review video is DIR/fusion_multiview/raw_wilor_2d_4view.mp4. It contains
only native camera images, WiLoR detector boxes, and projected 21-point 2-D
skeletons. There is no SAM3, triangulation, 3-D fusion, or temporal correction.

Options:
  --conda-env NAME, --device auto|cuda|cpu, --gpu-profile compatible|rtx5090d
  --cameras camera0 camera1 ...  --reference-camera CAMERA
  --max-frames N (0 means all)  --batch-size N --frame-batch-size N
  --preprocess-workers N --max-detections-per-class N --compile-backbone 0|1
  --detector FILE --checkpoint FILE --model-config FILE --mano-model-dir DIR
  --confidence FLOAT --iou FLOAT --rescale-factor FLOAT --max-delta-us N
  --calibration-dir DIR (optional camera*.json files) --video-codec auto|h264_nvenc|mp4v
  --tile-width N --no-video --optimized-video
EOF
}

while (($#)); do
  case "$1" in
    --input|--mcap|--session) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --gpu-profile) GPU_PROFILE="$2"; shift 2 ;;
    --detector) DETECTOR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --mano-model-dir) MANO_DIR="$2"; shift 2 ;;
    --cameras)
      CAMERAS=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        CAMERAS+=("$1")
        shift
      done
      ;;
    --reference-camera) REFERENCE_CAMERA="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --frame-batch-size) FRAME_BATCH_SIZE="$2"; shift 2 ;;
    --preprocess-workers) PREPROCESS_WORKERS="$2"; shift 2 ;;
    --max-detections-per-class) MAX_DETECTIONS_PER_CLASS="$2"; shift 2 ;;
    --compile-backbone) COMPILE_BACKBONE="$2"; shift 2 ;;
    --confidence) CONFIDENCE="$2"; shift 2 ;;
    --iou) IOU="$2"; shift 2 ;;
    --rescale-factor) RESCALE_FACTOR="$2"; shift 2 ;;
    --max-delta-us) MAX_DELTA_US="$2"; shift 2 ;;
    --calibration-dir) CALIBRATION_DIR="$2"; shift 2 ;;
    --video-codec) VIDEO_CODEC="$2"; shift 2 ;;
    --tile-width) TILE_WIDTH="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    --optimized-video) OPTIMIZED_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then usage >&2; exit 2; fi
if [[ ! -e "$INPUT" ]]; then echo "input does not exist: $INPUT" >&2; exit 2; fi
if [[ ! -f "$DETECTOR" ]]; then echo "detector does not exist: $DETECTOR" >&2; exit 2; fi
if [[ ! -f "$CHECKPOINT" ]]; then echo "WiLoR checkpoint does not exist: $CHECKPOINT" >&2; exit 2; fi
if [[ ! -f "$MODEL_CONFIG" ]]; then echo "WiLoR config does not exist: $MODEL_CONFIG" >&2; exit 2; fi
if [[ ! -d "$MANO_DIR" ]]; then echo "MANO directory does not exist: $MANO_DIR" >&2; exit 2; fi
if ((${#CAMERAS[@]} < 2)); then echo "at least two cameras are required" >&2; exit 2; fi
if [[ ! "$MAX_FRAMES" =~ ^[0-9]+$ || ! "$MAX_DELTA_US" =~ ^[0-9]+$ ]]; then
  echo "max-frames/max-delta-us must be non-negative integers" >&2; exit 2
fi
case "$GPU_PROFILE" in compatible|rtx5090d) ;; *) echo "invalid --gpu-profile: $GPU_PROFILE" >&2; exit 2 ;; esac
case "$VIDEO_CODEC" in auto|h264_nvenc|mp4v) ;; *) echo "invalid --video-codec: $VIDEO_CODEC" >&2; exit 2 ;; esac
declare -A CAMERA_SEEN=()
for CAMERA in "${CAMERAS[@]}"; do
  [[ "$CAMERA" =~ ^camera[0-9]+$ ]] || { echo "invalid camera id: $CAMERA" >&2; exit 2; }
  [[ -z "${CAMERA_SEEN[$CAMERA]:-}" ]] || { echo "duplicate camera: $CAMERA" >&2; exit 2; }
  CAMERA_SEEN[$CAMERA]=1
done
[[ -n "${CAMERA_SEEN[$REFERENCE_CAMERA]:-}" ]] || {
  echo "reference camera is not selected: $REFERENCE_CAMERA" >&2; exit 2
}
if [[ -n "$CALIBRATION_DIR" && ! -d "$CALIBRATION_DIR" ]]; then
  echo "calibration directory does not exist: $CALIBRATION_DIR" >&2; exit 2
fi

case "$GPU_PROFILE" in
  compatible)
    BATCH_SIZE="${BATCH_SIZE:-4}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-1}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-1}"
    MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-0}"
    COMPILE_BACKBONE="${COMPILE_BACKBONE:-0}"
    ;;
  rtx5090d)
    BATCH_SIZE="${BATCH_SIZE:-64}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-16}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
    MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-1}"
    COMPILE_BACKBONE="${COMPILE_BACKBONE:-1}"
    ;;
esac
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$FRAME_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PREPROCESS_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_DETECTIONS_PER_CLASS" =~ ^[0-9]+$ && "$COMPILE_BACKBONE" =~ ^[01]$ ]]; then
  echo "invalid batch sizes/workers/detection limit/compile flag" >&2; exit 2
fi

OUTPUT="$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd)"
if [[ -f "$OUTPUT/run_config.json" ]]; then
  python3 - "$OUTPUT/run_config.json" "$INPUT" "$MAX_FRAMES" "${CAMERAS[*]}" <<'PY'
import json
from pathlib import Path
import sys

config_path, input_path, max_frames, cameras = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
expected = {
    "stage": "orbbec_ego_pro_four_view_raw_wilor",
    "input": str(Path(input_path).expanduser().resolve()),
    "max_frames": int(max_frames),
    "camera_ids": cameras.split(),
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"output belongs to a different run ({key}: {config.get(key)!r}); "
            "choose a new --output directory to avoid reusing incompatible results"
        )
PY
fi
NORMALIZED="$OUTPUT/normalized_multiview"
RAW_PREDICTIONS="$OUTPUT/wilor_multiview"
VIDEO_ROOT="$OUTPUT/fusion_multiview"
VIDEO="$VIDEO_ROOT/raw_wilor_2d_4view.mp4"
OPTIMIZED_VIDEO_PATH="$VIDEO_ROOT/optimized_wilor_2d_dedup_quality15_4view.mp4"
mkdir -p "$VIDEO_ROOT"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    python "$@"
}

CAMERA_ARGS=(--cameras "${CAMERAS[@]}" --reference-camera "$REFERENCE_CAMERA")
if [[ ! -f "$NORMALIZED/manifest.json" ]]; then
  [[ ! -e "$NORMALIZED" ]] || { echo "normalized output is incomplete: $NORMALIZED" >&2; exit 2; }
  if [[ -d "$INPUT" ]] && compgen -G "$INPUT/*_calibration_camera.yaml" > /dev/null; then
    [[ "${CAMERAS[*]}" == "camera0 camera1 camera2 camera3" ]] || {
      echo "EGO4 MP4 exports require camera0 camera1 camera2 camera3" >&2; exit 2;
    }
    echo "[orbbec] normalize four-camera MP4 export"
    run_python "$ROOT/scripts/normalize_orbbec_ego4_session.py" \
      --session "$INPUT" --output "$NORMALIZED" \
      --reference-camera "$REFERENCE_CAMERA" --max-delta-us "$MAX_DELTA_US" \
      --max-frames "$MAX_FRAMES"
  else
    NORMALIZE_ARGS=(--input "$INPUT" --output "$NORMALIZED" "${CAMERA_ARGS[@]}" --max-delta-us "$MAX_DELTA_US")
    if [[ -n "$CALIBRATION_DIR" ]]; then NORMALIZE_ARGS+=(--calibration-dir "$CALIBRATION_DIR"); fi
    echo "[orbbec] normalize four-camera MCAP"
    run_python "$ROOT/scripts/normalize_orbbec_multiview_mcap.py" "${NORMALIZE_ARGS[@]}"
  fi
else
  echo "[orbbec] reuse normalized dataset: $NORMALIZED"
fi

if [[ ! -f "$RAW_PREDICTIONS/summary.json" ]]; then
  [[ ! -e "$RAW_PREDICTIONS" ]] || { echo "WiLoR output is incomplete: $RAW_PREDICTIONS" >&2; exit 2; }
  INFER_ARGS=(
    --dataset "$NORMALIZED" --output "$RAW_PREDICTIONS"
    --checkpoint "$CHECKPOINT" --model-config "$MODEL_CONFIG" --detector "$DETECTOR"
    --mano-model-dir "$MANO_DIR" --device "$DEVICE" --gpu-profile "$GPU_PROFILE"
    --batch-size "$BATCH_SIZE" --frame-batch-size "$FRAME_BATCH_SIZE"
    --preprocess-workers "$PREPROCESS_WORKERS" --max-detections-per-class "$MAX_DETECTIONS_PER_CLASS"
    --compile-backbone "$COMPILE_BACKBONE" --confidence "$CONFIDENCE"
    --iou "$IOU" --rescale-factor "$RESCALE_FACTOR" --max-frames "$MAX_FRAMES"
  )
  echo "[orbbec] raw WiLoR dual-hypothesis inference (no SAM3/fusion/3-D)"
  run_python "$ROOT/scripts/wilor_multiview_inference.py" "${INFER_ARGS[@]}"
else
  echo "[orbbec] reuse WiLoR predictions: $RAW_PREDICTIONS"
fi

if ((NO_VIDEO == 0)); then
  if [[ ! -f "$VIDEO" ]]; then
    echo "[orbbec] render raw native 2-D WiLoR review video"
    # radius=0 and max-hold=0 make this a raw baseline: no temporal smoothing
    # or gap filling is applied by the review renderer.
    run_python "$ROOT/scripts/render_native_2d_multiview.py" \
      --dataset "$NORMALIZED" --predictions "$RAW_PREDICTIONS" --output "$VIDEO" \
      --tile-width "$TILE_WIDTH" --video-codec "$VIDEO_CODEC" \
      --smoothing-radius 0 --smoothing-strength 0 \
      --wrist-outlier-threshold-px 1000000000 --outlier-threshold-px 1000000000 \
      --max-hold-frames 0 --max-frames "$MAX_FRAMES"
  else
    echo "[orbbec] reuse review video: $VIDEO"
  fi
  if ((OPTIMIZED_VIDEO)); then
    if [[ ! -f "$OPTIMIZED_VIDEO_PATH" ]]; then
      echo "[orbbec] render previous-method stabilized 2-D WiLoR review video"
      # This is the calibration-free part of the earlier improvement: lock
      # detector identities, collapse duplicate boxes, reject wrist teleports,
      # and apply a conservative local temporal median.  It does not invent
      # depth in this raw 2-D review mode.
      run_python "$ROOT/scripts/render_native_2d_multiview.py" \
        --dataset "$NORMALIZED" --predictions "$RAW_PREDICTIONS" \
        --output "$OPTIMIZED_VIDEO_PATH" --tile-width "$TILE_WIDTH" \
        --video-codec "$VIDEO_CODEC" --smoothing-radius 2 \
        --smoothing-strength 0.35 --outlier-threshold-px 25 \
        --wrist-outlier-threshold-px 80 --max-hold-frames 2 \
        --min-visible-joints 15 \
        --max-frames "$MAX_FRAMES"
    else
      echo "[orbbec] reuse optimized review video: $OPTIMIZED_VIDEO_PATH"
    fi
  fi
fi

python3 - "$OUTPUT/run_config.json" "$INPUT" "$NORMALIZED" "$RAW_PREDICTIONS" "$VIDEO" "$GPU_PROFILE" "$DEVICE" "$MAX_FRAMES" "${CAMERAS[*]}" "$OPTIMIZED_VIDEO" <<'PY'
import json
from pathlib import Path
import sys
path, input_path, normalized, predictions, video, profile, device, max_frames, cameras, optimized = sys.argv[1:]
normalized_manifest = json.loads((Path(normalized) / "manifest.json").read_text(encoding="utf-8"))
calibration_available = bool(normalized_manifest.get("calibration", {}).get("available", False))
payload = {
    "schema_version": 1,
    "stage": "orbbec_ego_pro_four_view_raw_wilor",
    "input": str(Path(input_path).expanduser().resolve()),
    "normalized_dataset": str(Path(normalized).resolve()),
    "raw_predictions": str(Path(predictions).resolve()),
    "raw_2d_video": str(Path(video).resolve()),
    "optimized_2d_video": str(Path(video).with_name("optimized_wilor_2d_dedup_quality15_4view.mp4").resolve()),
    "camera_ids": cameras.split(),
    "device": device,
    "gpu_profile": profile,
    "max_frames": int(max_frames),
    "calibration_available": calibration_available,
    "calibration_used": False,
    "sam3_used": False,
    "fusion_used": False,
    "temporal_correction_used": False,
    "optimized_2d_temporal_tracking_available": True,
    "optimized_2d_temporal_tracking_used": bool(int(optimized)),
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "Raw Orbbec four-view WiLoR result:"
echo "  normalized: $NORMALIZED"
echo "  predictions: $RAW_PREDICTIONS"
if ((NO_VIDEO == 0)); then echo "  video: $VIDEO"; fi
