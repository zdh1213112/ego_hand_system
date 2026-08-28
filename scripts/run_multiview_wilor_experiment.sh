#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCAP=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
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
DETECTOR_HANDEDNESS=""
GLOVE_MARKER_ASSIST="${EGO_GLOVE_MARKER_ASSIST:-${EGO_NOKOV_WILOR_ASSIST:-0}}"
MARKER_SATURATION_MAX="${EGO_MARKER_SATURATION_MAX:-100}"
MARKER_VALUE_MIN="${EGO_MARKER_VALUE_MIN:-160}"
MARKER_MIN_MATCHES="${EGO_MARKER_MIN_MATCHES:-5}"
MARKER_MIN_FINGER_GROUPS="${EGO_MARKER_MIN_FINGER_GROUPS:-3}"
MARKER_SEARCH_PADDING_PX="${EGO_MARKER_SEARCH_PADDING_PX:-45}"
MARKER_SEED_DISTANCE_PX="${EGO_MARKER_SEED_DISTANCE_PX:-35}"
MARKER_MATCH_DISTANCE_PX="${EGO_MARKER_MATCH_DISTANCE_PX:-13}"
MARKER_MAX_SHIFT_PX="${EGO_MARKER_MAX_SHIFT_PX:-30}"
MARKER_BLEND="${EGO_MARKER_BLEND:-0.35}"
TORCHINDUCTOR_CACHE="${EGO_TORCHINDUCTOR_CACHE_DIR:-/tmp/ego-hand-torchinductor}"

usage() {
  echo "Usage: $0 --mcap FILE --output DIR [--cameras camera0 camera1 ...] [--reference-camera camera2] [--max-frames N] [--device cuda] [--conda-env NAME] [--gpu-profile compatible|rtx5090d] [--batch-size N] [--frame-batch-size N] [--preprocess-workers N] [--max-detections-per-class N] [--compile-backbone 0|1] [--fusion-workers N] [--glove-marker-assist 0|1] [--marker-value-min 160] [--marker-saturation-max 100] [--marker-min-matches 5] [--marker-blend 0.35] [--detector-handedness strict|ignore|adaptive] [--no-video]"
}

while (($#)); do
  case "$1" in
    --mcap) MCAP="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
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
    --fusion-workers) FUSION_WORKERS="$2"; shift 2 ;;
    --glove-marker-assist|--nokov-wilor-assist) GLOVE_MARKER_ASSIST="$2"; shift 2 ;;
    --marker-saturation-max) MARKER_SATURATION_MAX="$2"; shift 2 ;;
    --marker-value-min) MARKER_VALUE_MIN="$2"; shift 2 ;;
    --marker-min-matches) MARKER_MIN_MATCHES="$2"; shift 2 ;;
    --marker-min-finger-groups) MARKER_MIN_FINGER_GROUPS="$2"; shift 2 ;;
    --marker-search-padding-px) MARKER_SEARCH_PADDING_PX="$2"; shift 2 ;;
    --marker-seed-distance-px) MARKER_SEED_DISTANCE_PX="$2"; shift 2 ;;
    --marker-match-distance-px) MARKER_MATCH_DISTANCE_PX="$2"; shift 2 ;;
    --marker-max-shift-px) MARKER_MAX_SHIFT_PX="$2"; shift 2 ;;
    --marker-blend) MARKER_BLEND="$2"; shift 2 ;;
    --detector-handedness) DETECTOR_HANDEDNESS="$2"; shift 2 ;;
    --gpu-profile) GPU_PROFILE="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
    BATCH_SIZE="${BATCH_SIZE:-16}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-4}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
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
if [[ "${CAMERA_SEEN[camera2]:-}" == 1 && "${CAMERA_SEEN[camera3]:-}" == 1 ]]; then
  ANCHOR_CAMERAS=(camera2 camera3)
else
  ANCHOR_CAMERAS=("${CAMERAS[0]}" "${CAMERAS[1]}")
fi
if [[ ! -f "$MCAP" ]]; then
  echo "MCAP does not exist: $MCAP" >&2
  exit 2
fi
if ! [[ "$GLOVE_MARKER_ASSIST" =~ ^[01]$ ]]; then
  echo "--glove-marker-assist must be 0 or 1: $GLOVE_MARKER_ASSIST" >&2
  exit 2
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
if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$FRAME_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PREPROCESS_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_DETECTIONS_PER_CLASS" =~ ^[0-9]+$ && "$COMPILE_BACKBONE" =~ ^[01]$ && "$FUSION_WORKERS" =~ ^[1-9][0-9]*$ && "$MARKER_SATURATION_MAX" =~ ^[0-9]+$ && "$MARKER_VALUE_MIN" =~ ^[0-9]+$ && "$MARKER_MIN_MATCHES" =~ ^[1-9][0-9]*$ && "$MARKER_MIN_FINGER_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "frames/detection limit must be non-negative; batch sizes/workers must be positive" >&2
  exit 2
fi
for VALUE in "$MARKER_SEARCH_PADDING_PX" "$MARKER_SEED_DISTANCE_PX" "$MARKER_MATCH_DISTANCE_PX" "$MARKER_MAX_SHIFT_PX" "$MARKER_BLEND"; do
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
VIDEO="$FUSION/diagnostic_${#CAMERAS[@]}view.mp4"
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

run_python - "$OUTPUT/run_config.json" "$MCAP" "$MAX_FRAMES" "$DEVICE" "$BATCH_SIZE" "$GPU_PROFILE" "$FRAME_BATCH_SIZE" "$PREPROCESS_WORKERS" "$MAX_DETECTIONS_PER_CLASS" "$COMPILE_BACKBONE" "$FUSION_WORKERS" "$CAMERA_CSV" "$REFERENCE_CAMERA" "${ANCHOR_CAMERAS[*]}" "$DETECTOR_HANDEDNESS" "$GLOVE_MARKER_ASSIST" "$MARKER_SATURATION_MAX" "$MARKER_VALUE_MIN" "$MARKER_MIN_MATCHES" "$MARKER_MIN_FINGER_GROUPS" "$MARKER_SEARCH_PADDING_PX" "$MARKER_SEED_DISTANCE_PX" "$MARKER_MATCH_DISTANCE_PX" "$MARKER_MAX_SHIFT_PX" "$MARKER_BLEND" <<'PY'
import json
from pathlib import Path
import sys

from glove_marker_assist import MarkerAssistConfig

config_path, source_text, max_frames, device, batch_size, gpu_profile, frame_batch_size, preprocess_workers, max_detections_per_class, compile_backbone, fusion_workers, camera_csv, reference_camera, anchor_cameras_text, detector_handedness, marker_enabled, marker_saturation_max, marker_value_min, marker_min_matches, marker_min_finger_groups, marker_search_padding_px, marker_seed_distance_px, marker_match_distance_px, marker_max_shift_px, marker_blend = sys.argv[1:]
source = Path(source_text).resolve()
stat = source.stat()
camera_ids = camera_csv.split(",")
marker_config = MarkerAssistConfig(
    saturation_max=int(marker_saturation_max),
    value_min=int(marker_value_min),
    min_matches=int(marker_min_matches),
    min_finger_groups=int(marker_min_finger_groups),
    search_padding_px=float(marker_search_padding_px),
    seed_distance_px=float(marker_seed_distance_px),
    match_distance_px=float(marker_match_distance_px),
    max_shift_px=float(marker_max_shift_px),
    marker_blend=float(marker_blend),
)
marker_config.validate()
config = {
    "mcap": {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    "cameras": camera_ids,
    "reference_camera": reference_camera,
    "anchor_cameras": anchor_cameras_text.split(),
    "max_frames": int(max_frames),
    "device": device,
    "batch_size": int(batch_size),
    "gpu_profile": gpu_profile,
    "frame_batch_size": int(frame_batch_size),
    "preprocess_workers": int(preprocess_workers),
    "max_detections_per_class": int(max_detections_per_class),
    "compile_backbone": bool(int(compile_backbone)),
    "fusion_workers": int(fusion_workers),
    "detector_handedness": detector_handedness,
    "glove_marker_assist": {"enabled": bool(int(marker_enabled)), **marker_config.to_dict()},
    "camera_confidences": {"camera0": 0.2, "camera1": 0.3, "camera2": 0.3,
                           "camera3": 0.3, "camera4": 0.1, "camera5": 0.1},
    "fusion_algorithm": "anchor_guided_dynamic_temporal_marker_v4",
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
    previous.setdefault(
        "glove_marker_assist",
        {"enabled": False, **MarkerAssistConfig().to_dict()},
    )
    if previous.get("fusion_algorithm") == "anchor_guided_dynamic_temporal_handedness_v3":
        previous["fusion_algorithm"] = "anchor_guided_dynamic_temporal_marker_v4"
    if previous != config:
        changed = next(key for key in config if previous.get(key) != config[key])
        raise SystemExit(
            f"run configuration differs at {changed}:\n"
            f"  previous={previous.get(changed)!r}\n  current ={config[changed]!r}"
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
  require_complete_or_absent "$PREDICTIONS" summary.json
fi
require_complete_or_absent "$FUSION" summary.json

if [[ ! -f "$NORMALIZED/manifest.json" ]]; then
  echo "[multiview] normalize and synchronize ${CAMERAS[*]}"
  run_python "$ROOT/scripts/normalize_multiview_recording.py" \
    --input "$MCAP" --output "$NORMALIZED" \
    --cameras "${CAMERAS[@]}" \
    --reference-camera "$REFERENCE_CAMERA" --max-delta-us 1500 --max-frames "$MAX_FRAMES"
else
  echo "[multiview] reuse normalized dataset"
fi

if [[ ! -f "$RAW_PREDICTIONS/summary.json" ]]; then
  echo "[multiview] run ${#CAMERAS[@]}-view dual-hypothesis WiLoR"
  run_python "$ROOT/scripts/wilor_multiview_inference.py" \
    --dataset "$NORMALIZED" --output "$RAW_PREDICTIONS" \
    --device "$DEVICE" --gpu-profile "$GPU_PROFILE" \
    --batch-size "$BATCH_SIZE" --frame-batch-size "$FRAME_BATCH_SIZE" \
    --preprocess-workers "$PREPROCESS_WORKERS" \
    --max-detections-per-class "$MAX_DETECTIONS_PER_CLASS" \
    --compile-backbone "$COMPILE_BACKBONE" \
    --max-frames "$MAX_FRAMES" \
    "${CAMERA_CONFIDENCE_ARGS[@]}"
else
  echo "[multiview] reuse WiLoR predictions"
fi

if ((GLOVE_MARKER_ASSIST)) && [[ ! -f "$PREDICTIONS/summary.json" ]]; then
  echo "[multiview] refine WiLoR image joints with visible glove markers"
  run_python "$ROOT/scripts/assist_wilor_with_glove_markers.py" \
    --dataset "$NORMALIZED" --predictions "$RAW_PREDICTIONS" --output "$PREDICTIONS" \
    --cameras "${CAMERAS[@]}" --max-frames "$MAX_FRAMES" \
    --saturation-max "$MARKER_SATURATION_MAX" --value-min "$MARKER_VALUE_MIN" \
    --min-matches "$MARKER_MIN_MATCHES" \
    --min-finger-groups "$MARKER_MIN_FINGER_GROUPS" \
    --search-padding-px "$MARKER_SEARCH_PADDING_PX" \
    --seed-distance-px "$MARKER_SEED_DISTANCE_PX" \
    --match-distance-px "$MARKER_MATCH_DISTANCE_PX" \
    --max-shift-px "$MARKER_MAX_SHIFT_PX" --marker-blend "$MARKER_BLEND"
elif ((GLOVE_MARKER_ASSIST)); then
  echo "[multiview] reuse glove-marker-assisted WiLoR predictions"
fi

if [[ ! -f "$FUSION/summary.json" ]]; then
  echo "[multiview] fuse native Double-Sphere rays with RANSAC"
  FUSION_ARGS=(
    --dataset "$NORMALIZED"
    --predictions "$PREDICTIONS"
    --output "$FUSION"
    --cameras "${CAMERAS[@]}"
    --anchor-cameras "${ANCHOR_CAMERAS[@]}"
    --detector-handedness "$DETECTOR_HANDEDNESS"
    --workers "$FUSION_WORKERS"
    --max-frames "$MAX_FRAMES"
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
    --max-frames "$MAX_FRAMES"
fi

echo "[multiview] finished: $FUSION"
cat "$FUSION/summary.json"
