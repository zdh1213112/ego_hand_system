#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCAP=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
MAX_FRAMES=60
BATCH_SIZE=""
FRAME_BATCH_SIZE=""
PREPROCESS_WORKERS=""
MAX_DETECTIONS_PER_CLASS=""
COMPILE_BACKBONE=""
GPU_PROFILE="compatible"
NO_VIDEO=0
TORCHINDUCTOR_CACHE="${EGO_TORCHINDUCTOR_CACHE_DIR:-/tmp/ego-hand-torchinductor}"

usage() {
  echo "Usage: $0 --mcap FILE --output DIR [--max-frames N] [--device cuda] [--conda-env NAME] [--gpu-profile compatible|rtx5090d] [--batch-size N] [--frame-batch-size N] [--preprocess-workers N] [--max-detections-per-class N] [--compile-backbone 0|1] [--no-video]"
}

while (($#)); do
  case "$1" in
    --mcap) MCAP="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --frame-batch-size) FRAME_BATCH_SIZE="$2"; shift 2 ;;
    --preprocess-workers) PREPROCESS_WORKERS="$2"; shift 2 ;;
    --max-detections-per-class) MAX_DETECTIONS_PER_CLASS="$2"; shift 2 ;;
    --compile-backbone) COMPILE_BACKBONE="$2"; shift 2 ;;
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
    ;;
  rtx5090d)
    BATCH_SIZE="${BATCH_SIZE:-16}"
    FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-4}"
    PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
    MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-1}"
    COMPILE_BACKBONE="${COMPILE_BACKBONE:-1}"
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
if [[ ! -f "$MCAP" ]]; then
  echo "MCAP does not exist: $MCAP" >&2
  exit 2
fi
if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$FRAME_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PREPROCESS_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_DETECTIONS_PER_CLASS" =~ ^[0-9]+$ && "$COMPILE_BACKBONE" =~ ^[01]$ ]]; then
  echo "frames/detection limit must be non-negative; batch sizes/workers must be positive" >&2
  exit 2
fi

OUTPUT="$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd)"
NORMALIZED="$OUTPUT/normalized_multiview"
PREDICTIONS="$OUTPUT/wilor_multiview"
FUSION="$OUTPUT/fusion_multiview"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE" \
    python "$@"
}

run_python - "$OUTPUT/run_config.json" "$MCAP" "$MAX_FRAMES" "$DEVICE" "$BATCH_SIZE" "$GPU_PROFILE" "$FRAME_BATCH_SIZE" "$PREPROCESS_WORKERS" "$MAX_DETECTIONS_PER_CLASS" "$COMPILE_BACKBONE" <<'PY'
import json
from pathlib import Path
import sys

config_path, source_text, max_frames, device, batch_size, gpu_profile, frame_batch_size, preprocess_workers, max_detections_per_class, compile_backbone = sys.argv[1:]
source = Path(source_text).resolve()
stat = source.stat()
config = {
    "mcap": {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    "cameras": [f"camera{index}" for index in range(6)],
    "reference_camera": "camera2",
    "max_frames": int(max_frames),
    "device": device,
    "batch_size": int(batch_size),
    "gpu_profile": gpu_profile,
    "frame_batch_size": int(frame_batch_size),
    "preprocess_workers": int(preprocess_workers),
    "max_detections_per_class": int(max_detections_per_class),
    "compile_backbone": bool(int(compile_backbone)),
    "camera_confidences": {"camera0": 0.2, "camera1": 0.3, "camera2": 0.3,
                           "camera3": 0.3, "camera4": 0.1, "camera5": 0.1},
    "fusion_algorithm": "anchor_guided_dynamic_temporal_handedness_v3",
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
require_complete_or_absent "$PREDICTIONS" summary.json
require_complete_or_absent "$FUSION" summary.json

if [[ ! -f "$NORMALIZED/manifest.json" ]]; then
  echo "[multiview] normalize and synchronize camera0..camera5"
  run_python "$ROOT/scripts/normalize_multiview_recording.py" \
    --input "$MCAP" --output "$NORMALIZED" \
    --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
    --reference-camera camera2 --max-delta-us 1500 --max-frames "$MAX_FRAMES"
else
  echo "[multiview] reuse normalized dataset"
fi

if [[ ! -f "$PREDICTIONS/summary.json" ]]; then
  echo "[multiview] run six-view dual-hypothesis WiLoR"
  run_python "$ROOT/scripts/wilor_multiview_inference.py" \
    --dataset "$NORMALIZED" --output "$PREDICTIONS" \
    --device "$DEVICE" --gpu-profile "$GPU_PROFILE" \
    --batch-size "$BATCH_SIZE" --frame-batch-size "$FRAME_BATCH_SIZE" \
    --preprocess-workers "$PREPROCESS_WORKERS" \
    --max-detections-per-class "$MAX_DETECTIONS_PER_CLASS" \
    --compile-backbone "$COMPILE_BACKBONE" \
    --max-frames "$MAX_FRAMES" \
    --camera-confidence camera0=0.2 \
    --camera-confidence camera4=0.1 \
    --camera-confidence camera5=0.1
else
  echo "[multiview] reuse WiLoR predictions"
fi

if [[ ! -f "$FUSION/summary.json" ]]; then
  echo "[multiview] fuse native Double-Sphere rays with RANSAC"
  run_python "$ROOT/scripts/fuse_multiview_wilor_guided.py" \
    --dataset "$NORMALIZED" --predictions "$PREDICTIONS" --output "$FUSION" \
    --anchor-cameras camera2 camera3 --detector-handedness strict --max-frames "$MAX_FRAMES"
else
  echo "[multiview] reuse fused result"
fi

if [[ "$NO_VIDEO" == 0 && ! -f "$FUSION/diagnostic_6view.mp4" ]]; then
  echo "[multiview] render 3x2 diagnostic video"
  run_python "$ROOT/scripts/render_multiview_wilor.py" \
    --dataset "$NORMALIZED" --fusion "$FUSION" \
    --output "$FUSION/diagnostic_6view.mp4" --max-frames "$MAX_FRAMES"
fi

echo "[multiview] finished: $FUSION"
cat "$FUSION/summary.json"
