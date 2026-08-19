#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCAP=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
MAX_FRAMES=60
BATCH_SIZE=4
NO_VIDEO=0

usage() {
  echo "Usage: $0 --mcap FILE --output DIR [--max-frames N] [--device cuda] [--conda-env NAME] [--batch-size N] [--no-video]"
}

while (($#)); do
  case "$1" in
    --mcap) MCAP="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MCAP" || -z "$OUTPUT" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$MCAP" ]]; then
  echo "MCAP does not exist: $MCAP" >&2
  exit 2
fi
if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-frames must be non-negative and --batch-size must be positive" >&2
  exit 2
fi

OUTPUT="$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd)"
NORMALIZED="$OUTPUT/normalized_multiview"
PREDICTIONS="$OUTPUT/wilor_multiview"
FUSION="$OUTPUT/fusion_multiview"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    python "$@"
}

run_python - "$OUTPUT/run_config.json" "$MCAP" "$MAX_FRAMES" "$DEVICE" "$BATCH_SIZE" <<'PY'
import json
from pathlib import Path
import sys

config_path, source_text, max_frames, device, batch_size = sys.argv[1:]
source = Path(source_text).resolve()
stat = source.stat()
config = {
    "mcap": {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    "cameras": [f"camera{index}" for index in range(6)],
    "reference_camera": "camera2",
    "max_frames": int(max_frames),
    "device": device,
    "batch_size": int(batch_size),
    "camera_confidences": {"camera0": 0.2, "camera1": 0.3, "camera2": 0.3,
                           "camera3": 0.3, "camera4": 0.1, "camera5": 0.1},
    "fusion_algorithm": "anchor_guided_dynamic_temporal_v2",
}
path = Path(config_path)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
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
    --device "$DEVICE" --batch-size "$BATCH_SIZE" --max-frames "$MAX_FRAMES" \
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
    --anchor-cameras camera2 camera3 --max-frames "$MAX_FRAMES"
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
