#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT=""
FUSION=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
LEFT_CAMERA="camera2"
RIGHT_CAMERA="camera3"
MAX_SAMPLES=0
MANO_SOURCE="${GLOVE_MANO_SOURCE:-$ROOT/third_party/MANO}"
MANO_MODEL_DIR="${GLOVE_MANO_MODEL_DIR:-$ROOT/models/mano}"
REFERENCE_NPY="/home/zdh/tool/npy_decoder/000865.npy"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_multiview_wilor_label_export.sh \
    --experiment /path/to/six_view_experiment \
    [--fusion /path/to/strict_fusion] [--output /path/to/label_output]

Options:
  --conda-env NAME          Conda environment (default ego-hand)
  --device cuda|cpu|auto    MANO fitting device (default cuda)
  --left-camera CAMERA      First training view (default camera2)
  --right-camera CAMERA     Second training view (default camera3)
  --max-samples N           Export sample limit; 0 means all
  --mano-source DIR         Licensed MANO Python source
  --mano-model-dir DIR      Directory containing MANO_LEFT/RIGHT.pkl
  --reference-npy FILE      Reference schema sample (default 000865.npy)
EOF
}

while (($#)); do
  case "$1" in
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --fusion) FUSION="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --left-camera) LEFT_CAMERA="$2"; shift 2 ;;
    --right-camera) RIGHT_CAMERA="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --mano-source) MANO_SOURCE="$2"; shift 2 ;;
    --mano-model-dir) MANO_MODEL_DIR="$2"; shift 2 ;;
    --reference-npy) REFERENCE_NPY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXPERIMENT" ]] || { usage >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "--max-samples must be non-negative" >&2; exit 2; }
[[ "$LEFT_CAMERA" != "$RIGHT_CAMERA" ]] || { echo "left/right camera must differ" >&2; exit 2; }
EXPERIMENT="$(cd "$EXPERIMENT" && pwd)"
NORMALIZED="$EXPERIMENT/normalized_multiview"
[[ -f "$NORMALIZED/manifest.json" ]] || { echo "Missing multiview dataset: $NORMALIZED" >&2; exit 2; }

if [[ -z "$FUSION" ]]; then
  if [[ -f "$EXPERIMENT/fusion_handedness_strict_full/summary.json" ]]; then
    FUSION="$EXPERIMENT/fusion_handedness_strict_full"
  else
    FUSION="$EXPERIMENT/fusion_multiview"
  fi
fi
FUSION="$(cd "$FUSION" && pwd)"
[[ -s "$FUSION/accepted.jsonl" && -f "$FUSION/summary.json" ]] || {
  echo "Missing completed fusion result: $FUSION" >&2; exit 2;
}
[[ -f "$MANO_MODEL_DIR/MANO_LEFT.pkl" && -f "$MANO_MODEL_DIR/MANO_RIGHT.pkl" ]] || {
  echo "Missing MANO model files in: $MANO_MODEL_DIR" >&2; exit 2;
}
[[ -f "$MANO_SOURCE/mano/model.py" ]] || { echo "Invalid MANO source: $MANO_SOURCE" >&2; exit 2; }

OUTPUT="${OUTPUT:-$EXPERIMENT/wilor_training_labels}"
[[ "$OUTPUT" != "/" && "$OUTPUT" != "$EXPERIMENT" ]] || { echo "Unsafe output path" >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
MANO_INPUT="$OUTPUT/mano_input_multiview.npz"
RECTIFICATION="$OUTPUT/training_rectification.npz"
MANO_FIT="$OUTPUT/mano_fit_multiview"
TRAINING_DATASET="$OUTPUT/dataset"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" \
    python "$@"
}

run_python - "$OUTPUT/run_config.json" "$NORMALIZED/manifest.json" \
  "$FUSION/accepted.jsonl" "$MANO_SOURCE/mano/model.py" \
  "$MANO_MODEL_DIR/MANO_LEFT.pkl" "$MANO_MODEL_DIR/MANO_RIGHT.pkl" \
  "$LEFT_CAMERA" "$RIGHT_CAMERA" "$MAX_SAMPLES" "$DEVICE" <<'PY'
import json
from pathlib import Path
import sys

config_path, *values = sys.argv[1:]
manifest, accepted, mano_source, mano_left, mano_right, left_camera, right_camera, max_samples, device = values

def signature(value):
    path = Path(value).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

config = {
    "schema_version": 1,
    "algorithm": "strict_six_view_fusion_shared_mano_wilor_dataset_v1",
    "normalized_manifest": signature(manifest),
    "fusion_accepted": signature(accepted),
    "mano_assets": [signature(mano_source), signature(mano_left), signature(mano_right)],
    "left_camera": left_camera,
    "right_camera": right_camera,
    "max_samples": int(max_samples),
    "device": device,
    "fit": {
        "shape_iterations": 180, "pose_iterations": 120,
        "pose_window": 48, "pose_overlap": 16, "learning_rate": 0.008,
        "w_3d": 1.0, "w_2d": 0.30, "min_fit_observed_points": 12,
        "max_unobserved_gap": 3, "w_pose": 0.003, "w_shape": 0.015,
        "w_temporal": 0.08, "w_rigid_temporal": 0.02,
        "w_acceleration": 0.01,
    },
}
path = Path(config_path)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != config:
        changed = next(key for key in sorted(set(previous) | set(config))
                       if previous.get(key) != config.get(key))
        raise SystemExit(
            f"label run configuration differs at {changed}:\n"
            f"  previous={previous.get(changed)!r}\n  current ={config.get(changed)!r}"
        )
else:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

if [[ ! -f "$MANO_INPUT" || ! -f "${MANO_INPUT%.npz}.json" || ! -f "$RECTIFICATION" ]]; then
  if [[ -e "$MANO_INPUT" || -e "${MANO_INPUT%.npz}.json" || -e "$RECTIFICATION" ]]; then
    echo "Incomplete MANO preparation exists in $OUTPUT; preserve it and use a new output directory." >&2
    exit 2
  fi
  echo "[labels] convert strict six-view fusion to shared MANO observations"
  run_python "$ROOT/scripts/prepare_multiview_mano_input.py" \
    --fusion "$FUSION" --dataset "$NORMALIZED" \
    --output "$MANO_INPUT" --rectification-output "$RECTIFICATION" \
    --left-camera "$LEFT_CAMERA" --right-camera "$RIGHT_CAMERA"
else
  echo "[labels] reuse MANO observations and rectification"
fi

if [[ ! -f "$MANO_FIT/summary.json" || ! -f "$MANO_FIT/track_0.npz" || ! -f "$MANO_FIT/track_1.npz" ]]; then
  if [[ -e "$MANO_FIT" ]]; then
    echo "Incomplete MANO fit exists: $MANO_FIT; preserve it and use a new output directory." >&2
    exit 2
  fi
  echo "[labels] fit one temporally consistent MANO shape/pose track per physical hand"
  run_python "$ROOT/scripts/fit_mano_sequence.py" \
    --input "$MANO_INPUT" \
    --mano-source "$MANO_SOURCE" --model-dir "$MANO_MODEL_DIR" \
    --output "$MANO_FIT" --device "$DEVICE" --no-video \
    --shape-iterations 180 --pose-iterations 120 \
    --pose-window 48 --pose-overlap 16 --learning-rate 0.008 \
    --w-3d 1.0 --w-2d 0.30 --min-fit-observed-points 12 \
    --max-unobserved-gap 3 --w-pose 0.003 --w-shape 0.015 \
    --w-temporal 0.08 --w-rigid-temporal 0.02 --w-acceleration 0.01 \
    --boundary-weight 0.10 --max-orient-step-deg 45 \
    --max-translation-step-m 0.05
else
  echo "[labels] reuse completed multiview MANO fit"
fi

if [[ ! -f "$TRAINING_DATASET/summary.json" ]]; then
  if [[ -e "$TRAINING_DATASET" ]]; then
    echo "Incomplete training dataset exists: $TRAINING_DATASET; preserve it and use a new output directory." >&2
    exit 2
  fi
  echo "[labels] export paired rectified images and 000865-compatible NPY labels"
  EXPORT_ARGS=(
    --fusion "$FUSION" --mano-fit "$MANO_FIT" --dataset "$NORMALIZED"
    --rectification "$RECTIFICATION" --output "$TRAINING_DATASET"
    --cameras "$LEFT_CAMERA" "$RIGHT_CAMERA" --max-samples "$MAX_SAMPLES"
  )
  run_python "$ROOT/scripts/export_multiview_wilor_training_dataset.py" "${EXPORT_ARGS[@]}"
else
  echo "[labels] reuse completed paired training dataset"
fi

CHECK_ARGS=("$ROOT/scripts/check_wilor_training_dataset.py" "$TRAINING_DATASET")
[[ -f "$REFERENCE_NPY" ]] && CHECK_ARGS+=(--reference "$REFERENCE_NPY")
echo "[labels] validate image/NPY pairing, schema and exact mesh projection"
run_python "${CHECK_ARGS[@]}"
echo "[labels] finished: $TRAINING_DATASET"
