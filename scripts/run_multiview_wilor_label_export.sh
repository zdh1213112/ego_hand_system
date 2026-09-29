#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT=""
FUSION=""
OUTPUT=""
CONDA_ENV="ego-hand"
DEVICE="cuda"
GPU_PROFILE="compatible"
FIT_POSE_WINDOW=""
FIT_POSE_OVERLAP=""
CUDA_MATMUL_PRECISION=""
CHECK_BATCH_SIZE=""
MANO_REPLAY_TOLERANCE=""
LEFT_CAMERA="camera2"
RIGHT_CAMERA="camera3"
EXPORT_CAMERAS=()
VIEW_FILTER="legacy"
RENDER_VISUALIZATION=1
VISUALIZATION_SAMPLES=12
VISUALIZATION_SEED=42
MAX_SAMPLES=0
SAMPLE_STRIDE=0
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
  --gpu-profile PROFILE     compatible|rtx5090d (default compatible)
  --fit-pose-window N       Override MANO pose window (default 48)
  --fit-pose-overlap N      Override MANO pose overlap (default 16)
  --cuda-matmul-precision P highest|high|medium (default highest)
  --check-batch-size N      MANO replay validation batch (profile default: 256|2048)
  --mano-replay-tolerance-m X MANO replay tolerance (profile default: 1e-5|2e-4 m)
  --left-camera CAMERA      First training view (default camera2)
  --right-camera CAMERA     Second training view (default camera3)
  --export-camera CAMERA    Export only this view; repeat for multiple views (default camera2+camera3)
  --view-filter MODE        legacy|complete21 (default legacy)
  --render-visualization N  Write random per-camera overlay JPGs: 1|0 (default 1)
  --visualization-samples N Random sync frames per camera (default 12)
  --visualization-seed N    Reproducible random seed (default 42)
  --max-samples N           Export sample limit; 0 means all
  --sample-stride N         Fixed sync-frame stride; 0 means motion-adaptive sampling (default)
  --mano-source DIR         Licensed MANO Python source
  --mano-model-dir DIR      Directory containing MANO_RIGHT.pkl
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
    --gpu-profile) GPU_PROFILE="$2"; shift 2 ;;
    --fit-pose-window) FIT_POSE_WINDOW="$2"; shift 2 ;;
    --fit-pose-overlap) FIT_POSE_OVERLAP="$2"; shift 2 ;;
    --cuda-matmul-precision) CUDA_MATMUL_PRECISION="$2"; shift 2 ;;
    --check-batch-size) CHECK_BATCH_SIZE="$2"; shift 2 ;;
    --mano-replay-tolerance-m) MANO_REPLAY_TOLERANCE="$2"; shift 2 ;;
    --left-camera) LEFT_CAMERA="$2"; shift 2 ;;
    --right-camera) RIGHT_CAMERA="$2"; shift 2 ;;
    --export-camera) EXPORT_CAMERAS+=("$2"); shift 2 ;;
    --view-filter) VIEW_FILTER="$2"; shift 2 ;;
    --render-visualization) RENDER_VISUALIZATION="$2"; shift 2 ;;
    --visualization-samples) VISUALIZATION_SAMPLES="$2"; shift 2 ;;
    --visualization-seed) VISUALIZATION_SEED="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --sample-stride) SAMPLE_STRIDE="$2"; shift 2 ;;
    --mano-source) MANO_SOURCE="$2"; shift 2 ;;
    --mano-model-dir) MANO_MODEL_DIR="$2"; shift 2 ;;
    --reference-npy) REFERENCE_NPY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$GPU_PROFILE" in
  compatible)
    FIT_POSE_WINDOW="${FIT_POSE_WINDOW:-48}"
    FIT_POSE_OVERLAP="${FIT_POSE_OVERLAP:-16}"
    CUDA_MATMUL_PRECISION="${CUDA_MATMUL_PRECISION:-highest}"
    CHECK_BATCH_SIZE="${CHECK_BATCH_SIZE:-256}"
    MANO_REPLAY_TOLERANCE="${MANO_REPLAY_TOLERANCE:-0.00001}"
    ;;
  rtx5090d)
    FIT_POSE_WINDOW="${FIT_POSE_WINDOW:-48}"
    FIT_POSE_OVERLAP="${FIT_POSE_OVERLAP:-16}"
    CUDA_MATMUL_PRECISION="${CUDA_MATMUL_PRECISION:-highest}"
    CHECK_BATCH_SIZE="${CHECK_BATCH_SIZE:-2048}"
    MANO_REPLAY_TOLERANCE="${MANO_REPLAY_TOLERANCE:-0.00001}"
    ;;
  *)
    echo "--gpu-profile must be compatible or rtx5090d: $GPU_PROFILE" >&2
    exit 2
    ;;
esac

[[ -n "$EXPERIMENT" ]] || { usage >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "--max-samples must be non-negative" >&2; exit 2; }
[[ "$SAMPLE_STRIDE" =~ ^[0-9]+$ ]] || { echo "--sample-stride must be a non-negative integer" >&2; exit 2; }
[[ "$VIEW_FILTER" == "legacy" || "$VIEW_FILTER" == "complete21" ]] || {
  echo "--view-filter must be legacy or complete21" >&2; exit 2;
}
[[ "$RENDER_VISUALIZATION" == "0" || "$RENDER_VISUALIZATION" == "1" ]] || {
  echo "--render-visualization must be 0 or 1" >&2; exit 2;
}
[[ "$VISUALIZATION_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
  echo "--visualization-samples must be a positive integer" >&2; exit 2;
}
[[ "$VISUALIZATION_SEED" =~ ^[0-9]+$ ]] || {
  echo "--visualization-seed must be a non-negative integer" >&2; exit 2;
}
[[ "$FIT_POSE_WINDOW" =~ ^[1-9][0-9]*$ && "$FIT_POSE_OVERLAP" =~ ^[0-9]+$ ]] || {
  echo "fit pose window must be positive and overlap must be non-negative" >&2; exit 2;
}
[[ "$CHECK_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--check-batch-size must be a positive integer" >&2; exit 2;
}
python3 - "$MANO_REPLAY_TOLERANCE" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError as error:
    raise SystemExit("--mano-replay-tolerance-m must be numeric") from error
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit("--mano-replay-tolerance-m must be finite and positive")
PY
((FIT_POSE_OVERLAP < FIT_POSE_WINDOW)) || {
  echo "--fit-pose-overlap must be smaller than --fit-pose-window" >&2; exit 2;
}
[[ "$CUDA_MATMUL_PRECISION" == "highest" || "$CUDA_MATMUL_PRECISION" == "high" || "$CUDA_MATMUL_PRECISION" == "medium" ]] || {
  echo "--cuda-matmul-precision must be highest, high, or medium" >&2; exit 2;
}
if [[ "$GPU_PROFILE" == "rtx5090d" && "$DEVICE" == "cpu" ]]; then
  echo "--gpu-profile rtx5090d requires --device cuda or auto" >&2
  exit 2
fi
[[ "$LEFT_CAMERA" != "$RIGHT_CAMERA" ]] || { echo "left/right camera must differ" >&2; exit 2; }
if ((${#EXPORT_CAMERAS[@]} == 0)); then
  EXPORT_CAMERAS=("$LEFT_CAMERA" "$RIGHT_CAMERA")
fi
for camera in "${EXPORT_CAMERAS[@]}"; do
  [[ "$camera" == "$LEFT_CAMERA" || "$camera" == "$RIGHT_CAMERA" ]] || {
    echo "--export-camera must be $LEFT_CAMERA or $RIGHT_CAMERA for this rectification" >&2
    exit 2
  }
done
EXPORT_CAMERAS_TEXT="${EXPORT_CAMERAS[*]}"
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
[[ -f "$MANO_MODEL_DIR/MANO_RIGHT.pkl" ]] || {
  echo "Missing MANO_RIGHT.pkl in: $MANO_MODEL_DIR" >&2; exit 2;
}
[[ -f "$MANO_SOURCE/mano/model.py" ]] || { echo "Invalid MANO source: $MANO_SOURCE" >&2; exit 2; }

OUTPUT="${OUTPUT:-$EXPERIMENT/wilor_training_labels_physical_v1}"
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
  "$MANO_MODEL_DIR/MANO_RIGHT.pkl" \
  "$LEFT_CAMERA" "$RIGHT_CAMERA" "$EXPORT_CAMERAS_TEXT" "$VIEW_FILTER" \
  "$MAX_SAMPLES" "$SAMPLE_STRIDE" "$DEVICE" "$GPU_PROFILE" \
  "$FIT_POSE_WINDOW" "$FIT_POSE_OVERLAP" "$CUDA_MATMUL_PRECISION" <<'PY'
import json
from pathlib import Path
import sys

config_path, *values = sys.argv[1:]
(
    manifest, accepted, mano_source, mano_right,
    left_camera, right_camera, export_cameras, view_filter,
    max_samples, sample_stride, device, gpu_profile,
    fit_pose_window, fit_pose_overlap, cuda_matmul_precision,
) = values

def signature(value):
    path = Path(value).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

config = {
    "schema_version": 5,
    "algorithm": "strict_six_view_fusion_physical_labels_right_mano_v5",
    "normalized_manifest": signature(manifest),
    "fusion_accepted": signature(accepted),
    "mano_fit_convention": "wilor_right_canonical_v1",
    "mano_assets": [signature(mano_source), signature(mano_right)],
    "left_camera": left_camera,
    "right_camera": right_camera,
    "export_cameras": export_cameras.split(),
    "view_filter": view_filter,
    "max_samples": int(max_samples),
    "sampling": {
        "mode": "fixed_stride" if int(sample_stride) > 0 else "motion_adaptive",
        "sample_stride": int(sample_stride),
    },
    "device": device,
    "fit": {
        "shape_iterations": 180, "pose_iterations": 120,
        "pose_window": int(fit_pose_window), "pose_overlap": int(fit_pose_overlap),
        "learning_rate": 0.008,
        "w_3d": 1.0, "w_2d": 0.30, "min_fit_observed_points": 12,
        "max_unobserved_gap": 3, "w_pose": 0.003, "w_shape": 0.015,
        "w_temporal": 0.08, "w_rigid_temporal": 0.02,
        "w_acceleration": 0.01,
    },
}
if gpu_profile != "compatible" or cuda_matmul_precision != "highest":
    config["gpu_profile"] = gpu_profile
    config["cuda_matmul_precision"] = cuda_matmul_precision
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
  echo "[labels] fit both physical hands in the MANO_RIGHT canonical space"
  echo "[labels] GPU profile=$GPU_PROFILE pose_window=$FIT_POSE_WINDOW pose_overlap=$FIT_POSE_OVERLAP matmul_precision=$CUDA_MATMUL_PRECISION"
  run_python "$ROOT/scripts/fit_mano_sequence.py" \
    --input "$MANO_INPUT" \
    --mano-source "$MANO_SOURCE" --model-dir "$MANO_MODEL_DIR" \
    --mano-convention wilor_right_canonical_v1 \
    --output "$MANO_FIT" --device "$DEVICE" --no-video \
    --shape-iterations 180 --pose-iterations 120 \
    --pose-window "$FIT_POSE_WINDOW" --pose-overlap "$FIT_POSE_OVERLAP" \
    --cuda-matmul-precision "$CUDA_MATMUL_PRECISION" --learning-rate 0.008 \
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
    --cameras "${EXPORT_CAMERAS[@]}" --view-filter "$VIEW_FILTER"
    --max-samples "$MAX_SAMPLES"
    --sample-stride "$SAMPLE_STRIDE"
  )
  run_python "$ROOT/scripts/export_multiview_wilor_training_dataset.py" "${EXPORT_ARGS[@]}"
else
  echo "[labels] reuse completed paired training dataset"
fi

CHECK_ARGS=(
  "$ROOT/scripts/check_wilor_training_dataset.py" "$TRAINING_DATASET"
  --mano-source "$MANO_SOURCE" --mano-model-dir "$MANO_MODEL_DIR"
  --device "$DEVICE" --batch-size "$CHECK_BATCH_SIZE"
  --mano-reconstruction-tolerance-m "$MANO_REPLAY_TOLERANCE"
)
[[ -f "$REFERENCE_NPY" ]] && CHECK_ARGS+=(--reference "$REFERENCE_NPY")
echo "[labels] validate image/NPY pairing, schema and exact mesh projection"
run_python "${CHECK_ARGS[@]}"
if [[ "$RENDER_VISUALIZATION" == "1" ]]; then
  VISUALIZATION_DIR="$TRAINING_DATASET/visualization"
  echo "[labels] render random complete 21-joint training-view images"
  run_python "$ROOT/scripts/render_wilor_training_dataset.py" \
    --dataset "$TRAINING_DATASET" --output "$VISUALIZATION_DIR" \
    --cameras "${EXPORT_CAMERAS[@]}" \
    --samples "$VISUALIZATION_SAMPLES" --seed "$VISUALIZATION_SEED"
fi
echo "[labels] finished: $TRAINING_DATASET"
