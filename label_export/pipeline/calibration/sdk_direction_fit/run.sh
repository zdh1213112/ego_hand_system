#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$TASK_ROOT"
EXP="$TASK_ROOT/output/native_kb_training_20261111"
CONTEXT="$EXP/explicit_pipeline_sampled_smoke_v2/context.json"
SOURCE="$EXP/full_silhouette_v1/shape_only"
PREVIOUS_REVIEW="$EXP/full_silhouette_v1/review"
SKELETON="$EXP/nokov_skeleton_audit_v1/aligned_skeleton.npz"
SELECTION="$EXP/keyframe_calibration_v3/selection.json"
RESULT="$EXP/sdk_direction_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${SDK_FIT_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
while (($#)); do
  case "$1" in
    --output) RESULT="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --previous-review) PREVIOUS_REVIEW="$2"; shift 2;;
    --skeleton) SKELETON="$2"; shift 2;;
    --selection) SELECTION="$2"; shift 2;;
    --help|-h) echo 'SDK方向辅助实验：默认复用192916录制已审核的个人手型与附着、SDK和优质帧；Finger0为参考根。输出marker对照、方向辅助、原图四目对照及视频，不修改正式默认导出。'; exit 0;;
    *) echo "未知参数：$1" >&2; exit 2;;
  esac
done
[[ ! -e "$RESULT" ]] || { echo "输出目录已存在：$RESULT" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/build_mapping.py --context "$CONTEXT" --source "$SOURCE" --skeleton "$SKELETON" --selection "$SELECTION" --output "$RESULT/profile"
for variant in marker_only direction; do
  "$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/fit.py --context "$CONTEXT" --source "$SOURCE" --profile "$RESULT/profile" --variant "$variant" --output "$RESULT/$variant"
  "$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/export_labels.py --context "$CONTEXT" --source "$RESULT/$variant"
done
DATASET="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset"])' "$CONTEXT")"
"$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/build_review.py --previous-review "$PREVIOUS_REVIEW" --result "$RESULT" --dataset "$DATASET" --output "$RESULT/review"
"$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/evaluate_fit.py --context "$CONTEXT" --selection "$SELECTION" --source "$SOURCE" --result "$RESULT"
for variant in direction marker_only; do
  "$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/render_previews.py --review "$RESULT/review" --output "$RESULT/closeups_$variant" --root "$variant"
done
"$PYTHON_BIN" -B label_export/pipeline/calibration/sdk_direction_fit/verify.py --source "$SOURCE" --result "$RESULT"
echo "完成：$RESULT"
echo "本地查看：$PYTHON_BIN -m http.server 8776 --bind 127.0.0.1 --directory $RESULT"
echo '打开：http://127.0.0.1:8776/review/'
