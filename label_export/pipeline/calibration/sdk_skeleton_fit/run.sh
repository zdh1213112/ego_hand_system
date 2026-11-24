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
RESULT="$EXP/sdk_skeleton_assumed_internal_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${SDK_FIT_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
ASSUME=false
while (($#)); do
  case "$1" in
    --assume-sdk-internal-offsets) ASSUME=true; shift;;
    --output) RESULT="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --previous-review) PREVIOUS_REVIEW="$2"; shift 2;;
    --skeleton) SKELETON="$2"; shift 2;;
    --selection) SELECTION="$2"; shift 2;;
    --help|-h) echo 'SDK骨架假设标签对照。必须显式添加 --assume-sdk-internal-offsets。默认复用本机192916录制的标定、骨架审核和个人手型；可用 --output 指定新目录。'; exit 0;;
    *) echo "未知参数：$1" >&2; exit 2;;
  esac
done
[[ "$ASSUME" == true ]] || { echo '必须显式传入 --assume-sdk-internal-offsets；该选项仅启用假设，不修改 NOKOV 配置。' >&2; exit 2; }
[[ ! -e "$RESULT" ]] || { echo "输出目录已存在：$RESULT" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for variant in hand finger0; do
  "$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/fit.py --context "$CONTEXT" --source "$SOURCE" --skeleton "$SKELETON" --output "$RESULT/sdk_$variant" --root "$variant" --assume-sdk-internal-offsets
  "$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/export_labels.py --context "$CONTEXT" --source "$RESULT/sdk_$variant" --skeleton "$SKELETON"
  "$PYTHON_BIN" label_export/pipeline/calibration/marker_placement/build_label_viewer.py --dataset "$RESULT/sdk_$variant/export/merged" --output "$RESULT/sdk_$variant/export/viewer" --standalone --initial-sync 632
done
DATASET="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset"])' "$CONTEXT")"
"$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/build_review.py --previous-review "$PREVIOUS_REVIEW" --sdk-hand "$RESULT/sdk_hand" --sdk-finger0 "$RESULT/sdk_finger0" --dataset "$DATASET" --output "$RESULT/review"
"$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/evaluate.py --context "$CONTEXT" --selection "$SELECTION" --source "$SOURCE" --sdk-hand "$RESULT/sdk_hand" --sdk-finger0 "$RESULT/sdk_finger0" --output "$RESULT/comparison_metrics.json"
for variant in hand finger0; do
  "$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/render_previews.py --review "$RESULT/review" --output "$RESULT/closeups_$variant" --root "$variant"
done
"$PYTHON_BIN" label_export/pipeline/calibration/sdk_skeleton_fit/verify_results.py --source "$SOURCE" --skeleton "$SKELETON" --output "$RESULT"
echo "完成：$RESULT"
echo "交互对照：$RESULT/review/index.html"
echo "实际标签：$RESULT/sdk_{hand,finger0}/export/merged/"
echo "四路视频：$RESULT/closeups_{hand,finger0}/"
echo "本地查看：$PYTHON_BIN -m http.server 8775 --bind 127.0.0.1 --directory $RESULT"
