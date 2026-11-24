#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$TASK_ROOT"
EXP="$TASK_ROOT/output/native_kb_training_20261111"
CONTEXT="$EXP/explicit_pipeline_sampled_smoke_v2/context.json"
SELECTION="$EXP/keyframe_calibration_v3/selection.json"
SOURCE="$EXP/keyframe_calibration_v3/fit"
RESULT="$EXP/full_silhouette_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${FULL_MASK_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
while (($#)); do
  case "$1" in
    --context) CONTEXT="$2"; shift 2;;
    --selection) SELECTION="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --output) RESULT="$2"; shift 2;;
    --help|-h) echo '完整掩膜共享手型对照：默认使用本机 pair_20261111_192916 的已筛选样本与上一轮标定；--output 指定新目录。也可覆盖 --context、--selection、--source。'; exit 0;;
    *) echo "未知参数：$1" >&2; exit 2;;
  esac
done
[[ ! -e "$RESULT" ]] || { echo "输出目录已存在：$RESULT" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/setup_renderer.py
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/check_renderer.py --output "$RESULT/renderer_checks.json"
for variant in shape_only shape_pose; do
  EXTRA=()
  [[ "$variant" == shape_pose ]] && EXTRA=(--refine-calibration-pose)
  "$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/fit_shape.py --context "$CONTEXT" --selection "$SELECTION" --source "$SOURCE" --output "$RESULT/$variant" "${EXTRA[@]}"
done
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/evaluate.py --context "$CONTEXT" --selection "$SELECTION" --source "$SOURCE" --shape-only "$RESULT/shape_only" --shape-pose "$RESULT/shape_pose" --output "$RESULT/validation.json"
DATASET="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset"])' "$CONTEXT")"
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/build_review.py --previous-review "$SOURCE/review" --selection "$SELECTION" --shape-only "$RESULT/shape_only" --shape-pose "$RESULT/shape_pose" --dataset "$DATASET" --output "$RESULT/review"
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/render_previews.py --review "$RESULT/review" --output "$RESULT/closeups"
for variant in shape_only shape_pose; do
  "$PYTHON_BIN" label_export/pipeline/calibration/personal_hand_calibration/materialize_fit.py --context "$CONTEXT" --source "$RESULT/$variant" --output "$RESULT/mano_fit_$variant"
done
"$PYTHON_BIN" label_export/pipeline/calibration/full_silhouette/verify_artifacts.py --context "$CONTEXT" --selection "$SELECTION" --source "$SOURCE" --output "$RESULT"
echo "完成：$RESULT"
echo "交互对照：$RESULT/review/index.html"
echo "四路近景视频：$RESULT/closeups/"
echo "独立验证：$RESULT/validation.json"
echo "本地查看：$PYTHON_BIN -m http.server 8773 --bind 127.0.0.1 --directory $RESULT"
