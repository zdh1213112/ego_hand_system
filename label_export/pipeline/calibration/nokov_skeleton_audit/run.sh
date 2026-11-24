#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$TASK_ROOT"
EXP="$TASK_ROOT/output/native_kb_training_20261111"
CONTEXT="$EXP/explicit_pipeline_sampled_smoke_v2/context.json"
RESULT="$EXP/nokov_skeleton_audit_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${SKELETON_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
if [[ ${1:-} == --output ]]; then RESULT="$2"; shift 2; fi
if (($#)); then echo '用法：bash label_export/pipeline/calibration/nokov_skeleton_audit/run.sh [--output 新目录]' >&2; exit 2; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" label_export/pipeline/calibration/nokov_skeleton_audit/analyze.py --context "$CONTEXT" --output "$RESULT"
"$PYTHON_BIN" label_export/pipeline/calibration/nokov_skeleton_audit/build_review.py --context "$CONTEXT" --audit "$RESULT" --mano "$EXP/full_silhouette_v1/shape_only" --images "$EXP/joint_pose_attachment_contour_audit_v1/images" --output "$RESULT/review"
echo "报告：$RESULT/report.json"
echo "原图核对：$RESULT/review/index.html"
echo "查看：$PYTHON_BIN -m http.server 8774 --bind 127.0.0.1 --directory $RESULT"
