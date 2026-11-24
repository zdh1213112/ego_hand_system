#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$TASK_ROOT"
EXP="$TASK_ROOT/output/native_kb_training_20261111"
CONTEXT="$EXP/explicit_pipeline_sampled_smoke_v2/context.json"
AUDIT="$EXP/joint_pose_attachment_contour_audit_v1"
SOURCE="$EXP/explicit_pipeline_sampled_smoke_v2/marker24"
PREVIOUS="$EXP/joint_pose_attachment_all_v1"
RESULT="$EXP/keyframe_calibration_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${KEYFRAME_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
STEPS=180
while (($#)); do
  case "$1" in
    --context) CONTEXT="$2"; shift 2;;
    --audit) AUDIT="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --previous) PREVIOUS="$2"; shift 2;;
    --output) RESULT="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --help|-h) cat <<'HELP'
优质样本筛选 → 个人手型/附着标定 → 固定共享参数 → NOKOV 动作拟合 → 留出验证 → 四路视频。
默认输入为本机 pair_20261111_192916 已完成的分割审核与基线。
用法：bash label_export/pipeline/calibration/keyframe_calibration/run.sh [--output 新目录]
可覆盖：--context FILE --audit DIR --source DIR --previous DIR --steps 180
新录制需先准备对应 context、分割审核和 marker24 输入，不能只换录制目录。
不训练 WiLoR，不改变默认标签路线，产物保留实验状态。
HELP
      exit 0;;
    *) echo "未知参数：$1" >&2; exit 2;;
  esac
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ -e "$RESULT" ]]; then echo "输出目录已存在，请换一个新目录：$RESULT" >&2; exit 2; fi
"$PYTHON_BIN" label_export/pipeline/calibration/keyframe_calibration/select_keyframes.py --context "$CONTEXT" --audit "$AUDIT" --source "$SOURCE" --output "$RESULT"
"$PYTHON_BIN" - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'selection.json').read_text())
for group,parity in [('A',0),('B',1)]:
 e=json.loads(json.dumps(d))
 for s in e['report']['hands'].values():
  s['calibration_sync']=s['calibration_sync'][parity::2];keep=set(s['calibration_sync'])
  s['calibration_views']=[r for r in s['calibration_views'] if r['sync'] in keep]
 (p/f'selection_{group}.json').write_text(json.dumps(e,indent=2))
PY
for variant in fit stability_A stability_B; do
  SELECTION=selection.json
  [[ "$variant" == stability_A ]] && SELECTION=selection_A.json
  [[ "$variant" == stability_B ]] && SELECTION=selection_B.json
  "$PYTHON_BIN" label_export/pipeline/calibration/keyframe_calibration/calibrate.py --context "$CONTEXT" --source "$SOURCE" --selection "$RESULT/$SELECTION" --output "$RESULT/$variant" --steps "$STEPS"
done
"$PYTHON_BIN" label_export/pipeline/calibration/keyframe_calibration/validate.py --context "$CONTEXT" --source "$SOURCE" --selection "$RESULT/selection.json" --fit "$RESULT/fit" --previous "$PREVIOUS" --stability-a "$RESULT/stability_A" --stability-b "$RESULT/stability_B" --output "$RESULT/validation.json"
"$PYTHON_BIN" label_export/pipeline/calibration/keyframe_calibration/build_gallery.py --selection "$RESULT/selection.json" --output "$RESULT/selection_review"
DATASET="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"])' "$CONTEXT")"
"$PYTHON_BIN" label_export/pipeline/calibration/marker24_pilot/build_review.py --dataset "$DATASET" --pilot "$RESULT/fit" --initial-sync 632
"$PYTHON_BIN" label_export/pipeline/calibration/keyframe_calibration/finalize.py --context "$CONTEXT" --output "$RESULT"
echo "完成。优质样本页面：$RESULT/selection_review/index.html"
echo "Mesh 对照：$RESULT/fit/review/index.html"
echo "四路视频：$RESULT/fit/review/camera{0,1,2,3}_comparison.mp4"
echo "固定标定参数：$RESULT/fit/shared_calibration.json"
echo "验证报告：$RESULT/validation.json"
echo "查看：$PYTHON_BIN -m http.server 8772 --bind 127.0.0.1 --directory $RESULT"
