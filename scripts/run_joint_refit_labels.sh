#!/usr/bin/env bash
# 联合重拟合标签：共享手型＋marker贴点重新标定，逐帧姿态放开，marker＋优质多目轮廓共同约束；导出标签并生成四路对照视频。
# 用法：bash scripts/run_joint_refit_labels.sh [输出目录]
# 腕根：以 SDK Finger0 公共根为锚点（每手标定一个固定的掌内偏移），不再使用袖口上的三个腕部 marker。
# 输入可用环境变量覆盖：JOINT_REFIT_CONTEXT、JOINT_REFIT_SELECTION、JOINT_REFIT_SOURCE、JOINT_REFIT_SKELETON、JOINT_REFIT_PREVIOUS（视频对照）；解释器用 EGO_NATIVE_PYTHON。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; CODE="$SCRIPT_DIR/joint_refit"
EXP="${JOINT_REFIT_EXPERIMENT:-$SCRIPT_DIR/../../wilor_training/output/native_kb_training_20261111}"; EXP="$(cd -- "$EXP" && pwd)"
PY="${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}"
CONTEXT="${JOINT_REFIT_CONTEXT:-$EXP/explicit_pipeline_sampled_smoke_v2/context.json}"
SELECTION="${JOINT_REFIT_SELECTION:-$EXP/keyframe_calibration_v3/selection.json}"
SOURCE="${JOINT_REFIT_SOURCE:-$EXP/full_silhouette_v1/shape_only}"
SKELETON="${JOINT_REFIT_SKELETON:-$EXP/nokov_skeleton_audit_v1/aligned_skeleton.npz}"
PREVIOUS="${JOINT_REFIT_PREVIOUS:-$SOURCE}"
RESULT="${1:-$EXP/joint_refit_$(date +%Y%m%d_%H%M%S)}"
# 贴点修正：左右手同一 marker 的贴点相差过大时，统一到更靠手背的一侧（见 symmetrize_attachments.py）。
"$PY" -B "$CODE/symmetrize_attachments.py" --source "$SOURCE" --output "$RESULT/attachments"
# 对齐修正：用原图里检测到的 marker 亮点，逐帧修正动捕到相机的对齐误差（见 image_alignment.py）。
"$PY" -B "$CODE/image_alignment.py" --context "$CONTEXT" --selection "$SELECTION" --source "$RESULT/attachments" --skeleton "$SKELETON" --output "$RESULT/aligned"
"$PY" -B "$CODE/fit.py" --context "$CONTEXT" --selection "$SELECTION" --source "$RESULT/aligned" --output "$RESULT" --anat-weight .01 --normal-min -.002 --final-silhouette --skeleton "$SKELETON" --root-offset free --wrist-marker-weight 0
# 以上只在采样时刻上标定共享手型、贴点和腕根偏移。以下把标定结果用到整段录制的全部时刻：
# 建输入 → 逐帧对齐修正 → 全时刻拟合并自动验收（只保留合格的手帧，剔除原因写入 dense/fit/*_rejected.json）。
"$PY" -B "$CODE/build_dense_source.py" --context "$CONTEXT" --calibrated "$RESULT/final" --skeleton "$SKELETON" --output "$RESULT/dense/source"
"$PY" -B "$CODE/image_alignment.py" --context "$CONTEXT" --selection "$SELECTION" --source "$RESULT/dense/source" --skeleton "$SKELETON" --output "$RESULT/dense/aligned"
"$PY" -B "$CODE/dense_drive.py" --selection "$SELECTION" --source "$RESULT/dense/aligned" --output "$RESULT/dense/fit"
"$PY" -B "$CODE/export.py" --context "$CONTEXT" --source "$RESULT/dense/fit/final"
# 导出标签的图像证据检查：把标签里的 marker 用标签自带的相机参数投到标签自带的图上，与图中亮点比对。
"$PY" -B "$CODE/verify_export.py" --result "$RESULT/dense/fit/final"
"$PY" -B "$CODE/render_video.py" --context "$CONTEXT" --selection "$SELECTION" --previous "$PREVIOUS" --result "$RESULT/dense/fit/final" --output "$RESULT/closeups_final"
echo "完成：$RESULT"
echo "标签：$RESULT/dense/fit/final/export/merged"
echo "验收：$RESULT/dense/fit/summary.json  剔除清单：$RESULT/dense/fit/{left,right}_rejected.json"
echo "查看：$PY -m http.server 8777 --bind 127.0.0.1 --directory $RESULT  →  http://127.0.0.1:8777/closeups_final/"
