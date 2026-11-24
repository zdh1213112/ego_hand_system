#!/usr/bin/env bash
# 代码已移至 ego_hand_system/scripts/joint_refit；此处仅转发，保持旧命令可用。
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../ego_hand_system/scripts" && pwd)/run_joint_refit_labels.sh" "$@"
