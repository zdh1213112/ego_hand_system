# GEN 六目 SAM3 手套掩膜辅助 WiLoR

该功能将外部 `auto-obb-annotator` 中的 SAM3 防护手套分割器接入 GEN 六目流程。SAM3
不直接生成 21 个关节点，而是给每个 YOLO detector 框和 WiLoR 左右手假设附加：

- SAM3 掩膜置信度；
- SAM3 掩膜框与 YOLO 框的 IoU；
- WiLoR 21 点落在手套掩膜内的比例（带 8 px 容差）。

默认还启用一个保守的 2D 修正：只有非腕点跑到掩膜容差之外、且在
`--sam3-max-2d-correction-px` 范围内能找到真实掩膜像素时，才将该点向掩膜边界小幅
拉回。原始点保存在 `wilor_joints_2d`，供 A/B 对照；下游融合使用修正后的
`joints_2d`。腕点不自动拉回，因为手套分割经常不包含袖口/皮肤连接处。

这些证据进入六目融合的假设选择和逐视角一致性门控。只有当至少两个其它视角仍然支持
同一只手时，才会剔除一个明显不符合手套掩膜的视角；掩膜缺失时保持原来的 WiLoR/几何
路径，不会因为遮挡而强行删除结果。因此这是一个保守的检测、二维骨架和三维融合辅助。

## 单条记录

使用新的输出目录运行完整流程（`--max-frames 60` 可先做冒烟测试）：

```bash
cd /home/p3/data_sda1/ego_hand_system
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/record.mcap \
  --output /path/to/sam3_run \
  --conda-env ego-hand \
  --device cuda \
  --max-frames 60 \
  --sam3-glove-assist 1 \
  --sam3-conda-env annotator \
  --sam3-model /home/p3/data_sda1/tool/auto-obb-annotator/models/sam3.pt \
  --sam3-prompt "hand wearing a gray and black protective work glove"
```

SAM3 默认使用 **YOLO 框 + 文本提示**，比整幅图的文本搜索更适合当前 egocentric 画面。
如需只测试文本提示，可添加 `--sam3-text-only 1`。不同手套外观应替换
`--sam3-prompt`，但提示词最好同时包含 `hand` 和 `glove`。

2D 修正默认参数为 `--sam3-refine-2d 1 --sam3-max-2d-correction-px 24
--sam3-2d-correction-blend 0.75 --sam3-max-2d-bone-change-ratio 0.55`。如只想验证掩膜而不改 2D 点，可加
`--sam3-refine-2d 0`；如需更保守，可降低 correction blend。

输出结构：

```text
sam3_run/
├── wilor_multiview/                         # 原始 WiLoR，不改动
├── wilor_multiview_sam3_assisted/           # 2D 修正 + sam3_assist 字段
│   ├── summary.json
│   └── camera*/predictions.jsonl
└── fusion_multiview_sam3_assisted/
    ├── accepted.jsonl
    ├── rejected.jsonl
    ├── summary.json
    ├── diagnostic_6view.mp4
    ├── sam3_2d_only_6view.mp4               # 只看修正后的相机内 2D
    ├── sam3_segmentation_6view.mp4          # SAM3 手套掩膜叠加（独立渲染）
    └── final_only_6view.mp4                 # 添加 --final-video 才生成
```

`fusion_multiview_sam3_assisted/summary.json` 中的
`sam3_matched_view_count`、`sam3_accepted_view_count` 和
`sam3_rejected_matched_view_count` 用于检查掩膜门控是否实际参与；
`wilor_multiview_sam3_assisted/summary.json` 中的 `corrected_hypotheses`、
`corrected_joints` 用于检查 2D 修正是否实际触发。每个
`accepted.jsonl -> hands -> views -> cameraX -> sam3_assist` 保存该视角的详细分数。
原始输出与 SAM3 输出应使用相同的 normalized 数据和帧范围进行 A/B 比较。

只看修正后的相机内 2D 骨架（不叠加 3D）时：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/sam3_run/fusion_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/sam3_2d_only_6view.mp4 \
  --overlay-mode sam3-2d-only --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

如需直接查看 SAM3 的像素级手套掩膜、YOLO 框、SAM 框和修正后的 2D 骨架，使用独立
渲染器：

```bash
conda run --no-capture-output -n annotator \
  python scripts/render_sam3_glove_multiview.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --predictions /path/to/sam3_run/wilor_multiview \
  --assisted-predictions /path/to/sam3_run/wilor_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/sam3_segmentation_6view.mp4 \
  --tool-root /home/p3/data_sda1/tool/auto-obb-annotator \
  --model /home/p3/data_sda1/tool/auto-obb-annotator/models/sam3.pt \
  --prompt "hand wearing a gray and black protective work glove" \
  --device cuda --max-frames 0 \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

主辅助阶段只保存掩膜统计和关节点结果，不保存全分辨率掩膜像素；因此该可视化命令会
按相同提示词重新运行 SAM3，但不会修改任何预测或融合文件。

主要阈值：

```text
--sam3-conf 0.25
--sam3-min-score 0.25
--sam3-min-box-iou 0.20
--sam3-min-joint-support 0.55
--sam3-hard-joint-support 0.35
```

`min-joint-support` 是“可信假设”门限，`hard-joint-support` 是“在其它相机一致时才可
剔除该视角”的更低门限。建议先保持默认值；如果遮挡较多导致误删，可降低
`--sam3-min-joint-support`，不要直接关闭六目几何和时序门控。

## 批量处理

```bash
./scripts/run_multiview_wilor_batch.sh \
  --input-dir /home/p3/data_sda1/ego_hand_system/recordings/20260907 \
  --output-root /home/p3/data_sda1/ego_hand_system/output/gen6_sam3_batch \
  --device cuda \
  --gpu-profile rtx5090d \
  --max-frames 0 \
  --sam3-glove-assist 1 \
  --sam3-conda-env annotator
```

SAM3 会为每个相机、每个同步帧运行一次，速度明显慢于纯 WiLoR。推荐先用 60 帧检查
`summary.json` 和视频，再处理完整记录。脚本会保留原始 WiLoR 预测，方便在没有实际
改善时安全回退。
由于 SAM3 本身使用 GPU，批量脚本在启用该开关时会自动关闭 `rtx5090d` 的并行流水线，
避免下一条记录的 WiLoR 推理和当前记录的 SAM3 同时争用显存。

## 当前作用边界

SAM3 可改善错误检测、左右手假设选择、明显越出手套区域的 2D 点、错误相机视角剔除和
由此产生的 3D 深度跳变；它不能单独恢复所有相机同时错误的姿态，也不能替代 WiLoR 的
关节点回归或相机标定。最终
是否提升应比较同一记录的检测保留率、六目重投影误差、3D 掌部步长/加速度和人工抽帧结果。
