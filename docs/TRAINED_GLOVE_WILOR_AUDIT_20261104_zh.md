# 手套微调权重在 ego_hand_system 中的推理检查

## 当前系统的预测方式

`scripts/wilor_multiview_inference.py` 从原始四路视频读取帧，用 YOLO 找手框，按框裁成 256×256，分别测试左右手假设。其 `joints_2d` 由 WiLoR 的虚拟裁剪相机参数 `pred_cam`、固定焦距 `EXTRA.FOCAL_LENGTH=5000` 和 `cam_crop_to_full` 算出。原始相机标定虽在数据集中存在，当前 WiLoR 2D 推理分支没有将每路真实 `K` 传入模型。后续的多视角融合是另一阶段，不改变这一步的单视角投影定义。

新训练的 `physical_camera` 权重是 `wilor_training` 的可组合 adapter，不能直接当作系统的 `--checkpoint` 使用。物理相机头要同时拿到矫正图、对应的矫正内参 `K`、裁剪仿射矩阵和框尺寸；系统常规入口读取的是原始视频。因此新增 `scripts/evaluate_trained_glove_wilor.py` 作为诊断入口，使用 **ego_hand_system 的快速裁剪器、YOLO 检测器和旧投影公式**，并在同一模型前向结果上计算物理相机头投影。它只验证 cam0/cam1 单视角，没有接入四路融合或实时入口。

## 独立录制的结果

- 训练录制：`195844_195852`；留出验证录制：`195732_195743`。
- 验证图：562 张，cam0/cam1 各 281 张；21 个 Nokov marker 作为参考。
- 下表单位为**矫正后整图像素**，按有效 marker 点汇总；P95 是点误差的第 95 百分位。
- “旧投影”与“物理投影”使用**同一份已微调手形权重和同一裁剪**，仅投影方式不同；不是原始预训练 WiLoR 与微调模型的比较。

| 输入框 | 相机 | 有预测的帧 | 物理投影均值 / P95 | 旧投影均值 / P95 |
| --- | --- | ---: | ---: | ---: |
| 标签框 | cam0 | 281/281 | 21.67 / 67.58 | 27.44 / 69.49 |
| 标签框 | cam1 | 281/281 | 19.80 / 52.23 | 23.54 / 56.67 |
| 检测框 | cam0 | 194/281 | 47.73 / 85.23 | 21.91 / 58.56 |
| 检测框 | cam1 | 192/281 | 50.37 / 96.30 | 21.61 / 56.72 |

相同预测映射回**原始 cam0/cam1 画面**并与 Nokov 原始投影对比：标签框下物理投影均值为 cam0 **23.13 px**、cam1 **23.45 px**；检测框下分别为 **51.29 px**、**59.44 px**。原图详细统计见 `work/20261104/ego_195732/trained_wilor_ego_system_full/raw_marker_report.json`。

## 误差来源

1. 在找到检测框的样本里，其与标签框的平均 IoU 为 cam0 **0.721**、cam1 **0.728**。检测框中心的横向偏移中位数为 cam0 **−24.1 px**、cam1 **−25.7 px**；尺寸比中位数为 **1.19**、**1.17**。在相同有检测结果的帧上，物理投影从标签框的 cam0 **18.80 px**、cam1 **19.42 px** 增至检测框的 **47.73 px**、**50.37 px**。这表明物理相机头对当前检测框的中心和尺度变化不够稳健。
2. cam0 有 87/281 张、cam1 有 89/281 张在 `confidence=0.1` 下没有任何检测框；其余帧的误差不能代表这些漏检帧。选中的检测框有些被 YOLO 判成右手（cam0 40/194，cam1 58/192）。本诊断用标签 IoU 选最接近的框，并固定左手假设，以单独衡量定位误差；完整自动关联只会增加其他误差来源。
3. 标签 MANO 的 21 点自身相对 Nokov marker 的矫正图平均误差约 cam0 **5.54 px**、cam1 **5.76 px**。因此模型与 marker 的误差也包含标签拟合误差，不能把全部误差归因于相机头。
4. 两路验证标签中的 `K` 在整段录制内完全相同，并与 `rectification_cam01.npz` 的 `P1/P2` 左上 3×3 完全一致。这次检查没有发现内参文件混用。

当前物理相机头适合标签框条件下的离线分析；若直接用于现有 YOLO 检测框，效果比旧投影更差。下一步应在训练时用真实检测框或模拟其中心、大小分布的框扰动，仍使用相同的矫正图与 `K`，并对独立录制重新验证。检测器的漏检也要单独解决。

## 复现

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
/home/p3/miniforge3/envs/ego-hand/bin/python scripts/evaluate_trained_glove_wilor.py \
  --root /home/p3/data_sda1/EGO_glove/work/20261104/wilor_training_195844_195732_cam01 \
  --adapter /home/p3/data_sda1/EGO_glove/wilor_training/output/ego_nokov_20261104_physical_camera/best_trainable.pt \
  --marker-input /home/p3/data_sda1/EGO_glove/work/20261104/training_labels_195732_195743/mano_input_cam01.npz \
  --output /home/p3/data_sda1/EGO_glove/work/20261104/ego_195732/trained_wilor_ego_system_full
```

输出：`report.json`、逐样本 `predictions.jsonl` 和 `overlays/*.jpg`。示例图中紫色为 Nokov marker，红色为物理投影，黄色为旧投影，蓝框为输入框。
