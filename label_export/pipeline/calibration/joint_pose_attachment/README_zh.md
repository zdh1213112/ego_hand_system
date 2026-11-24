# 每帧动作＋共享 marker 附着联合修正（JOINT-PA-1）

**后续轮廓检查发现 camera0 / sync632 右手包含背景横梁，且错误边缘实际进入了优化。此前自动轮廓指标不能证明贴合改善，结果需要修复观测后重新评估。** [检查实际轮廓](http://127.0.0.1:8766/contour_audit/) 。

已完成独立实验，原来的分阶段结果继续保留。本轮使用当前查看器中的低迭代链路结果 `explicit_pipeline_sampled_smoke_v2/contour_fit` 初始化，固定其左右手各自的共享 betas，同时优化每帧整手旋转、平移、45 维手指姿态，以及跨帧共享的 marker 重心坐标、有效外法向偏移和腕点局部坐标。固定三角面编号，避免附着点跳到其他手指。

## 查看效果

- 修正前后对比：http://127.0.0.1:8766/joint_pose_attachment/ 。默认 camera0 / sync632，三栏分别为原图、修正前的当前标签、联合修正结果。
- 实际新标签：http://127.0.0.1:8771/ 。共 488 张原图、976 份手标签。
- `review/camera0_632_right_crop.jpg`：632 帧右手放大对比。
- `review/camera0_comparison.mp4` ～ `camera3_comparison.mp4`：四路采样对比视频，每路 174 个共同采样时刻；3.75 张／秒，并非原视频连续速度。

## 求解方式

两组实验采用相同初值和限制，各运行 160 次：

1. `all`：使用每帧全部可用相机轮廓，生成当前展示及导出的结果。
2. `leave_one_out`：每帧确定性留一路相机不参与本轮求解，用该视角检查增量是否有效。

共同目标包含 NOKOV24 残差、多视角外轮廓损失、相对初值的位姿与附着先验，以及相邻采样帧的增量连续性。轮廓损失批量可微计算，迭代中刷新一次可见外轮廓顶点；最终另外重新渲染外轮廓评分。

共享附着参数只接收标定时间戳的直接 marker 梯度；非标定帧中附着参数取 detach 值，但姿态仍可调整。时序项可能跨相邻的标定／非标定样本，因此这不是严格独立的时间留出验证。相机留出检查只证明本轮增量的跨视角表现，早期初始化可能已见过相关图像。

只有至少两路轮廓候选可用的帧参与：左手 127、右手 134 个采样帧。其他帧保留原来的动作拟合，并在导出索引中明确记录 `joint_pose_attachment_frame_optimized: false`。976 个导出标签中 639 个手实例对应参与联合求解的帧；其他实例不冒充已完成本轮动作优化。

本轮没有放宽既有保护范围：平移增量 8 mm、旋转轴角增量范数 0.06 rad、手指参数各分量增量 ±0.12 rad；法向有效偏移限制 0～4 mm，腕点局部增量 ±2 mm。没有固定减去 2 mm。

## 重新渲染评分

| 指标 | 左手：前→后 | 右手：前→后 |
|---|---:|---:|
| 当前展示结果，全部参与视角的轮廓距离 | 10.12 → 8.20 px | 12.43 → 9.71 px |
| 另一次实验，未参与求解的相机轮廓距离 | 10.69 → 8.90 px | 12.56 → 9.93 px |
| NOKOV24 残差中位数，全部视角方案 | 6.43 → 4.52 mm | 6.34 → 4.37 mm |

自动轮廓和 marker 残差都不能直接等同于 mesh 的真实三维精度。共享手型本轮固定，未解决手套厚度、局部宽度表达等所有问题。

左手有 2 个视角退步超过 1 px，都在 sync1184（camera2：8.40→10.10 px；camera1：8.59→9.84 px）；右手没有退步超过 1 px 的视角。完整逐视角结果保存在 `rendered_scores.json`，没有只保留改善帧。左右手分别有 1402／1760 个手指参数分量触及增量界限，不能因此继续无约束扩大弯曲范围。

## 输出与参数

目录：`output/native_kb_training_20261111/joint_pose_attachment_all_v1`

- `left_pilot.npz`、`right_pilot.npz`：联合修正后的实际参数、mesh、球心预测及共享附着。
- `joint_fit_mask`：NPZ 内记录哪些帧参加本轮联合修正。
- `summary.json`：参数变化、约束触界和参与帧。
- `rendered_scores.json`：原结果、全部视角方案、留出相机方案的重新渲染评分。
- `mano_fit/`：经参数重建核验的导出格式。
- `export/merged/`：新标签，带联合修正标识及每帧参与状态；`approved_for_training` 仍为 false。
- `checks.json`：隔离检查及实际导出记录。

## 显式开启统一路线

原配置保持原来的分阶段模式。新增配置：`ego_hand_system/configs/native_personal_marker24_joint.json`，其中：

```json
"joint_pose_attachments": {
  "enabled": true,
  "steps": 160,
  "view_policy": "all"
}
```

执行：

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
bash scripts/export_ego_nokov_labels.sh   --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916   --optimization-config configs/native_personal_marker24_joint.json
```

统一入口先执行原分阶段优化，再运行联合动作／附着阶段，最后从 `optimized/joint_fit` 导出；不是将分项诊断结果自动当作联合结果。参考帧与留出窗口仍针对当前录制。

## 验证

14 项投影、参数组变化、边界和配置测试通过；10 项原生数据／合并回归通过。人为移动全部留出相机的观测，再运行相同 5 步，两只手的动作、附着和 mesh 完全一致，确认这些观测没有进入求解。新标签通过逐标签 MANO 重建、原图投影和合并检查。浏览器已核验默认 632 帧及修正前／后列的来源。
