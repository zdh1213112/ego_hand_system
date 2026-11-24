# WiLoR、GEM-X、FAH 统一标签评估

## 评估协议

`evaluate_hand_models.py` 执行**给定标注框和左右手身份的单帧姿态评估**。
多视角融合结果使用下文的 shared 评估入口。
推理接口只接收图像、框、左右手标识和相机标定；标签 MANO 参数、三维关节、网格和平移
均不传入模型。每种模型沿用自身裁剪实现；WiLoR/GEM 的扩框倍率由 `--crop-scale` 控制，
默认 2.0，FAH 使用原生裁剪策略。

| 指标 | 定义 |
| --- | --- |
| `root_relative_mpjpe_mm` | 预测和标签分别减去腕部，再计算 21 点平均欧氏距离；包含腕部零误差 |
| `pa_mpjpe_mm` | 每手相似变换对齐后平均关节误差，允许旋转、平移、尺度，不允许反射 |
| `absolute_mpjpe_mm` | 相机坐标系平均关节误差，包含全局位置误差 |
| `wrist_error_mm` | 相机系腕部位置误差 |
| `mean_2d_error_px` | 模型二维预测与 21 点标签真实镜头投影的平均像素误差 |
| 3D PCK@20/30/50 | 对应绝对／腕部对齐／PA 误差小于等于阈值的关节比例 |
| 3D AUC 0–50 mm | 对经验 PCK 曲线在 0–50 mm 上精确积分并归一化到 [0,1] |
| 2D PCK@5/10/20 px | 原始像素阈值下的关节正确率，不是按框大小归一化的 PCK |
| MPVPE | WiLoR/FAH 的同拓扑 778 点网格误差；提供绝对、腕部对齐、关节 PA 变换后的版本 |

## 数据约定和标签来源

读取器验证共享图像索引、唯一标签 ID、左右手、相机参数、图像尺寸和网格投影一致性。
`joints_3d + trans` 才是相机系关节点；`trans` 不是腕部。
`geometry_space=physical_side` 的左手已完成物理镜像，不能再次翻转 x。
NPY 中名为 `joints_2d` 的 `(778,2)` 数组是网格顶点投影，
21 点二维标签从三维关节和实际镜头模型重新计算。
NPY 为含 torch.Tensor 的对象字典，读取时使用 pickle，仅用于可信的本地导出文件。

任一三维点非有限值时整手计为姿态失败。失败不进入平均三维误差，但仍在 PCK/AUC 和覆盖率
的分母中；平移求解失败不会抹掉可用的腕部对齐姿态，退化 PA 也单独计失败。
提供总体、按相机、左右手、source_group 以及按关节点统计。
每个相机／帧／手标签等权，因此同一物理时刻可有多个视角样本。
## 输出与续跑

每种模型输出：

```text
run_config.json       数据／模型／代码指纹、环境、协议
predictions/*.npz     按图像块保存的数值预测，含样本 ID、21 点、二维点、可用 MANO 网格
per_sample.csv        每手误差和成功状态
joint_errors.npz      每关节误差、有效标签掩码、样本 ID
summary.json          总体、相机、左右手、来源和逐关节统计
report.html           可直接打开的报告
report.csv            汇总表
```

`--backends` 在父目录额外输出 `comparison.json`、`comparison.html`、`comparison.csv`，
以及 `comparison_common_samples.json`（各模型共同成功的姿态样本上配对统计）。
对比入口检查相同的数据指纹、样本 ID、有效掩码和评估协议。

## 多视角共享模型评估

`evaluate_shared_hand_models.py` 直接读取原实验的 `shared_pose.pt`、`accepted.jsonl`、
`summary.json` 和 `normalized_multiview`。WiLoR／FAH 使用每手、每时刻唯一的共享 MANO，
GEM 使用共享 SOMA。共享是指同一物理手在所有视角共用模型参数，不是左右手或整段录像共用一个姿态。

评分使用保存的最终三维关节和网格，经 `T_base_camera` 转到标注相机，再通过实际 KB／DS／
针孔模型投影。绝对误差保留融合结果的全局位置，不执行单目平移重估；二维指标比较最终共享三维
投影与标签关节投影，不读融合前的二维网络输出。腕部对齐、PA、PCK、AUC 和 MANO 网格指标
沿用上表定义。GEM 的 SOMA 网格没有 MANO 顶点对应，MPVPE 不适用。

先按参考纳秒时间戳匹配，再检查相机帧号和内外参，不依赖不同导出间可能重新编号的 `sync_index`。
每个被接受的手模型必须至少有两个标定视角支持；所有有标签的视角都会评分，包括未参加该次
融合的视角。未检出、融合拒绝、未处理和不在实验范围内的标签均保留为失败，计入覆盖率和
PCK／AUC 分母，不回退到单视角预测。均值仅在有预测的样本上计算，应同时查看覆盖率。

主表按相机／手标签等权；`summary.json` 的 `unique_hand_frames` 另外按参考时间戳和物理左右手
去重，避免一个时刻因导出较多视角而权重更大。去重前检查跨视角标签转换到基准系后的一致性。
