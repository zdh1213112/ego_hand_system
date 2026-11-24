# 固定手型后的分项对照（20261112）

状态：完成独立参数组对照和“重标定附着→重新拟合手指”对照，**仍未通过手套贴合验收**。没有改变生产标签、相机参数、默认导出或上一版实验。所有方案固定 PERSONAL-3 的手型 betas。

## 为什么重跑一次

旧图像观测取外轮廓，mesh 边界却可能包含内部孔洞。现已统一为外轮廓，生成 `personal_contours_external_v2/observations.json`；原图观测点没有移动。初始轮廓顶点重新从 V3 mesh 提取，旧的 ablation_*_v1 只作诊断历史。

优化仍采用初始外轮廓顶点近似；最终另行逐视角重新光栅化 mesh，取外轮廓像素后评分，避免仅根据固定顶点集合判断改善。剔除腕部的参考平面在同一对照中固定。

## 对照与数据分离

1. rigid：只开放每帧整体旋转和平移，手指和附着固定。
2. fingers：只开放每帧手指姿态，整体位姿和附着固定。
3. attachments：只开放共享三角面内重心坐标、向外法向偏移、腕点局部坐标；面片身份不变。共享参数只用标定时间段的 marker 拟合。此分支不改变 mesh，是负对照，不能用它排除附着误差作为根因。
4. recalibrated_fingers：用第 3 组的附着结果，按第 2 组完全相同条件重新拟合手指，检查附着微调是否有额外收益。

每个手／时刻至少需要两路可用轮廓，确定性留出其中一路相机，其余视角进入本轮姿态优化。共检查左手 48、右手 55 个留出视角。整手／手指分支可拟合验证时间窗内的逐帧参数，因此这次是**视角留出检查**，不是以前“时间窗全部不参与图像优化”的实验。V3 初始化此前见过部分图像，所以仍非完全独立真值评测。

每次只改变指定参数组。整体平移增量上限 8 mm，旋转轴角增量范数 0.06 rad；手指参数每分量增量限制 ±0.12 rad。损失包含 marker、稳健图像轮廓距离和参数增量先验。没有调整相机内外参或对原图去畸变。

## 重新渲染后的留出视角结果

| 方案 | 左手，前→后（px） | 右手，前→后（px） |
|---|---:|---:|
| 仅整手旋转／平移 | 17.10 → 16.32 | 16.28 → 15.63 |
| 仅手指姿态 | 17.10 → 16.15 | 16.28 → 15.41 |
| 仅附着参数 | 17.10 → 17.10 | 16.28 → 16.28 |
| 重标定附着后再调手指 | 17.10 → 16.23 | 16.28 → 15.43 |

这些是自动图像候选上的双向外轮廓距离，不是三维精度。由于视角留出协议与边界评分方法变化，不能直接与上轮 13.23／19.29 px 的表格比较。

- 手指方案平均改善略大于整手位姿方案，但二者都没有消除剩余偏差。
- 手指方案左／右分别有 639／644 个参数分量触及本轮增量限制，不等于这些关节已找到正确角度；不能凭轮廓距离下降继续扩大活动范围。
- 重新标定附着后再调手指，没有比直接调手指更好。只说明当前固定面片、局部偏移和动作覆盖下未获得额外收益，不能排除更根本的附着区域、手型表达或投影误差。
- 重新渲染评分中，手指方案右手有 1 个视角退步超过 1 px；重标定＋手指方案左右各 1 个。没有仅保留变好的画面。完整逐视角数值见 JSON。

结论：本轮受限微调没有解决贴合问题，尚不能唯一定位物理根因。下一步需要更强的独立几何证据，核对每根手指长度／关节位置、固定附着区域及球心投影，或设计专门标定姿态；不能继续把自动轮廓误差当作唯一目标而放任手指变形。

## 查看

页面：http://127.0.0.1:8766/personal_ablation/ 。共 26 张放大对照，包含 632 帧以及有改善、退步的样本；可切换、播放采样图和全屏。不要求用户逐点点击。

从左到右：原图／V3／仅整手位姿／仅手指／重标定附着＋手指。播放每秒 2 张采样图，可能跨相机或左右手，并非原始连续视频。

- `report.json`：所有分支的均值、限制计数、参数隔离检查、静态边界评分及重新渲染均值。
- `rendered_external_contours.json`：重新渲染后的全部留出视角评分，含退步样本。
- `held_camera_isolation_test.json`：人为移动留出相机的全部观测点后，相同 5 步拟合所得姿态和 mesh 完全一致，确认未泄漏进求解。

## 命令

当前实验依赖已有 PERSONAL-3 和轮廓候选，不是新录制的通用生产入口。输出目录必须不存在。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
PY=/home/p3/miniforge3/envs/ego-hand/bin/python
$PY label_export/pipeline/calibration/personal_hand_calibration/ablate_geometry.py --mode rigid   --output output/native_kb_training_20261111/ablation_rigid_new --steps 100
$PY label_export/pipeline/calibration/personal_hand_calibration/ablate_geometry.py --mode fingers   --output output/native_kb_training_20261111/ablation_fingers_new --steps 100
$PY label_export/pipeline/calibration/personal_hand_calibration/ablate_geometry.py --mode attachments   --output output/native_kb_training_20261111/ablation_attachments_new --steps 100
$PY label_export/pipeline/calibration/personal_hand_calibration/ablate_geometry.py --mode fingers   --source output/native_kb_training_20261111/ablation_attachments_new   --output output/native_kb_training_20261111/ablation_recalibrated_fingers_new --steps 100
```

`--observations` 可显式指定候选文件，默认使用统一外轮廓定义的 v2 文件。结果汇总脚本目前固定指向本次 v2 实验目录，避免把新旧结果混在一起。
