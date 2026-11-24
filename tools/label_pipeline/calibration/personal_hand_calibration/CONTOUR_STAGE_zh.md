# 原图轮廓约束实验 PERSONAL-3

状态：已接入 KB 原图轮廓约束并完成受控实验，**未通过个人手型／标签精度验收**。当前正式基线、标定与导出流程均未修改。

## 方法和覆盖

使用真实 KB 内参、畸变和相机外参，直接投影至四路原图，不对图像去畸变。用低饱和度、较高亮度提取本批白色手套的连通区域；NOKOV 投影只用于选择对应手的区域。检查不同阈值下边界稳定性并排除粘连双手、出画、过大背景连通区等。腕部远端裁剪和 mesh 初始轮廓顶点来自上一版模型，所以不是完全独立真值。

共接受 326 份手／帧／相机轮廓候选，剔除 234 份：双手共用连通区 112，出画／深度无效 87，区域过大 23，边界不稳定 11，mesh 边界不足 1。不是 326 张不同原图，两只手可能对应同一图像。

接受的候选中标定 233 份、验证 93 份。验证时间范围仍为 sync [512,704)，含 632 帧。优化仅使用标定图像和标定 marker；可见指尖的独立身份约束尚未接入。

本轮只调整左右手各自的共享 betas（相对上一版每维最多 ±0.5），固定姿态、旋转、平移、附着关系和相机。目标包含稳健的双向轮廓距离、三维 marker 残差和手型先验。mesh 轮廓使用上一版三角形光栅得到的边界顶点集合，未对轮廓拓扑切换求导。该近似及自动分割质量会影响测量。

## 留出图像结果

| 相对于上一版 | 左手 | 右手 |
|---|---:|---:|
| 自动轮廓双向距离，调整前 | 13.23 px | 19.29 px |
| 自动轮廓双向距离，调整后 | 13.15 px | 19.19 px |
| 验证 marker 残差中位数，调整前 | 2.86 mm | 3.17 mm |
| 验证 marker 残差中位数，调整后 | 2.71 mm | 2.97 mm |

轮廓距离是按手／帧／相机等权的平均双向最近距离，由自动提取点和模型轮廓顶点计算；不是手套表面三维精度。左手有一个 beta 增量达到本轮边界，右手没有。**改善很小，不能宣称已经贴合，也不能仅由此断言具体根因。**

结果支持下一步做整体位姿／局部姿态／附着关系的分离对照，而不是只继续调整 betas。仍需检查自动轮廓局限和时空投影误差。没有放宽参数界限来强行降低残差。

## 产物和入口

入口：http://127.0.0.1:8766/personal_contour_calibration/ 。新页面明确标记 PERSONAL-3，旧版本仍保留。

- `review/`：原图／原始基线／当前方案三栏查看，及四路对比视频（每路 20 个共同采样时刻，非连续原速视频）。
- `../personal_contours_v1/index.html`：按相机、左右手均匀选取的轮廓候选抽查页。
- `summary.json`：标定、验证的逐帧误差、beta 变化及限制。
- `checks.json`：数值投影及数据分离检查。
- 左右手 NPZ：拟合参数和 mesh，尚不可作为已验收训练标签。

## 复现

脚本仍针对 pair_20261111_192916 和上一轮 v2 输出，颜色阈值仅适配本批白色手套；不是任意新录制的通用分割器。输出目录必须不存在。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
PY=/home/p3/miniforge3/envs/ego-hand/bin/python
$PY tools/label_pipeline/calibration/personal_hand_calibration/image_constraints.py   --output output/native_kb_training_20261111/personal_contours_new
$PY tools/label_pipeline/calibration/personal_hand_calibration/refine_contours.py   --observations output/native_kb_training_20261111/personal_contours_new/observations.json   --output output/native_kb_training_20261111/personal_calibration_image_new --steps 120
$PY tools/label_pipeline/calibration/marker24_pilot/build_review.py   --pilot output/native_kb_training_20261111/personal_calibration_image_new
```

## 验证

可微 KB 投影对照 OpenCV 的最大误差约 4.55e-13 px（双精度），梯度有限。将全部验证轮廓人为平移，再运行相同的 5 步标定，两只手所得 betas 完全一致，确认验证轮廓未泄漏进求解。检查姿态、附着参数保持不变以及 beta 边界和 mesh 有限性。抽查轮廓预览未发现足以宣布独立精度合格的证据，因此仍保留实验状态。
