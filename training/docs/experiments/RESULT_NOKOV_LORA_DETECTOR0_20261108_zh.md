# Nokov 手套数据：detector0 与 ViT LoRA 实验结果

实验数据：`195844_195852` 训练（2979 张，左手）；`195732_195743` 验证（562 张，cam0/cam1 各 281 张）。训练与验证按录制分开。验证数据已用于多次模型选择，最终泛化结论还需要一段新的录制。

## 训练

1. 原基线 `ego_nokov_20261104_original`：冻结 ViT，训练 coarse heads 和 RefineNet，2000 步。
2. `ego_nokov_20261104_joint_lora`：从原始 `wilor_final.ckpt` 训练 ViT LoRA、coarse heads 和 RefineNet，2000 步；最佳验证 loss 在第 1850 步。LoRA 覆盖 32 个 ViT block 的注意力和 MLP，rank 16，原始 ViT 权重冻结。
3. `ego_nokov_20261104_joint_lora_posttrain`：从 LoRA 最佳权重继续，只训练 coarse heads 和 RefineNet，冻结 ViT 与 LoRA，1000 步；该阶段最佳 loss 在第 250 步。

完整留出录制，**标签框裁剪**：

| 权重 | 手腕对齐 3D MPJPE | 矫正图到 MANO 标签平均 2D 误差 | 原图到物理 marker cam0 均值 / P95 | cam1 均值 / P95 |
| --- | ---: | ---: | ---: | ---: |
| 冻结 ViT 基线 | 14.25 mm | 25.17 px | 25.59 / 53.02 px | 24.80 / 51.16 px |
| ViT LoRA 联合训练 | **8.38 mm** | **22.28 px** | 22.08 / **46.41 px** | 21.76 / **45.54 px** |
| LoRA 后训练 | 8.40 mm | 22.30 px | **22.01** / 46.74 px | **21.68** / 45.84 px |

后训练仅改善约 0.07–0.08 px 的 marker 平均误差，同时 3D、MANO 2D 和 marker P95 略退化。当前推荐 **LoRA 第一阶段** `output/ego_nokov_20261104_joint_lora/best_trainable.pt`。

## detector0.pt 实际框

使用同一组矫正图、同一个 `detector0.pt`。诊断脚本以标签框 IoU 选择最接近的检测框，固定左手假设，借此单独测框定位与模型误差；这比完整自动关联更有利。实时入口还使用跟踪器，当前诊断没有模拟跟踪。

| 阈值 | 相机 | 有任意框 | IoU ≥ 0.5 | 原基线在合格框的原图 marker 均值 / P95 | LoRA 在合格框的均值 / P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.1 | cam0 | 236/281 | 213/281 | 24.01 / 52.49 px | **22.24 / 49.04 px** |
| 0.1 | cam1 | 250/281 | 250/281 | 23.19 / 49.57 px | **21.24 / 46.98 px** |
| 0.3（实时默认） | cam0 | 191/281 | 143/281 | 27.25 / 57.31 px | **23.89 / 53.52 px** |
| 0.3（实时默认） | cam1 | 220/281 | 218/281 | 23.80 / 50.45 px | **21.66 / 48.18 px** |

cam0 在 0.3 阈值下有 48 个“有框但 IoU < 0.5”的样本，还有 90 个无框样本。LoRA 可以降低有正确框时的姿态误差，但检测阶段的缺口仍是完整系统的主要限制。`detector0.pt` 在 0.1 阈值下比 `detector_1009.pt` 找到更多框，cam0 也产生更多严重错框。原始含错框均值非常不稳定，应连同检测覆盖率、合格框数量、P95 一起查看。

物理相机头在标签框上的 MANO 2D 表现较好，但使用 `detector0.pt` 框时原图 marker 误差明显高于 LoRA 虚拟投影；当前不推荐将物理相机头接入常规检测框推理。LoRA 权重预测的是 MANO 21 关节，物理 marker 是独立评估目标，两者的标签拟合差异仍存在。

## 文件与复现

- 第一阶段配置：`configs/glove/ego_nokov_20261104_joint_lora.yaml`；权重和训练记录：`output/ego_nokov_20261104_joint_lora/`。
- 第二阶段配置：`configs/glove/ego_nokov_20261104_joint_lora_posttrain.yaml`；权重和训练记录：`output/ego_nokov_20261104_joint_lora_posttrain/`。
- 检测框诊断入口：`../ego_hand_system/scripts/evaluate_trained_glove_wilor.py`；各实验的 `report.json`、`raw_marker_report.json`、逐图预测及叠加图在 `../work/20261104/ego_195732/` 下对应实验目录。
- 机器可读四模型对比：`output/ego_nokov_20261104_joint_lora/experiment_comparison.json`。

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --mode joint_lora \
  --config configs/glove/ego_nokov_20261104_joint_lora.yaml
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py \
  --config configs/glove/ego_nokov_20261104_joint_lora_posttrain.yaml
```
