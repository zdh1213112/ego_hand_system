# EGO + Nokov 训练与投影链路

## 数据到原始 WiLoR 模型

1. 同步 EGO cam0/cam1 和 Nokov，并将 Nokov 的 21 个物理 marker 放到相机坐标系；生成 `training_labels_*/mano_input_cam01.npz`。该文件同时保存 marker 的相机空间 3D 坐标和两路矫正图上的 2D 坐标。
2. 用 marker 轨迹拟合 MANO，生成每张图的 `joints_3d`、`trans`、相机内参 `K` 和框 `bbox`。图片和 `K` 都属于**矫正后的** cam0/cam1。
3. 合并数据集：`work/20261104/wilor_training_195844_195732_cam01`。`195844` 录制用于训练，`195732` 录制用于验证，保证录制级别隔离。
4. `GloveDataset` 读取矫正图，按标注框扩大并裁成 256×256；左手镜像成 WiLoR 的统一输入方向。MANO 3D 经过 `trans` 和 `K` 投影为裁剪图 2D 标签。
5. 原始配置 `configs/glove/ego_nokov_20261104_original.yaml` 从 `wilor_final.ckpt` 训练：冻结 ViT，更新 coarse pose/shape/camera heads 和 RefineNet；损失包含根相对 3D、裁剪图 2D 和 MANO 参数。原模型的 2D 输出来自固定虚拟焦距的相机，不直接表示真实 cam0/cam1 深度。

## 新增相机与 marker 阶段

- `joint_lora_finetune`：从原始 WiLoR 权重联合训练 ViT LoRA、coarse heads 和 RefineNet。ViT 原参数冻结，LoRA 在注意力与 MLP 层适配手套图像。
- `joint_lora_posttrain`：从 LoRA 第一阶段最佳权重继续，冻结 ViT 与 LoRA，只训练 coarse heads 和 RefineNet。
- `physical_camera_finetune`：从原始最佳权重开始，冻结手形网络，只训练物理相机头。它使用每张图的 `K`、裁剪变换、图像宽度和左右手信息，预测相机空间手腕位置，做真实透视投影；以 MANO 2D 标签和相机空间手腕 3D 位置监督。
- `marker_offset_finetune`：接着冻结手形网络和物理相机头，只学习 MANO 21 点到 Nokov 21 个物理 marker 的局部 3D 偏移及手腕偏移。它同时使用 marker 2D、根相对 3D 和手腕 3D 监督。训练集中 cam1 有 7 张不完整 marker 样本，在该阶段剔除；验证集 562 张均保留。
- `marker_camera_finetune` 是相机头与 marker 偏移同时训练的实验配置。留出录制上相机位置退化，不应把该实验检查点用于最终推理。

## 本次留出录制验证（195732，cam0/cam1 各 281 张）

| 权重 | cam0 原图到物理 marker 平均 / P95 | cam1 原图到物理 marker 平均 / P95 |
| --- | ---: | ---: |
| 原始 WiLoR 微调 | 25.59 / 53.02 px | 24.80 / 51.16 px |
| 物理相机头 | 23.29 / 58.89 px | 23.58 / 56.51 px |
| 物理相机头 + marker 偏移 | 24.43 / 61.29 px | 26.60 / 64.43 px |
| ViT LoRA 联合训练 | 22.08 / **46.41 px** | 21.76 / **45.54 px** |
| LoRA 后训练 | **22.01** / 46.74 px | **21.68** / 45.84 px |

物理相机头把 MANO 标签的整图平均误差从 25.17 px 降到 20.00 px，但物理 marker 的 P95 变差；使用 `detector0.pt` 检测框时，它的 marker 误差也明显高于 LoRA。ViT LoRA 联合训练把手腕对齐 3D MPJPE 从 14.25 mm 降至 8.38 mm，marker 均值和 P95 均下降。后训练只带来约 0.07–0.08 px 的 marker 均值改善，3D 和 P95 略退化。**当前推荐 LoRA 第一阶段检查点** `output/ego_nokov_20261104_joint_lora/best_trainable.pt`。这仍是 MANO 关节预测，不能据此认为预测的 21 关节与物理 marker 严格重合。检测框覆盖率和错框仍限制完整系统效果；详见 [detector0 与 LoRA 完整评估](../experiments/RESULT_NOKOV_LORA_DETECTOR0_20261108_zh.md)。

训练命令（本机环境）：

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --config configs/glove/ego_nokov_20261104_original.yaml
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --config configs/glove/ego_nokov_20261104_physical_camera.yaml
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --config configs/glove/ego_nokov_20261104_marker_offset.yaml
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --mode joint_lora --config configs/glove/ego_nokov_20261104_joint_lora.yaml
/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --config configs/glove/ego_nokov_20261104_joint_lora_posttrain.yaml
```

验证 marker 模型：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python evaluate_glove.py \
  --root /home/p3/data_sda1/EGO_glove/work/20261104/wilor_training_195844_195732_cam01 \
  --adapter output/ego_nokov_20261104_marker_offset/best_trainable.pt \
  --marker-input training_labels_195732_195743=/home/p3/data_sda1/EGO_glove/work/20261104/training_labels_195732_195743/mano_input_cam01.npz \
  --output output/ego_nokov_20261104_marker_offset/val_full_best \
  --device cuda:0 --batch-size 8 --num-workers 4 --save-overlays
```

该验证使用标签框裁剪输入，不包含手部检测框误差；评估图像是矫正图。`metrics.json` 的 2D 误差以**矫正后整图像素**计，`mpjpe_mm` 是手腕对齐后的 3D 误差；`mean_camera_root_error_mm` 则单独衡量绝对相机空间位置。原图上的 marker 对照由 `compare_marker_projection.py` 将预测通过矫正映射返回原图后计算。必须同时检查平均值与 P95，不能只看训练 loss 或根相对 MPJPE。
