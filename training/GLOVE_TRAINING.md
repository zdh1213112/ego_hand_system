# WiLoR 手套训练

## 原图 KB 上的 NOKOV 表面 marker 监督

在 `joint_lora_finetune` 配方中设置 `recipe.camera_model: physical_kb` 和正值 `losses.surface_markers_3d`，可让实测点直接监督预测的 MANO 网格。每个录制在 `dataset_profile.json` 的 `nokov_surface_archives` 中列出对应的 NPZ 路径及 SHA-256。归档保存每手的 20 个光学点、四相机外参，以及 marker 在网格上的三角形、重心坐标和局部偏移。录制标识与清单中的 `source_group` 一致。

模型按真实 KB 相机几何把预测网格与实测点放到同一坐标系，分别计算手形误差和整体位置误差。带 `measured_surface_only: true` 的训练样本只提供实测目标，其旧 MANO、腕位和关节不参与监督。可在清单中用 `input_bbox_xyxy` 指定检测框，`data.crop_jitter` 仅在训练时扰动输入框。推理使用 RGB 与相机标定。

## 最新已汇总标签：trainset_v1

本机 2026-11-14 准备的原图 KB 标签位于 `datasets/trainset_v1`，包含两段录制、
13874 个手实例，按每段录制的时间划分为 12468 个训练实例和 1406 个验证实例。
对应的联合 LoRA 配置和命令为：

```bash
python train_glove.py --mode joint_lora --config configs/glove/trainset_v1_joint_lora.yaml
```

该配置显式启用 `physical_kb`，沿用已有 4000 步实验参数，重跑输出到
`output/trainset_v1_joint_lora/`。所需 `coarse_start.ckpt` 初始化、数据质量状态及
已有权重见 [最新标签训练说明](docs/training/START_TRAINSET_V1_zh.md)。
下节为 11 月 4 日历史数据的联合 LoRA 流程；单独使用 `--mode joint_lora` 默认仍走该旧配置。

## 新增：NOKOV 联合 LoRA 微调

已从 `/home/p3/data_sda1/WiLoR` 接入 `joint_lora_finetune`。使用现有相机对齐的NOKOV→MANO标签，联合训练ViT LoRA、pose/shape/camera heads和RefineNet；原始ViT与MANO权重冻结。无需先跑OpenTouch Stage 1。

```bash
python train_glove.py --mode joint_lora --config configs/glove/ego_nokov_20261104_joint_lora.yaml
```

本机使用 `/home/p3/miniforge3/envs/ego-hand/bin/python`。`--mode joint_lora` 不给config时也默认读取上述EGO配置；通用模板 `configs/glove/joint_lora_finetune.yaml` 需先修改数据路径。
当前配置为相同的2979张训练/562张场次隔离验证、batch4×累积8、2000步。沿用源方案的32个block、rank16/alpha32/dropout0.05，LoRA/head/RefineNet学习率1e-5/1e-4/1e-5。
后台启动及实测证据见 [联合训练说明](docs/training/START_NOKOV_JOINT_LORA_zh.md)。原有冻结ViT基线保留。
`lora_stage1/lora_stage2/two_stage` 入口也已接入，但需要另外准备OpenTouch数据或Stage 1权重；本次真实GPU验证的是joint_lora。

## 冻结 ViT 的基线：自有 Ego/Nokov 标签

使用 `configs/glove/refiner_finetune.yaml`：`checkpoint` 指向原始 `wilor_final.ckpt`，`init_weights: null`；`refiner_finetune` 会冻结 ViT，只训练 coarse pose/shape/camera heads 和 RefineNet。先将 `paths.dataset_root` 设为导出的相机对齐数据目录，再运行 `python train_glove.py --config configs/glove/refiner_finetune.yaml`。需要 `joints_3d`、`trans`、`K`、`bbox`、`side` 等标签；若图像已去畸变，`K` 和 bbox 必须对应去畸变图像。

NOKOV 21 点要生成本项目严格的完整标签时，先在 `ego_hand_system` 使用
`prepare_nokov21_mano_input.py`、`fit_mano_sequence.py` 和
`export_multiview_wilor_training_dataset.py`。完整步骤见
[`nokov_ego_hand_validation` 的 Orbbec/NOKOV 操作说明](../docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)。
至少两段独立录制合并后，运行 `prepare_glove_dataset.py --root datasets/ego_nokov
--split-by-source-group`，使验证集与训练集按录制场次隔离。

以下 Stage 1 是可选的 OpenTouch 域适配流程；使用原始 ViT 时无需运行它。下文涉及 Stage 1 检查点初始化的示例仅适用于选择了域适配的训练。

训练入口为 `train_glove.py`，主要支持三个 recipe：

1. `domain_adapt`：OpenTouch 裸手/手套视觉域适配，可选择 ViT LoRA 或全量 ViT；
2. `refiner_finetune`：使用相机对齐的自有数据，训练 coarse heads 与 RefineNet。
3. `joint_lora_finetune`：使用相机对齐数据，同时训练 ViT LoRA、coarse heads 与 RefineNet。

另外保留 `physical_camera_finetune`、`marker_camera_finetune` 和
`marker_offset_finetune`，用于物理相机头及 marker 偏移的独立微调，详见
[相机与 marker 训练说明](docs/training/CAMERA_MARKER_TRAINING_zh.md)。
`recipe.camera_model` 支持 `virtual`、`physical_perspective` 和 `physical_kb`；
原始 KB 鱼眼图像必须使用 `physical_kb`，不能套用矫正图的透视投影。

所有 YAML 的相对路径均以仓库根目录解析，与启动时的当前目录无关。
通用模板与历史实验配置的用途见 [配置说明](configs/glove/README.md)。

Dataset 只决定“样本有什么监督”，Recipe 决定“模型走到哪里、训练哪些参数”，
`GenericTrainer` 只处理 AMP、反向、validation、checkpoint 和 resume。

## Stage 1：OpenTouch 域适配

### 已解码数据

HDF5 已解码到：

```text
datasets/opentouch_decoded/
├── frames.jsonl
├── clips.jsonl
├── images/<source>/<clip>/<frame>.jpg
└── clips/<source>/<clip>.npz
```

每帧和 bbox 标注只能用以下复合主键合并：

```text
(source_file, clip_id, source_frame_index)
```

`sync_index` 会在不同录制段重复，OpenTouch 流程完全不使用它做关联。
`sample_id` 必须等于
`<source_stem>::<clip_id>::<source_frame_index:06d>`，它是复合主键的校验和，
不是替代主键。

### bbox 标注协议

已生成待标注文件：

```text
datasets/opentouch_bbox_annotations.jsonl
```

共 327,030 行，当前 `bbox_xyxy` 全部为 `null`。每行结构为：

```json
{"sample_id":"eat_mcdonalds::demo_00::000000","source_file":"eat_mcdonalds.hdf5","clip_id":"demo_00","source_frame_index":0,"bbox_xyxy":null}
```

标注规则只有两类：

- 完整右手在图内且可用：填写 `[x1, y1, x2, y2]`；
- 手不完整、严重遮挡或不可用：保持 `null`。

bbox 只用于告诉 WiLoR 从哪里取手部 patch，不能判断手形。准备脚本会拒绝越过图像边界的框，
但“完整手在图内”仍必须由标注者判断。训练裁剪会沿用 WiLoR 的 2 倍扩框和方形 padding，
不会把一个本来完整的手强行截断。

如需重新创建全空待标注清单：

```bash
conda run -n wilor python prepare_glove_dataset.py \
  --data-type opentouch \
  --decoded-root datasets/opentouch_decoded \
  --bbox-annotations datasets/opentouch_bbox_annotations.jsonl \
  --init-empty-bboxes
```

已有文件不会被覆盖；只有明确加 `--force` 才会清空已有框。

### 生成训练 manifest

补完 bbox 后运行：

```bash
conda run -n wilor python prepare_glove_dataset.py \
  --data-type opentouch \
  --decoded-root datasets/opentouch_decoded \
  --bbox-annotations datasets/opentouch_bbox_annotations.jsonl \
  --output-root datasets/opentouch_training \
  --sample-fps 10 \
  --source-fps 30 \
  --val-ratio 0.1 \
  --split-seed 42
```

脚本要求 bbox manifest 精确覆盖全部解码帧；漏行、未知帧、重复复合主键、错误
`sample_id` 都会直接报错。有框样本按约 10 FPS 采样，train/val 按整个 clip 隔离，
避免相邻视频帧泄漏。所有 bbox 都为 `null` 时会明确停止，不会产出假训练数据。

### Stage 1 几何和 loss

OpenTouch 的 Rokoko 21 点不假设位于 RGB 相机坐标系，因此不使用 `K`、`trans`、
2D 投影、相机 head、shape head 或 RefineNet loss。

预测和 GT 分别转入各自的掌心局部坐标：

```text
原点 = wrist (0)
x轴  = pinky MCP (17) → index MCP (5)
y轴  = wrist → middle MCP (9)，并对 x 做正交化
z轴  = x × y
尺度 = |index MCP - pinky MCP|
```

Stage 1 前向为：

```text
RGB crop → ViT + frozen coarse pose head → predicted hand_pose
         → global_orient 强制为单位旋转
         → betas 固定为 MANO mean shape
         → frozen MANO → 21 joints → 掌心局部归一化
         → Smooth-L1(local_keypoints_3d)
```

这样 loss 对传感器到相机的整体旋转、平移和全局尺度不敏感，但仍监督手指相对姿态。
`recipe.vit_tuning` 决定 Stage 1 的训练范围：

- `lora`：所有 32 个 ViT block 的 `attn.qkv`、`attn.proj`、`mlp.fc1`、
  `mlp.fc2` 注入 LoRA；原 ViT 权重冻结；
- `full`：训练 ViT encoder 的原始权重，不注入 LoRA。

两种模式都冻结 pose/camera/shape heads、MANO 和 RefineNet。

启动：

```bash
conda run -n wilor python train_glove.py \
  --config configs/glove/domain_adapt.yaml
```

全量 ViT 使用完整 OpenTouch manifest，物理 batch 4、梯度累积 32，
有效 batch 为 128：

```bash
conda run -n wilor python train_glove.py \
  --config configs/glove/domain_adapt_opentouch_original_full_vit.yaml
```

## Stage 2：Refiner 微调

Stage 2 使用已有 `GloveDataset` 相机对齐协议：bbox、`joints_3d`、`trans`、`K`、
side，以及可选 MANO 参数。完整执行 coarse MANO、相机投影、RefineNet 和最终 MANO。

默认训练：

```text
backbone.decpose
backbone.decshape
backbone.deccam
refine_net
```

从 Stage 1 初始化时，在 `configs/glove/refiner_finetune.yaml` 中设置：

```yaml
paths:
  init_weights: output/opentouch_original_domain_adapt_full_vit/best_trainable.pt
  resume: null
```

然后启动同一个入口：

```bash
conda run -n wilor python train_glove.py \
  --config configs/glove/refiner_finetune.yaml
```

Stage 1 的 LoRA 或全量 ViT 权重会随 checkpoint 加载，在 Stage 2 保持冻结；coarse
heads 和 RefineNet 继续学习自有数据的相机几何。Stage 2 可联合使用 2D、
camera-axis/root-relative 3D 和 MANO 参数 loss。Stage 2 输出会继续携带冻结的 Stage 1
权重，因此可以独立用于推理。

## Checkpoint 语义

schema-v3 输出：

```text
best_trainable.pt
last_trainable.pt
resolved_config.yaml
train_report.json
loss_curves.png
val_visualization/        # 仅可选 Stage 2
```

- `paths.resume`：严格续训同一 recipe、数据 manifest、loss 和优化器，恢复 step/optimizer；
- `paths.init_weights`：只加载匹配模型的权重，从 step 0 开始新阶段。

checkpoint 格式为 `wilor_glove_trainable_v2`，保存所有当前可训练参数以及需要跨阶段
携带的 LoRA/全量 ViT 权重。普通推理仍可用 `load_wilor(..., adapter_path=...)`
加载该格式。

旧 `stage1_keypoints.yaml` / `stage2_mano.yaml` 与 schema-v2 engine 仍保留兼容，
但它们是旧的“冻结 ViT、训练 heads/RefineNet”相机数据流程，不代表新的 OpenTouch Stage 1。

## 验证

CPU 测试：

```bash
conda run -n wilor python -m pytest -q -m 'not gpu'
```

真实模型 GPU 测试：

```bash
WILOR_RUN_GPU_TESTS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n wilor python -m pytest -q tests/test_glove_gpu.py -s
```
