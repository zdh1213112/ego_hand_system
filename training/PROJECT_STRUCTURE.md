# WiLoR 当前训练结构

> 具体数据协议和命令以 `GLOVE_TRAINING.md` 为准；辅助工具见 `tools/README.md`，实验记录见 `docs/README.md`。

## 1. 设计边界

手套训练只有一个命令入口：

```text
train_glove.py
    → load_training_config
    → build_datasets
    → build_recipe
    → load base WiLoR / optional init_weights
    → recipe.configure_model
    → GenericTrainer
```

职责严格分开：

- Dataset：读取图像和该数据集真正具备的监督；
- Recipe：冻结策略、前向路径和优化器参数组；
- Loss：只消费标准 batch/output 字段；
- Engine：DataLoader、AMP、反向、validation、日志、checkpoint。

## 2. 训练方式

### 联合微调：`joint_lora_finetune`

使用相机对齐的 GloveDataset，在完整 coarse → RefineNet 路径上同时训练
ViT LoRA、coarse heads 和 RefineNet。原始 ViT 和 MANO 权重保持冻结；
可独立启动，不要求先运行 OpenTouch Stage 1。

相机模式由 `recipe.camera_model` 显式选择：`virtual` 使用原始 WiLoR
裁剪相机，`physical_perspective` 用于矫正图，`physical_kb` 用于原始鱼眼图。
物理相机及 marker 的独立微调 recipe 也保留在 `recipes/` 中。

### Stage 1：`domain_adapt`

```text
OpenTouch RGB + Rokoko 21×3
    → OpenTouchDataset
    → ViT 32 blocks 的 LoRA 或全量 ViT encoder
    → coarse hand_pose
    → identity global orient + MANO mean shape
    → frozen MANO
    → palm-local normalized joints
    → local_keypoints_3d Smooth-L1
```

不经过 2D 投影和 RefineNet，不使用 camera/global orientation/shape GT。

### Stage 2：`refiner_finetune`

```text
camera-aligned GloveDataset
    → ViT/coarse MANO/camera
    → mesh projection
    → RefineNet
    → final MANO/joints/2D
    → 2D + 3D + MANO losses
```

从 Stage 1 加载的 LoRA 或全量 ViT 权重继续生效但保持冻结；训练 coarse
pose/shape/camera heads 和 RefineNet。

## 3. 关键目录

```text
WiLoR/
├── train_glove.py
├── prepare_glove_dataset.py
├── pre_opentouch_dataset/
│   ├── decode_opentouch.py
│   └── README.md
├── configs/glove/
│   ├── domain_adapt.yaml
│   ├── domain_adapt_opentouch_original_full_vit.yaml
│   ├── refiner_finetune.yaml
│   ├── joint_lora_finetune.yaml
│   ├── lora_stage1.yaml / lora_stage2.yaml
│   ├── stage1_keypoints.yaml       # legacy schema-v2
│   └── stage2_mano.yaml            # legacy schema-v2
├── wilor/
│   ├── datasets/
│   │   ├── opentouch_dataset.py
│   │   ├── glove_dataset.py
│   │   └── glove_validation.py
│   ├── models/
│   │   ├── wilor.py
│   │   ├── lora.py
│   │   ├── physical_camera.py / kb_camera.py
│   │   └── adapter.py
│   ├── training/
│   │   ├── config.py
│   │   ├── engine.py
│   │   ├── losses.py
│   │   ├── recipes/
│   │   │   ├── base.py
│   │   │   ├── domain_adapt.py
│   │   │   ├── refiner_finetune.py / joint_lora_finetune.py
│   │   │   └── physical_camera_finetune.py / marker*_finetune.py
│   │   └── glove.py                # legacy schema-v2 compatibility
│   └── utils/
│       └── hand_geometry.py
├── docs/                        # 操作说明与历史实验记录
├── tools/                       # 可选标定、标签生成和查看工具
└── tests/                       # 不依赖本地录制数据的回归测试，以及显式启用的 GPU 测试
```

## 4. 文件职责

| 文件 | 职责 |
|---|---|
| `prepare_glove_dataset.py` | `glove` / `opentouch` 数据准备统一 CLI |
| `opentouch_dataset.py` | 复合主键合并、bbox 校验、clip-disjoint split、RGB/NPZ 加载 |
| `glove_dataset.py` | Stage 2 相机标签、左右手规范化和 crop |
| `wilor.py` | 共享 `_forward_coarse`、coarse-only API 和完整 forward |
| `lora.py` | LoRA 注入、启用、冻结和参数枚举 |
| `adapter.py` | legacy adapter 与 `wilor_glove_trainable_v2` checkpoint |
| `config.py` | schema-v3 严格配置 |
| `recipes/` | recipe factory、冻结策略、前向和 optimizer |
| `engine.py` | 与数据几何无关的通用训练循环 |
| `losses.py` | camera 2D/3D/MANO loss 与独立 local 3D loss |
| `hand_geometry.py` | wrist/palm 局部坐标和 palm-width 归一化 |

## 5. OpenTouch 主键和标注

OpenTouch 的永久帧主键是：

```text
(source_file, clip_id, source_frame_index)
```

`sync_index` 不参与任何 join。bbox 文件必须覆盖全部解码帧；`bbox_xyxy=null`
代表无用样本，非空 bbox 代表标注者确认完整手在图内。准备结果按 clip 划分，
不会把同一视频片段的相邻帧分到 train 和 val。

## 6. Checkpoint

新格式：

```text
wilor_glove_trainable_v2
├── recipe
├── base_checkpoint
├── model_modifications.lora
├── state_dict
├── optimizer
├── step / val_loss
└── training_metadata.signature
```

`init_weights` 允许跨阶段加载模型中存在且 shape 一致的键，不要求两个阶段的可训练
参数集合相同。之后冻结的 Stage 1 权重仍会被携带进 Stage 2 checkpoint，保证产物可独立
加载。`resume` 要求同一训练签名、严格相同的 trainable state 和 optimizer。

## 7. 常用命令

```bash
# 初始化全空 bbox 工作清单
conda run -n wilor python prepare_glove_dataset.py \
  --data-type opentouch --init-empty-bboxes

# bbox 完成后准备 Stage 1 数据
conda run -n wilor python prepare_glove_dataset.py \
  --data-type opentouch --output-root datasets/opentouch_training

# Stage 1
conda run -n wilor python train_glove.py \
  --config configs/glove/domain_adapt.yaml

# Stage 1（全量 ViT）
conda run -n wilor python train_glove.py \
  --config configs/glove/domain_adapt_opentouch_original_full_vit.yaml

# Stage 2
conda run -n wilor python train_glove.py \
  --config configs/glove/refiner_finetune.yaml

# CPU 测试
conda run -n wilor python -m pytest -q -m 'not gpu'

# 真实模型 GPU 集成测试
WILOR_RUN_GPU_TESTS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n wilor python -m pytest -q tests/test_glove_gpu.py -s
```

## 8. 兼容性

图片、视频和 Gradio 推理接口不变。`load_wilor(..., adapter_path=...)` 同时能加载
legacy `wilor_glove_adapter_v1` 和新 checkpoint。schema-v2 YAML/engine 暂时保留，
用于复现旧相机对齐实验；新的主流程使用 schema-v3 YAML。
