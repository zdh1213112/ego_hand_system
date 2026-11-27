# 手部模型训练

在 `ego_hand_system` 根目录用 `bash training/train.sh` 选择 WiLoR、FAH-S 或 GEM-X。三种后端共用原图和手部标签，各自负责数据预处理、参数更新、验证和模型保存。NOKOV 标定和标签验收见[标签导出说明](../label_export/README.md)，视频生成见[推理说明](../inference/README.md)。

## WiLoR 训练

WiLoR 默认使用 V5 局部针孔训练。以下命令均从 `ego_hand_system` 仓库根目录执行。

- 将数据集放在 `training/datasets/glove_trainset_20261123_v4/`，或链接已有目录。保留签收的 `train.jsonl`、`val.jsonl`、`dataset_profile.json` 及对应图片和 NPY。九段数据有 101,446 份训练样本、11,933 份验证样本。
- 准备 `models/wilor/coarse_start.ckpt` 和 `models/mano/`。训练默认使用 Conda 环境 `wilor_training`，依赖见 `training/requirements.txt`。也可用 `EGO_TRAIN_PYTHON=/path/to/python` 指定解释器。[模型准备](../models/README.md)

### V5 · 缓存局部针孔图（默认）

V5 默认读取原生 NPY 标签的手框，按 KB4 标定为每只手导出 256 × 256 紧视场针孔图和局部相机标签。左手在训练加载时镜像一次。模型从 `models/wilor/coarse_start.ckpt` 初始化，使用 `physical_perspective`、联合 LoRA 和恒定学习率，以 batch 64 训练 15,000 步，每 1,000 步验证。训练输出位于 `training/output/glove_v5_local_pinhole/`。[V5 配方](configs/glove/nokov_v5_bbox_pinhole.yaml)

```bash
bash training/train.sh wilor
```

- 首次运行从 `local_export.native_dataset` 读取原始 train/val 清单和标注，完成全部导出和几何校验后发布到 `paths.dataset_root`。再次运行同一命令会核对原始清单、图片和标签的文件大小与修改时间、框来源、裁剪设置和导出完整性，然后复用图块。更换输入或发现缓存不完整时，请在 YAML 中另选缓存目录。修改学习率或训练步数可以继续使用相同图块。
- 可在同一命令后追加 `--detector /path/to/weights.pt` 或 `--detector-cache /path/to/proposals.jsonl`，也可在 YAML 的 `local_export` 中设置对应路径。检测框与标签框按 IoU 一对一匹配，默认阈值 0.5；使用检测器时默认置信度 0.3。导出器只保留匹配的手，并写出每个划分的匹配统计。
- 仅准备或校验缓存时追加 `--prepare-only`。如果已经独立准备好兼容的局部针孔数据，可复制该配置、移除 `local_export` 段，将 `paths.dataset_root` 指向现有导出。逐框独立评估时，将 `evaluate_glove.py --root` 指向局部数据目录，重投影误差以局部图块像素计。

### V4 路线 A · 原图手框

按标签框直接裁剪原图，使用原始 KB 相机投影。从 `coarse_start.ckpt` 初始化，训练 30,000 步，每 700 步验证。最佳权重保存在 `training/output/glove_v4_joint_lora/best_trainable.pt`。[原图训练配置](configs/glove/nokov_v4_joint_lora.yaml)

```bash
bash training/train.sh wilor --config training/configs/glove/nokov_v4_joint_lora.yaml
```

### V4 路线 B · 原图手框去畸变

读取同一个原图标签框，用该相机的 KB4 标定将框内区域采样成 256 × 256 透视图，并同步变换关节和手腕标签。需要额外准备原图版 adapter `models/wilor/v4_step48300.pt`。从这份权重初始化，用新优化器训练 15,000 步。最佳权重保存在 `training/output/glove_v4_bbox_pinhole_15000/best_trainable.pt`。[去畸变训练配置](configs/glove/nokov_v4_bbox_pinhole.yaml)

```bash
bash training/train.sh wilor --config training/configs/glove/nokov_v4_bbox_pinhole.yaml
```

### 训练输出

v4 的两条路线都可将 `best_trainable.pt` 直接传给推理的 `--adapter`。续训时将 `last_trainable.pt` 填入 `paths.resume`，并把 `training.max_steps` 设为目标总步数。`best_full.ckpt` 和 `last_full.ckpt` 含完整模型，可用于 `training/evaluate_glove.py`。

自定义实验可复制对应 YAML、修改 `paths.output`，用 `--config` 指向新文件。命令行的配置路径相对当前目录，YAML 内的相对资源路径从 `training/` 解析。

## FAH-S 和 GEM-X

两种后端通过同一个训练入口接收自己的算法环境、数据目录和输出目录。环境安装、基础权重和附加资产见 [FAH-S / GEM-X 使用说明](docs/training/FAH_GEM_X_zh.md)。

```bash
EGO_TRAIN_PYTHON=/path/to/fah/bin/python bash training/train.sh fah \
  --dataset /path/to/dataset --output /path/to/runs/fah --epochs 1
EGO_TRAIN_PYTHON=/path/to/gem-x/bin/python bash training/train.sh gem-x \
  --dataset /path/to/dataset --output /path/to/runs/gem-x --epochs 1
```

FAH-S 导出 `model.pt`，GEM-X 导出 `egohand.ckpt`。可通过各后端的 `--help` 查看评估和初始化选项。

## 开发和验证

迁入来源为本地 `wilor_training/main` 的 `248d81a56bd742104cb9a1dea91686a4ea088f13`。模型实现位于 `wilor/`，FAH/GEM-X 适配器位于 `tools/hand_finetune/`。[数据和 checkpoint 协议](GLOVE_TRAINING.md) · [其他训练配方](configs/glove/README.md) · [历史实验](docs/README.md) · [上游说明](UPSTREAM_README.md) · [许可](license.txt)

在训练环境中测试本模块：

```bash
cd training
python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m 'not gpu'
```
