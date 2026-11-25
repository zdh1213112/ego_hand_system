# 手部模型训练

在 `ego_hand_system` 根目录用 `bash training/train.sh` 选择 WiLoR、FAH-S 或 GEM-X。三种后端共用原图和手部标签，各自负责数据预处理、参数更新、验证和模型保存。NOKOV 标定和标签验收见[标签导出说明](../label_export/README.md)，视频生成见[推理说明](../inference/README.md)。

## WiLoR 训练

两条路线共用九段 v4 原图标签、ViT LoRA、coarse heads、RefineNet 和可学习的相机头，batch size 都是 64。从仓库根目录执行以下命令。

- 将数据集放在 `training/datasets/glove_trainset_20261123_v4/`，或链接已有目录。保留签收的 `train.jsonl`、`val.jsonl`、`dataset_profile.json` 及对应图片和 NPY。九段数据有 101,446 份训练样本、11,933 份验证样本。
- 准备 `models/wilor/coarse_start.ckpt` 和 `models/mano/`。训练默认使用 Conda 环境 `wilor_training`，依赖见 `training/requirements.txt`。也可用 `EGO_TRAIN_PYTHON=/path/to/python` 指定解释器。[模型准备](../models/README.md)

### 路线 A · 原图手框（默认）

按标签框直接裁剪原图，使用原始 KB 相机投影。从 `coarse_start.ckpt` 初始化，训练 30,000 步，每 700 步验证。最佳权重保存在 `training/output/glove_v4_joint_lora/best_trainable.pt`。[原图训练配置](configs/glove/nokov_v4_joint_lora.yaml)

```bash
bash training/train.sh wilor
```

### 路线 B · 原图手框去畸变

读取同一个原图标签框，用该相机的 KB4 标定将框内区域采样成 256 × 256 透视图，并同步变换关节和手腕标签。需要额外准备原图版 adapter `models/wilor/v4_step48300.pt`。从这份权重初始化，用新优化器训练 15,000 步。最佳权重保存在 `training/output/glove_v4_bbox_pinhole_15000/best_trainable.pt`。[去畸变训练配置](configs/glove/nokov_v4_bbox_pinhole.yaml)

```bash
bash training/train.sh wilor --config training/configs/glove/nokov_v4_bbox_pinhole.yaml
```

**路线 B 的 v5 配方**在训练时加入手框扰动、颜色微扰、模糊和局部遮挡。同一帧同一只手的多路视角进入同一批次，损失函数额外约束指尖位置、骨段方向、指节反折、腕根粗估位置和多视角手姿一致性。它也从 v4 第 48,300 步初始化，切换相机模式时将相机头末层归零，以 500 步预热和余弦学习率训练 15,000 步。输出目录为 `training/output/glove_v5_bbox_pinhole_15000/`。[v5 配方](configs/glove/nokov_v5_bbox_pinhole.yaml)

```bash
bash training/train.sh wilor --config training/configs/glove/nokov_v5_bbox_pinhole.yaml
```

### 训练输出

两条路线都可将 `best_trainable.pt` 直接传给推理的 `--adapter`。续训时将 `last_trainable.pt` 填入 `paths.resume`，并把 `training.max_steps` 设为目标总步数。`best_full.ckpt` 和 `last_full.ckpt` 含完整模型，可用于 `training/evaluate_glove.py`。

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
