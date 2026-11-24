# 手部模型训练

在 `ego_hand_system` 根目录用 `bash training/train.sh` 选择 WiLoR、FAH-S 或 GEM-X。三种后端共用原图和手部标签，各自负责数据预处理、参数更新、验证和模型保存。NOKOV 标定和标签验收见[标签导出说明](../label_export/README.md)，视频生成见[推理说明](../inference/README.md)。

## WiLoR 九段 v4 配方

1. 将已签收的数据集放到 `training/datasets/glove_trainset_20261123_v4/`，也可链接已有目录。目录内应有 `train.jsonl`、`val.jsonl`、`dataset_profile.json` 以及清单指向的图片和 NPY。历史九段数据为 101,446 份训练样本、11,933 份验证样本。换数据集时请保留签收的时间划分。
2. 在 `models/wilor/coarse_start.ckpt` 部署本配方的基础权重，在 `models/mano/` 部署对应的 MANO 模型。训练环境安装 `training/requirements.txt` 中的依赖，可用 `EGO_TRAIN_PYTHON=/path/to/python` 选择解释器，默认使用 Conda 环境 `wilor_training`。
3. 启动训练。默认配方联合训练 ViT LoRA、coarse heads、RefineNet 和 physical KB 相机头，batch size 64，总计 30,000 步，每 700 步验证。模型配置随源码提供。[完整配方](configs/glove/nokov_v4_joint_lora.yaml)

```bash
bash training/train.sh wilor
```

默认输出在 `training/output/glove_v4_joint_lora/`。新实验可复制 YAML，设置独立的 `paths.output`，再指定配置文件。命令行的配置文件路径相对当前目录，YAML 中的相对资源路径以 `training/` 为起点。

```bash
bash training/train.sh wilor --config training/configs/glove/nokov_v4_joint_lora.yaml
```

| 文件 | 用途 |
| --- | --- |
| `best_trainable.pt` | 将最佳微调参数交给 WiLoR 推理，配合同一基础权重和模型配置 |
| `last_trainable.pt` | 续训时填写到 `paths.resume`，`training.max_steps` 设置为目标总步数 |
| `best_full.ckpt`、`last_full.ckpt` | 含完整模型的检查点，可用 `training/evaluate_glove.py` 评估 |

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
