# 训练配置

配置内相对路径以本模块的 `training/` 目录解析。仓库训练入口 `bash training/train.sh wilor` 默认选用九段 v4 原图手框配方。[两条训练路线](../../README.md#wilor-训练) · [训练参数及数据协议](../../GLOVE_TRAINING.md)
数据、预训练模型、MANO 授权文件和已有适配器不随 Git 分发。

## 通用入口

| 配置 | 用途 |
| --- | --- |
| `nokov_v4_joint_lora.yaml` | 九段原图标签、physical KB 和联合 LoRA，batch size 64，30,000 步 |
| `nokov_v4_bbox_pinhole.yaml` | 原图手框按 KB4 去畸变，从 v4 第 48,300 步 adapter 初始化，batch size 64，新增 15,000 步 |
| `refiner_finetune.yaml` | 冻结 ViT，训练 coarse heads 和 RefineNet 的基线 |
| `joint_lora_finetune.yaml` | 联合训练 ViT LoRA、coarse heads 和 RefineNet |
| `domain_adapt.yaml` | OpenTouch 掌心局部 3D 监督，训练 ViT LoRA |
| `domain_adapt_opentouch_*.yaml` | 不同 OpenTouch 数据及全量 ViT 实验 |
| `lora_stage1.yaml` / `lora_stage2.yaml` | `--mode lora_stage1`、`lora_stage2`、`two_stage` 的默认配置 |

运行通用模板前修改 `paths.dataset_root`，必要时设置 `paths.init_weights`。
Stage 2 的 LoRA 两阶段入口需要 Stage 1 权重或续训检查点，基线 `refiner_finetune` 可从原始 WiLoR 权重直接开始。

## 指定录制的实验配置

`trainset_v1_joint_lora.yaml` 对应历史汇总的 2026-11-14 原图 KB 标签，
使用 `datasets/trainset_v1`、`physical_kb`、4000 步训练；
初始化仍使用 `output/native_kb_training_20261111/coarse_start.ckpt`，
重跑输出到新的 `output/trainset_v1_joint_lora/`，与已有训练结果分开。
完整准备条件和命令见 [最新标签训练说明](../../docs/training/START_TRAINSET_V1_zh.md)。

`ego_nokov_20261104_*.yaml` 保留 2026-11-04 录制的完整复现参数：基线、联合 LoRA、
后训练、物理相机、marker camera / offset，以及短程 smoke 配置。
数据路径位于工作区 `../../work/`，输出位于 `output/` 或该工作目录；路径已改为相对写法，实际指向保持一致。
不要把 `smoke` 配置当作完整训练：它们用于少步集成检查。

`python train_glove.py --mode joint_lora` 为兼容本机已有命令，仍默认选择
`ego_nokov_20261104_joint_lora.yaml`。在新数据上建议显式指定通用模板：

```bash
python train_glove.py --mode joint_lora --config configs/glove/joint_lora_finetune.yaml
```

## 历史兼容

`stage1_keypoints.yaml`、`stage2_mano.yaml` 为 schema-v2，仍通过
`python train_glove.py --config ...` 支持。其 checkpoint、续训和 loss 配置语义保留，
新训练优先使用 schema-v3 配置。
