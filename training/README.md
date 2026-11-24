# wilor_training

基于 WiLoR 的手部姿态估计训练项目。支持冻结ViT的heads/RefineNet基线微调，以及联合训练ViT LoRA、heads和RefineNet。两种方式都可使用相机对齐的NOKOV→MANO标签。

同一套原图标签也可用于 FAH-S 和 GEM-X EgoHand。运行入口为 `train_hand.py`，数据约定、微调命令和全量验证结果见 [FAH 与 GEM-X 微调说明](docs/training/FAH_GEM_X_zh.md)。

本机最新已汇总标签为 `datasets/trainset_v1`（2026-11-14 准备，13874 份手标签）。使用
`python train_glove.py --mode joint_lora --config configs/glove/trainset_v1_joint_lora.yaml`
启动对应训练；数据来源、初始化权重、后台命令和已完成的 4000 步记录见
[最新标签训练说明](docs/training/START_TRAINSET_V1_zh.md)。

省略 `--config` 的 `python train_glove.py --mode joint_lora` 仍使用 11 月 4 日旧数据，
该实验见 [历史联合训练说明](docs/training/START_NOKOV_JOINT_LORA_zh.md)；原流程见 [基线说明](docs/training/START_NOKOV_TRAINING_zh.md)。

EGO 原图双手数据按 [原始数据到标签操作流程](../ego_hand_system/docs/EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md) 处理，训练配置显式选择 `physical_kb`。cam0/cam1 的真实双手验收、原图与矫正图短程对照、可视化和限制见 [2026-11-11 验收记录](docs/experiments/RESULT_NATIVE_DUAL_20261111_zh.md)。新增 [四目空间优化与原图导出](../ego_hand_system/docs/EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)；MANO 手指姿态仍由 NOKOV 三维监督，尚未实现四路图像联合优化姿态。

## 准备

1. 按 [上游说明](UPSTREAM_README.md) 安装依赖。将原始 `wilor_final.ckpt` 放到 `pretrained_models/`，将有使用许可的 MANO 文件放到 `mano_data/mano_v1_2/`。
2. 将数据放在 `datasets/ego_nokov/`，或修改 `configs/glove/refiner_finetune.yaml` 中的 `paths.dataset_root`。数据格式及标签约定见 [训练说明](GLOVE_TRAINING.md)。相机内参 `K`、图像和 2D 投影必须处于相同的畸变处理坐标系。
3. 用 `python train_glove.py --config configs/glove/refiner_finetune.yaml` 启动训练。该配置的 `init_weights: null` 表示直接加载原始 WiLoR 权重，不加载已有域适配检查点。

本仓库不包含数据、训练输出、预训练权重或 MANO 授权文件。原 WiLoR 项目说明和许可分别见 [上游说明](UPSTREAM_README.md) 与 [许可](license.txt)。

微调时，每次验证选出新的最佳模型会写入 `best_full.ckpt`，训练结束时写入 `last_full.ckpt`。两者都包含完整网络权重、LoRA 结构和相机分支配置。评测使用 `python evaluate_glove.py --root DATASET --checkpoint OUTPUT/best_full.ckpt --mano-path MANO_DIR --output EVAL_DIR` 即可。续训继续使用原有的 `last_trainable.pt` 或旧版训练配置生成的 `last_adapter.pt`。

## 文档与辅助工具

- [文档索引](docs/README.md)：训练操作说明、实验记录及外部标签导出流程。
- [配置说明](configs/glove/README.md)：通用模板、历史配置与指定录制的实验配置。
- [辅助工具索引](tools/README.md)：标定、标签检查、手型调整及其外部依赖。

图片、视频及 Gradio 推理入口保持兼容；运行方法见 [上游说明](UPSTREAM_README.md) 和 [项目导览](PROJECT_GUIDE.md)。

## 回归检查

安装运行依赖和 `requirements-dev.txt` 后，在仓库根目录、已激活的训练环境中执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m 'not gpu'
```

该命令不需要数据或权重，并避免加载系统 ROS 等无关的 pytest 插件。真实 GPU 测试及工具集成测试的运行条件见 [测试说明](docs/README.md#测试)。
