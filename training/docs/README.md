# 文档索引

历史训练命令从本模块目录 `training/` 运行。当前仓库入口及 v4 配方见[训练首页](../README.md)。训练数据、权重和实验产物均需单独准备；历史记录中的本机路径和指标描述当时的运行环境。

## 训练说明

- [历史汇总标签 trainset_v1：训练命令与已有结果](training/START_TRAINSET_V1_zh.md)
- [训练协议与 checkpoint 约定](../GLOVE_TRAINING.md)
- [训练结构与模块职责](../PROJECT_STRUCTURE.md)
- [冻结 ViT 的 NOKOV 基线](training/START_NOKOV_TRAINING_zh.md)
- [NOKOV 联合 LoRA](training/START_NOKOV_JOINT_LORA_zh.md)
- [物理相机与 marker 微调](training/CAMERA_MARKER_TRAINING_zh.md)
- [配置分类及路径约定](../configs/glove/README.md)

## 实验记录

- [2026-11-08：LoRA 与 detector0 评估](experiments/RESULT_NOKOV_LORA_DETECTOR0_20261108_zh.md)
- [2026-11-11：双手原图 KB 与矫正图对照](experiments/RESULT_NATIVE_DUAL_20261111_zh.md)
- [早期训练冒烟测试记录](../TRAINING_SMOKE_TEST_PLAN.md)

## 标签与标定工具

标定工具随本仓库提供。[标定工具索引](../../tools/label_pipeline/calibration/README.md) · [四目原图标签导出](../../docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)。原始 NOKOV 轨迹处理使用工作区中的 `nokov_ego_hand_validation`。

## 测试

激活已安装运行依赖的环境后执行：

```bash
python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m 'not gpu'
```

`tests/` 中的 CPU 回归测试使用临时合成数据，不需要本机数据、外部仓库或模型权重。
覆盖数据裁剪与左右手转换、KB / 透视投影、配置、训练冻结策略、loss、checkpoint 以及已有可视化入口。
设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 可防止系统 ROS 等无关 pytest 插件污染测试环境。

真实 GPU 集成测试使用原始 WiLoR 权重和 MANO 模型：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WILOR_RUN_GPU_TESTS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m pytest -q tests/test_glove_gpu.py
```

其中 `test_unified_domain_engine_one_step` 使用临时合成 OpenTouch 样本，验证真实模型的一步训练、验证及 checkpoint 保存；
另外两项需要测试文件指定的真实 glove / OpenTouch 数据，缺失时会跳过。

`../tools/label_pipeline/calibration/` 中三个 `test_*.py` 是独立运行的工具集成检查，不在默认测试发现范围内。
它们依赖 MANO 及对应日期的本地实验产物，不能作为纯代码检出的验收条件。
准备好各工具 README 中的输入后，可分别显式运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q ../tools/label_pipeline/calibration/manual_shape_tuner/test_engine.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q ../tools/label_pipeline/calibration/joint_pose_attachment/test_joint.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q ../tools/label_pipeline/calibration/personal_hand_calibration/test_explicit_pipeline.py
```
