# WiLoR 手套训练冒烟测试计划与验收记录

> 本文只声明训练链路可运行、坐标变换和监督门控正确，不声明模型已经收敛，
> 不声明当前旧数据的 MANO 参数可用于训练，也不声明镜像后的原生左手几何
> 严格属于 `MANO_RIGHT.pkl` 的模型空间。

## 1. 当前阶段声明

- 数据使用左右手各自的原生 MANO 几何；暂不执行左手到 `MANO_RIGHT.pkl` 的重拟合。
- 左手关键点会在加载时镜像到右手规范，但当前阶段接受
  `mirror(MANO_LEFT geometry)` 与 `MANO_RIGHT` 之间的模型空间近似误差。
- 第一阶段只训练最终/粗分支的 2D、3D 关键点 loss。
- 不使用 vertex loss，不使用 adversarial loss。
- `global_orient`、`hand_pose`、`betas` 保留在稳定接口中，但第一阶段权重为 0。
- `joints_2d[778,2]` 是顶点投影校验字段，不直接作为 21 点关键点 loss。
- 可视化必须通过 `wilor/utils/visualization.py` 复用，同一相机帧的左右手合并输出。

## 2. 数据准备验收

执行：

```bash
conda run --no-capture-output -n wilor \
  python prepare_glove_dataset.py \
  --root datasets/dataset \
  --mano-path mano_data/mano_v1_2 \
  --train-ratio 0.9 \
  --split-seed 42
```

通过条件：

1. 图片和标签一一对应，bbox、side、K、trans 及几何数组有限且形状正确；
2. `vertices + trans + K` 与存储的 778 顶点投影误差不超过 `1e-3 px`；
3. `joints_3d + trans + K` 能产生有效的 21 点 2D 训练目标；
4. MANO 参数分别用对应侧模型重放，结果和禁用原因写入 `dataset_profile.json`；
5. MANO 重放失败只禁用参数监督，不掩盖或跳过几何校验错误；
6. train/val 按样本随机 90/10 划分、无重叠且完整覆盖 index；重复准备会覆盖
   旧 manifest 并删除旧 test，不追加旧成员。

2026-08-20 对现有数据的记录：4117 条几何全部通过；旧数据没有声明
`mano_model_by_side`，4117 条参数均不能重放存储几何，因此三项参数监督均为不可用。

## 3. Loss 与梯度验收

第一阶段配置为 `configs/glove/stage1_keypoints.yaml`。实际权重必须为：

```text
keypoints_2d          0.01
keypoints_3d          0.05
coarse_keypoints_2d   0.005
coarse_keypoints_3d   0.025
global_orient         0
hand_pose             0
betas                 0
```

通过条件：总 loss 有限；四项关节监督的可训练分支均有非零有限梯度；冻结的 ViT
参数不变化；batch 1/4/16 的 loss 总尺度一致；训练报告同时记录 raw loss、
weighted loss、请求权重和实际权重。

## 4. Adapter 与恢复验收

- `best_adapter.pt` 与 `last_adapter.pt` 仅保存粗回归头、RefineNet、优化器和元数据。
- YAML 中的 `paths.resume` 仅允许训练签名完全一致，恢复权重、优化器和全局 step。
- `paths.init_adapter` 只加载权重，适用于更换数据或 loss 后从 step 0 开始新阶段。
- `paths.resume` 和 `paths.init_adapter` 互斥。

2026-08-20 的真实 GPU 测试从 step 0 完成到 step 1，并从 `last_adapter.pt`
恢复到 step 2；两次训练签名一致，可训练参数数为 8,340,768，四项关节 loss
均为有限非零值，三项 MANO loss 均为 0。

## 5. 回归测试命令

```bash
conda run --no-capture-output -n wilor \
  python -m pytest -m "not gpu" -q

WILOR_RUN_GPU_TESTS=1 \
conda run --no-capture-output -n wilor \
  python -m pytest tests/test_glove_gpu.py -q
```

EGO 数据导出侧还需要运行：

```bash
conda run --no-capture-output -n ego-hand \
  python -m unittest discover -s tests \
  -p 'test_multiview_wilor_training_export.py'
```
