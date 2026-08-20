# GEN 六目 WiLoR 自动标注与训练数据导出操作手册

更新日期：2026-08-20

## 1. 目标

本手册说明如何把 GEN 六目头环 `.mcap` 转换为可用于 WiLoR 训练的成对数据：

```text
images/xxxxxx.jpg
labels/xxxxxx.npy
```

最终 `.npy` 与参考文件 `/home/zdh/tool/npy_decoder/000865.npy` 保持相同的：

- 字段及字段顺序；
- Python 对象类型；
- dtype；
- shape；
- `vertices + trans`、`K` 和 `joints_2d` 投影关系。

六目识别、几何融合和左右手身份修复的算法原理见
[`GEN_SIX_CAMERA_WILOR_FUSION.md`](GEN_SIX_CAMERA_WILOR_FUSION.md)。

## 2. 最终处理流程

```text
GEN camera0..camera5 MCAP
        |
        v
六路 H264 解码、微秒级同步、DS 标定读取
        |
        v
六路 detector.pt + WiLoR 双姿态假设
        |
        v
detector.pt 严格锁定 Left / Right 身份
        |
        v
动态锚点 + 原生 DS 射线 + 逐关节多目 RANSAC
        |
        v
GEN base 坐标系左右手 21x3 三维关节
        |
        v
转换到 camera2/camera3 共同针孔校正坐标
        |
        v
左右手分别进行共享 shape、逐帧 pose 的时序 MANO 拟合
        |
        v
778 网格顶点 + 21 关节 + MANO 参数
        |
        v
images/xxxxxx.jpg + labels/xxxxxx.npy
        |
        v
与 000865.npy 对照 + 图片配对 + 左右 MANO 参数重放校验
```

训练图片使用 `camera2/3` 的共同针孔校正视图，但三维标签仍来自六目融合。不能把原始
Double Sphere 鱼眼图片直接配上针孔矩阵 `K`，否则即使数组 shape 相同，投影语义也是
错误的。

## 3. 环境和资产检查

进入项目：

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand
```

需要存在：

```text
models/wilor/wilor_final.ckpt
models/wilor/detector.pt
models/wilor/model_config.yaml
models/mano/MANO_LEFT.pkl
models/mano/MANO_RIGHT.pkl
third_party/MANO/mano/model.py
```

如果 MANO 位于其他目录：

```bash
export GLOVE_MANO_SOURCE=/path/to/MANO/source
export GLOVE_MANO_MODEL_DIR=/path/to/MANO/models
```

参考 NPY 默认位置：

```text
/home/zdh/tool/npy_decoder/000865.npy
```

## 4. 情况一：已经完成六目融合，只导出训练数据

当前已经完成的实验使用：

```text
实验目录：output/gen6_pose_full_v3
严格融合：output/gen6_pose_full_v3/fusion_handedness_strict_full
```

执行：

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand

./scripts/run_multiview_wilor_label_export.sh \
  --experiment /home/zdh/ego_hand_system/output/gen6_pose_full_v3 \
  --fusion /home/zdh/ego_hand_system/output/gen6_pose_full_v3/fusion_handedness_strict_full \
  --output /home/zdh/ego_hand_system/output/gen6_pose_full_v3/wilor_training_labels \
  --conda-env ego-hand \
  --device cuda \
  --max-samples 0
```

`--max-samples 0` 表示导出全部合格样本。

该命令不会重新运行六路 WiLoR。它只执行：

1. 六目 3D 转换为 MANO 拟合输入；
2. 左右手 MANO 时序拟合；
3. camera2/3 图片校正和训练标签导出；
4. 全量格式与投影校验。

## 5. 情况二：从原始 MCAP 开始

### 5.1 先进行 60 帧冒烟测试

RTX 5060 Laptop 使用兼容配置：

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand

./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording.mcap \
  --output /path/to/output/gen6_smoke \
  --conda-env ego-hand \
  --device cuda \
  --gpu-profile compatible \
  --max-frames 60 \
  --batch-size 4
```

查看：

```text
/path/to/output/gen6_smoke/fusion_multiview/summary.json
/path/to/output/gen6_smoke/fusion_multiview/diagnostic_6view.mp4
```

可视化中：

- 黄色是右手；
- 蓝绿色是左手；
- `USED` 表示该相机的关节作为最终 RANSAC 内点；
- `INACTIVE` 表示该视角没有参与；
- `OUTLIER` 表示候选被几何质量检查拒绝；
- `REJECTED` 表示整帧未通过输出门限。

应确认手掌翻转过程中没有左右手交换。

### 5.2 运行完整六目识别

冒烟测试通过后，必须使用新的输出目录，把 `--max-frames` 改为 `0`：

```bash
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording.mcap \
  --output /path/to/output/gen6_full \
  --conda-env ego-hand \
  --device cuda \
  --gpu-profile compatible \
  --max-frames 0 \
  --batch-size 4
```

RTX 5090D 服务器使用：

```bash
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording.mcap \
  --output /path/to/output/gen6_full_5090d \
  --conda-env ego-hand \
  --device cuda \
  --gpu-profile rtx5090d \
  --max-frames 0 \
  --batch-size 16
```
```bash
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /home/p3/data_sda1/ego_hand_system/recordings/20260820/DAS-Ego_20260820112315_none_none_689985_a9952163.mcap \
  --output output/gen6_full_5090d \
  --conda-env ego-hand \
  --device cuda \
  --gpu-profile rtx5090d \
  --max-frames 0 \
  --batch-size 16
```
如果 5090D 显存不足，先把 `--batch-size 16` 改为 `8`。如果当前 PyTorch/驱动组合不能
使用编译优化，可加：

```text
--compile-backbone 0
```

### 5.3 从完整六目结果导出训练数据

新版本六目流程的 `fusion_multiview` 已默认启用 strict handedness，因此可省略
`--fusion`：

```bash
./scripts/run_multiview_wilor_label_export.sh \
  --experiment /path/to/output/gen6_full \
  --output /path/to/output/gen6_full/wilor_training_labels \
  --conda-env ego-hand \
  --device cuda \
  --max-samples 0
```

## 6. 标注导出内部阶段

一键入口：

```text
scripts/run_multiview_wilor_label_export.sh
```

内部依次调用：

| 阶段 | 文件 | 输出 |
|---|---|---|
| 六目结果转 MANO 输入 | `scripts/prepare_multiview_mano_input.py` | `mano_input_multiview.npz` |
| camera2/3 针孔校正定义 | 同上 | `training_rectification.npz` |
| MANO 时序拟合 | `scripts/fit_mano_sequence.py` | `mano_fit_multiview/track_0.npz`、`track_1.npz` |
| 图片和 NPY 导出 | `scripts/export_multiview_wilor_training_dataset.py` | `dataset/images`、`dataset/labels` |
| 最终严格校验 | `scripts/check_wilor_training_dataset.py` | 控制台 JSON 验收报告 |

MANO 拟合使用：

- 六目融合后的 21 个三维关节；
- 六目内点视角数量作为观测置信度；
- 左右手各自共享一组 shape 参数；
- 每帧独立 pose，并执行速度、加速度和窗口边界约束；
- camera2/3 校正平面二维投影约束。

导出时保存 `full_pose_axis_angle`（包含 MANO pose mean）对应的旋转矩阵，不能用
仅包含 PCA 增量的 `hand_pose_axis_angle` 代替。校正旋转绕 wrist 根节点作用：导出器
会将 `R @ wrist - wrist` 从局部顶点/关节移入 `trans`，使参数重放和相机空间投影
同时保持一致。`summary.json` 通过 `mano_model_by_side` 直接声明物理左、右手
分别使用的 PKL，并记录所用 MANO 资产的 SHA-256。

## 7. 输出目录

```text
wilor_training_labels/
├── run_config.json
├── mano_input_multiview.npz
├── mano_input_multiview.json
├── training_rectification.npz
├── mano_fit_multiview/
│   ├── summary.json
│   ├── track_0.npz
│   ├── track_0_joints.csv
│   ├── track_0_parameters.csv
│   ├── track_1.npz
│   ├── track_1_joints.csv
│   └── track_1_parameters.csv
└── dataset/
    ├── images/
    │   ├── 000000.jpg
    │   └── ...
    ├── labels/
    │   ├── 000000.npy
    │   └── ...
    ├── index.jsonl
    ├── rejected.jsonl
    └── summary.json
```

图片和标签通过同名文件对应：

```text
images/000865.jpg
labels/000865.npy
```

一个训练样本对应一只手和一个相机视图。同一帧出现左右手时，会生成不同编号的两个
样本；同一只手通过 camera2 和 camera3 质量门限时，也会分别生成两个样本。

## 8. NPY 字段契约

读取方式：

```python
import numpy as np

sample = np.load("labels/000000.npy", allow_pickle=True).item()
```

字段顺序和类型：

| 字段 | 类型 | shape | 含义 |
|---|---|---|---|
| `bbox` | `numpy.float64` | `(4,)` | 当前手在配对图片中的 xyxy 框 |
| `vertices` | `numpy.float32` | `(778,3)` | 平移前 MANO 网格顶点 |
| `joints_3d` | `numpy.float32` | `(21,3)` | 平移前 MANO 21 点 |
| `joints_2d` | `torch.float32` | `(778,2)` | 778 个网格顶点的二维投影 |
| `side` | `numpy.float32` | scalar | `0` 左手，`1` 右手 |
| `trans` | `numpy.float32` | `(3,)` | 当前相机坐标中的平移 |
| `K` | `numpy.float32` | `(3,3)` | 配对校正图片的针孔内参 |
| `mano.global_orient` | `numpy.float32` | `(1,3,3)` | 全局旋转矩阵 |
| `mano.hand_pose` | `numpy.float32` | `(15,3,3)` | 15 个 MANO 关节旋转矩阵 |
| `mano.betas` | `numpy.float32` | `(10,)` | 手形参数 |

虽然字段名为 `joints_2d`，它实际保存 778 个网格顶点投影，不是 21 个关节点。必须满足：

```text
P_camera = vertices + trans
P_image  = K @ P_camera
joints_2d = P_image.xy / P_image.z
```

## 9. 手动校验

一键脚本结束时已经自动校验。也可以单独重新执行：

```bash
cd /home/zdh/ego_hand_system

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/check_wilor_training_dataset.py \
  output/gen6_pose_full_v3/wilor_training_labels/dataset \
  --reference /home/zdh/tool/npy_decoder/000865.npy \
  --mano-source third_party/MANO \
  --mano-model-dir models/mano
```

校验内容：

1. `images/*.jpg` 和 `labels/*.npy` 数量相等、文件名一一对应；
2. `summary.json`、`index.jsonl` 和磁盘文件数量一致；
3. 字段顺序、Python 类型、dtype、shape 与参考 NPY 一致；
4. bbox 位于配对图片内；
5. 所有网格顶点位于相机前方；
6. 778 个顶点通过 `K` 投影后与 `joints_2d` 一致。
7. 左手参数用 `MANO_LEFT.pkl`、右手参数用 `MANO_RIGHT.pkl` 重建；778 顶点和
   21 关节与标签一致，并校验模型文件 SHA-256 与导出时完全相同。

成功输出示例：

```json
{
  "validated_sample_count": 4117,
  "total_sample_count": 4117,
  "paired_images": 4117,
  "schema": "000865-compatible",
  "maximum_projection_error_px": 0.0,
  "mano_model_by_side": {
    "left": "MANO_LEFT.pkl",
    "right": "MANO_RIGHT.pkl"
  },
  "maximum_mano_vertex_replay_error_m": 1.0e-8,
  "maximum_mano_joint_replay_error_m": 1.0e-8
}
```

## 10. 断点复用和输出目录规则

重新执行完全相同的标注导出命令时会复用：

- `mano_input_multiview.npz`；
- `training_rectification.npz`；
- 已完成的 MANO 拟合；
- 已完成的图片/NPY 数据集。

最后的严格校验会重新运行。

`run_config.json` 保存输入文件、MANO 资产、相机、设备和导出范围指纹。以下情况必须使用
新的输出目录：

- 更换 MCAP；
- 更换六目融合结果；
- 更换 MANO 模型；
- 更换 camera2/3；
- 改变 `--max-samples`；
- 改变关键拟合或质量参数。

如果某阶段只有目录但没有 `summary.json`，脚本会停止，不会自动删除中间结果。保留失败
目录用于诊断，然后使用新的输出目录重新运行。

## 11. 当前正式结果

当前正式数据：

```text
/home/zdh/ego_hand_system/output/gen6_pose_full_v3/wilor_training_labels/dataset
```

统计：

| 指标 | 数值 |
|---|---:|
| 图片 | 4117 |
| NPY | 4117 |
| 左手样本 | 2037 |
| 右手样本 | 2080 |
| camera2 样本 | 2059 |
| camera3 样本 | 2058 |
| 多目融合来源 | 4117（100%） |
| 质量拒绝 | 95 |
| 参考 NPY 全量校验 | 通过 |
| 顶点投影最大误差 | 0.0 px |

被拒绝的 95 个候选没有进入训练集：其中 83 个因为 MANO 重投影误差过大，12 个因为
对应相机的有效内点关节不足。

## 12. 常见问题

### 12.1 `run configuration differs`

当前输出目录已经属于另一套输入或参数。不要覆盖，改用新的输出目录。

### 12.2 `fusion is not strict-handedness clean`

输入融合结果仍包含 detector 身份冲突。不要继续导出训练数据，应先用当前 strict
handedness 融合代码重新生成融合结果。

### 12.3 缺少 MANO 文件

确认：

```text
models/mano/MANO_LEFT.pkl
models/mano/MANO_RIGHT.pkl
```

或者设置 `GLOVE_MANO_MODEL_DIR`。

### 12.4 图片和 NPY 数量不一致

该目录是不完整输出，不能用于训练。保留它用于排错，换一个新的标注输出目录重跑。

### 12.5 为什么只导出 camera2/3 图片

camera2/3 是稳定主视角，并且可以构造共同针孔校正平面，使图片和 NPY 内单个 `K`
严格一致。六路相机仍然全部参与三维融合；导出两个训练视图不等于只使用双目生成标签。

### 12.6 为什么没有给每一帧都生成标签

自动标注以质量优先。相机内点关节不足、MANO 重投影误差过大、网格超出图片或身份不
一致的候选会被拒绝，原因保存在 `dataset/rejected.jsonl`。

## 13. 关键脚本

| 文件 | 作用 |
|---|---|
| `scripts/run_multiview_wilor_experiment.sh` | 从 MCAP 完成六目推理、融合和诊断视频 |
| `scripts/run_multiview_wilor_label_export.sh` | 从六目结果一键导出训练数据 |
| `scripts/prepare_multiview_mano_input.py` | 坐标转换、身份审计、MANO 输入准备 |
| `scripts/fit_mano_sequence.py` | 左右手 MANO 时序拟合 |
| `scripts/export_multiview_wilor_training_dataset.py` | 校正图片和 NPY 成对导出 |
| `scripts/check_wilor_training_dataset.py` | 最终训练数据严格校验 |
