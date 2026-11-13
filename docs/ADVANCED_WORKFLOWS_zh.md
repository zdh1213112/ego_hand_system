# Orbbec 双目、四目与进阶工作流

本页保留 Orbbec EGO4 / EGO-4B 的双目重建、四目融合、原始 WiLoR、标签导出、实时和 VIO 入口。所有命令在仓库根目录执行。

四目共享 MANO 主入口、模型获取和真实结果见[项目首页](../README.md)；共享推理产物见[输出说明](outputs.md)。

## 效果展示

项目首页展示双手网格与关键点、导出的共享三维模型。已有双目重建仍可输出骨架、MANO 网格、21-DOF 面板和末端 6D 位姿，使用下述对应入口。

## 离线启动

### 一次性安装环境和公共依赖

```bash
git submodule update --init third_party/MANO third_party/basalt third_party/WiLoR
bash scripts/setup_python_environment.sh
conda activate ego-hand
unset PYTHONPATH
bash scripts/install_mediapipe_model.sh
```

安装脚本固定 Python 3.10 和运行时版本，分别安装 CUDA PyTorch、PyPI 依赖与旧版 chumpy，并检查 OpenCV、WiLoR、MediaPipe、MCAP 和 PyTorch。网络中断后可重新运行，下载缓存默认在 `/tmp/ego-hand-pip-cache`。

MANO v1.2 的反序列化依赖 chumpy；脚本按固定提交执行非隔离构建，并在加载时应用 NumPy 兼容辅助。不要混装多个提供同名 `cv2` 的 OpenCV 发行包；环境脚本会检查并修复这一情况。

WiLoR、检测器与 MANO 的来源和路径见[模型说明](../models/README.md)。准备后检查：

```bash
python scripts/check_third_party.py --require-mano --require-wilor
```

### Orbbec EGO：完整运行

双目入口读取平铺的 Orbbec 导出目录，包含左右 MP4、PTS 和相机标定。四路 `part0001` 录制使用[首页的共享推理入口](../README.md#快速开始)或下述四目入口。

```bash
export EGO_SOURCE=orbbec
export EGO_SESSION=/path/to/orbbec_stereo_session
export EGO_OUTPUT=/path/to/new_output
export EGO_HAND_ROUTE=parallel       # mediapipe | wilor | parallel
export EGO_DEVICE=auto               # auto | cuda | cpu
export EGO_MAX_PAIRS=0               # 0 表示全部
export EGO_NO_VIDEO=0
export EGO_WILOR_CAMERAS=both
export EGO_WILOR_BATCH_SIZE=4
bash scripts/run_offline.sh check
bash scripts/run_offline.sh all
```

`EGO_SOURCE` 默认是 `orbbec`。也可复制 [configs/offline_orbbec.env.example](../configs/offline_orbbec.env.example) 后填写路径。双目校正、MediaPipe/WiLoR 路由、稳定化、MANO 拟合和渲染保持各自的阶段输出。

### EGO + NOKOV 原图标签流程

原始数据、双手同步、原图 KB 投影、MANO 拟合、标签导出和验收见[完整流程](EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)，实际命令见[命令清单](EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)。

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /path/to/paired_recording \
  --optimization-config configs/native_joint_refit.json
```

图像对齐和标定轮廓使用 CPU 并行处理，后续联合重拟合保留原有配置和输出。该流程按文档准备同级 `wilor_training` 工具、NOKOV 数据和授权模型；本次相机入口清理没有迁移这些工具。

## 四目共享 MANO

```bash
bash scripts/run_hand_inference.sh \
  --input /path/to/recording/part0001 \
  --detector /path/to/hand_detector.pt \
  --output output/shared_hand
```

支持 MP4+PTS 或四路 H264 MCAP，自动查找配套标定，每时刻每只手拟合一个共享模型。FP32/FP16、批大小、同步和恢复规则见[共享推理说明](SHARED_HAND_INFERENCE_zh.md)。

## 原始四目 WiLoR

以下入口用于检查每路网络自己的二维框和骨架：

```bash
bash scripts/run_orbbec_multiview_wilor.sh \
  --input /path/to/orbbec_recording_part_or_mcap \
  --output output/orbbec_raw \
  --detector /path/to/hand_detector.pt \
  --device cuda --gpu-profile compatible
```

输出 `fusion_multiview/raw_wilor_2d_4view.mp4`；`--optimized-video` 可生成已有的二维去重/质量筛选对照。此入口保留各相机独立预测，三维共享模型使用上一节入口。参数见 `--help`。

## 四目多视角融合

此入口保留 Orbbec 四目的候选关联、KB 射线三角化、RANSAC、时序处理和诊断渲染。默认四路为 `camera0..camera3`，参考相机为 camera0，首选锚点为 camera0/camera1。

MP4 录制：

```bash
bash scripts/run_multiview_wilor_experiment.sh \
  --session /path/to/ego4_recording \
  --output output/orbbec_fusion \
  --detector /path/to/hand_detector.pt \
  --max-frames 60
```

MCAP 录制使用 `--mcap /path/to/recording.mcap`，程序查找配套 Orbbec YAML；也可直接运行标准化入口指定标定：

```bash
python scripts/normalize_multiview_recording.py \
  --input /path/to/recording.mcap \
  --calibration /path/to/recording_calibration_camera.yaml \
  --output output/orbbec_normalized \
  --max-frames 60
```

主要运行选项：

| 参数 | 用途 |
| --- | --- |
| `--gpu-profile compatible` | 保留逐帧 FP32 推理 |
| `--gpu-profile rtx5090d` | 既有 FP16/批处理路径，可覆盖批大小和工作线程 |
| `--compile-backbone 0` | 使用未编译的同一网络前向 |
| `--glove-marker-assist 1` | 用图像亮点辅助二维候选，需适合该特征的录制 |
| `--sam3-glove-assist 1` | 使用另行准备的 SAM3 分割运行时 |
| `--allow-single-hand-anchors 1` | 保留满足几何条件的单手锚点 |
| `--stop-after-inference` | 完成原始 WiLoR 后停止，便于检查输入 |
| `--final-video` | 输出已有的最终结果视频；渲染器可选择 hybrid/fused |

不同流程的质量门限和产物独立定义。使用真实 KB 标定和对应的原图像素，不把针孔投影参数混入鱼眼原图。所有参数以脚本 `--help` 为准。

### 多录制批量处理

```bash
bash scripts/run_multiview_wilor_batch.sh \
  --input-dir /path/to/orbbec_mcaps \
  --output-root output/orbbec_batch
```

每个 Orbbec MCAP 需要其配套标定，分别写入独立输出。该批处理入口继续转发 GPU、检测、融合和辅助参数。

## 训练标签导出

原图 NOKOV 标签优先使用[命令清单](EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)和[原图导出说明](EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)。

已有 Orbbec 多视角融合结果可继续使用原有 MANO/矫正标签导出入口：

```bash
bash scripts/run_multiview_wilor_label_export.sh \
  --experiment /path/to/orbbec_four_view_experiment \
  --output output/orbbec_training_labels \
  --left-camera camera0 --right-camera camera1
```

需要自有 schema 参考样本时，通过 `--reference-npy /path/to/reference.npy` 指定。导出后保留 MANO 参数重放、图像空间、左右手、投影和数量检查。

多个已有实验的标签可分别保存或合并：

```bash
bash scripts/run_multiview_wilor_label_batch.sh \
  --experiment-root /path/to/orbbec_experiments \
  --output-root output/orbbec_labels \
  --layout merged
```

标签的字段、坐标、单位、双手共享图像与标定来源约定见[交接说明](MANO_GLOVE_LABEL_HANDOFF_zh.md)，查看结果见[标签查看器](WILOR_LABEL_VIEWER_zh.md)。

## 分阶段运行与断点恢复

双目流程支持：

```bash
bash scripts/run_offline.sh prepare
bash scripts/run_offline.sh stereo
bash scripts/run_offline.sh stabilize
bash scripts/run_offline.sh fit
bash scripts/run_offline.sh render
bash scripts/run_offline.sh wilor
```

已有完整阶段按标记文件复用；中断留下的半成品应按对应脚本提示处理。共享推理、四目融合、标签流程分别维护自己的配置和恢复状态，参数变化时使用新输出目录。

## 常用运行变量

| 变量 | 用途 |
| --- | --- |
| `EGO_SESSION` / `EGO_OUTPUT` | 双目输入与独立输出 |
| `EGO_HAND_ROUTE` | mediapipe、wilor 或 parallel |
| `EGO_MAX_PAIRS` | 后续处理上限，0 表示全部 |
| `EGO_NORMALIZED_DATASET` / `EGO_RECTIFIED_DATASET` | 复用或指定标准化/校正目录 |
| `EGO_DEVICE` / `EGO_PYTHON` / `EGO_CONDA_ENV` | 设备及解释器选择 |
| `EGO_WILOR_CHECKPOINT` / `EGO_WILOR_DETECTOR` / `EGO_WILOR_CONFIG` | 原双目 WiLoR 资产 |
| `EGO_WILOR_CAMERAS` / `EGO_WILOR_FRAME_STRIDE` / `EGO_WILOR_BATCH_SIZE` | 推理视图与批处理 |
| `EGO_WILOR_FAST` / `EGO_WILOR_SAVE_VERTICES` | 原有快速模式与网格保存 |

完整默认值见 `bash scripts/run_offline.sh --help`；新共享推理的配置由自己的命令行参数控制。

## 输出目录

双目流程保留 `normalized/`、`rectified/`、`mediapipe_stereo/`、`wilor_stereo/`、MANO 拟合/渲染、CSV 和 JSONL 等阶段产物。四目独立预测和融合保留 `normalized_multiview/`、`wilor_multiview/`、`fusion_multiview/` 及各阶段汇总。

共享 MANO 输出采用[独立输出约定](outputs.md)，包含 `hand_model.pt`、`joints3d.jsonl`、四目视频、标定、时间戳和来源信息。

## 排错顺序

1. 先检查环境、模型、原始四路视频/PTS 和配套标定。
2. 检查原始 WiLoR 框和骨架，确认检测器适用于当前手套和视角。
3. 检查时间同步及内外参、单位、相机 ID，再检查三维与重投影。
4. 对标签检查图像空间、左右手镜像和 MANO 重建，避免把不同来源标定混为同一相机。
5. 实时和 VIO 问题按对应文档检查硬件运行时与时间/坐标接口。

## 坐标与数据约定

图像、关键点和相机模型必须处于相同像素空间；相机外参的方向和米/毫米转换必须明确。原图 KB 投影保留畸变，双目矫正流程使用其配套针孔参数。不同推理或标签流程的时间戳单位按各自 schema 读取。

## 可选功能

### Orbbec 实时跟踪

实时手部、多模态联动和 Orbbec SDK 运行时准备见[实时说明](EGO_REALTIME.md)。相机相关 C++ 工具与 Python 发布/预览入口继续保留。

### Basalt 世界坐标轨迹

Basalt 标定、IMU、轨迹和世界坐标合成见[VIO 说明](BASALT_STEREO_INERTIAL_WORLD_TRAJECTORY.md)。

## 构建与测试

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=scripts conda run -n ego-hand python -m pytest -q tests
```

相机采集与 VIO 的构建需按文档准备相应 SDK/运行时；纯离线共享推理使用项目的 Python 环境和授权模型。
