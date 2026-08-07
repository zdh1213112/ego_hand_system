# EGO Hand System

EGO 双目手部重建与 MANO 拟合系统。输入是 Orbbec Ego 的左右视频、硬件时间戳、双目标定和 IMU；输出是左右手的米制 3D 关节、MANO 网格、21 自由度角度和可选的世界坐标轨迹。

主流程：会话检查 -> 双目 MediaPipe -> 3D 稳定化 -> 两阶段 MANO 拟合 -> 原始鱼眼画面叠加。

## 快速开始：运行 20260806_110653

本录制位于：

```text
recordings/Orbbec_Ego_AZER764008C_20260806_110653
```

已检查的输入状态：1600x1300 双路视频、约 24.8 秒、723 对可配对双目帧、122.267 mm 基线、同步误差 P95 为 75 us。已验证的输出统一写到：

```text
output/recording_20260806_110653_reacquired
```

在项目根目录执行以下命令。首次使用前请确认 `models/hand_landmarker.task`、`models/mano/MANO_LEFT.pkl`、`models/mano/MANO_RIGHT.pkl` 和 `third_party/MANO` 已就绪。

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand
export PYTHONNOUSERSITE=1
unset PYTHONPATH

EGO_SESSION=recordings/Orbbec_Ego_AZER764008C_20260806_110653
EGO_OUTPUT=output/recording_20260806_110653_reacquired
```

### 1. 会话检查

```bash
./build/ego_session_inspect \
  --session "$EGO_SESSION" \
  --output "$EGO_OUTPUT/session_check"
```

检查输出：`stereo_raw.jpg`、`stereo_rectified.jpg`。如果配对帧数、基线或同步误差明显异常，先停止后续流程并检查录制和标定文件。

### 2. 双目 MediaPipe 与三角化

```bash
python scripts/mediapipe_stereo_triangulate.py \
  --session "$EGO_SESSION" \
  --model models/hand_landmarker.task \
  --output "$EGO_OUTPUT/mediapipe_stereo" \
  --track-max-missed 75 \
  --track-max-distance-px 280 \
  --track-reacquire-distance-px 700
```

输出：

- `stereo_annotated.mp4`：左右手关联和 21 点三角化诊断；
- `stereo_frames.csv`：每帧检测、匹配和有效 3D 点数；
- `stereo_landmarks_3d.csv`：后续处理的双目 3D 输入；
- `summary.json`：匹配率、重投影误差和深度统计。

该阶段在左右视角某一侧短暂漏检时，会优先使用真实 MediaPipe 结果，并以低置信度的跨帧 LK 候选补偿缺失视角；轨迹管理器会保留手身份最多 75 个配对帧，并在长遮挡后重获同一轨迹，避免同一只手被拆成多个 `track_id`。

### 3. 3D 稳定化与输入校验

```bash
python scripts/stabilize_hand_3d.py \
  --input "$EGO_OUTPUT/mediapipe_stereo/stereo_landmarks_3d.csv" \
  --output "$EGO_OUTPUT/mano_preparation" \
  --pixel-outlier-window 4 \
  --pixel-outlier-distance 0.45 \
  --pixel-scale-ratio 1.8

python scripts/check_mano_assets.py \
  --input "$EGO_OUTPUT/mano_preparation/mano_input.npz" \
  --input-only

python scripts/check_mano_assets.py \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --input "$EGO_OUTPUT/mano_preparation/mano_input.npz"
```

`mano_input.npz` 会保留真实观测、插值、离群拒绝和左右 2D 有效掩码。MANO 拟合只会把真实且支持度足够的 3D 点作为观测证据。

### 4. 第一阶段 MANO 拟合：形状与稳定初值

GPU 可用时使用 `--device cuda`；若 CUDA 不可用，改为 `--device cpu`。

```bash
python scripts/fit_mano_sequence.py \
  --input "$EGO_OUTPUT/mano_preparation/mano_input.npz" \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --output "$EGO_OUTPUT/mano_fit_optimized_initial_rigid" \
  --shape-iterations 300 \
  --pose-iterations 140 \
  --pose-window 32 \
  --pose-overlap 12 \
  --learning-rate 0.006 \
  --w-3d 1.0 \
  --w-2d 0.35 \
  --w-pinch 0.35 \
  --pinch-threshold-m 0.025 \
  --w-contact-tips 0.75 \
  --contact-tip-threshold-m 0.035 \
  --min-fit-observed-points 12 \
  --max-unobserved-gap 5 \
  --w-pose 0.0025 \
  --w-temporal 0.05 \
  --w-rigid-temporal 0.02 \
  --w-acceleration 0.015 \
  --boundary-weight 0.15 \
  --max-orient-step-deg 40 \
  --max-translation-step-m 0.04 \
  --max-pose-step 2.5 \
  --rigid-initialization \
  --no-image-rigid-alignment \
  --device cuda \
  --no-video
```

### 5. 第二阶段 MANO 拟合：低学习率精修

```bash
python scripts/fit_mano_sequence.py \
  --input "$EGO_OUTPUT/mano_preparation/mano_input.npz" \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --output "$EGO_OUTPUT/mano_fit_optimized_final" \
  --initial-output "$EGO_OUTPUT/mano_fit_optimized_initial_rigid" \
  --shape-iterations 0 \
  --pose-iterations 100 \
  --pose-window 48 \
  --pose-overlap 16 \
  --learning-rate 0.001 \
  --w-3d 1.0 \
  --w-2d 0.30 \
  --w-pinch 0.35 \
  --pinch-threshold-m 0.025 \
  --w-contact-tips 0.75 \
  --contact-tip-threshold-m 0.035 \
  --min-fit-observed-points 12 \
  --max-unobserved-gap 5 \
  --w-pose 0.0025 \
  --w-temporal 0.08 \
  --w-rigid-temporal 0.03 \
  --w-acceleration 0.025 \
  --boundary-weight 0.20 \
  --max-orient-step-deg 35 \
  --max-translation-step-m 0.035 \
  --max-pose-step 1.8 \
  --no-image-rigid-alignment \
  --device cuda \
  --no-video
```

近接触指尖锚定只在拇指尖和食指尖的观测距离小于 35 mm 时启用；它同时约束两根指尖的 3D 位置和左右图 2D 投影，避免只有指间距离正确、但整组指尖仍偏离人手的情况。

### 6. 渲染最终结果

```bash
python scripts/render_mano_overlay_angles.py \
  --session "$EGO_SESSION" \
  --mano-fit "$EGO_OUTPUT/mano_fit_optimized_final" \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --stereo-frames "$EGO_OUTPUT/mediapipe_stereo/stereo_frames.csv" \
  --output "$EGO_OUTPUT/mano_overlay_optimized"

xdg-open "$EGO_OUTPUT/mano_overlay_optimized/mano_overlay_21dof.mp4"
```

最终常用文件：

```text
output/recording_20260806_110653_reacquired/
  session_check/stereo_rectified.jpg
  mediapipe_stereo/stereo_annotated.mp4
  mediapipe_stereo/summary.json
  mano_preparation/summary.json
  mano_fit_optimized_initial_rigid/summary.json
  mano_fit_optimized_final/summary.json
  mano_overlay_optimized/mano_overlay_21dof.mp4
  mano_overlay_optimized/mano_joint_angles_21dof.csv
  mano_overlay_optimized/hand_end_effector_6d.csv
```

本次录制的验证结果：MediaPipe 匹配 692/723 对（95.7%），有效双目点 26,633；稳定化后保留真实长缺口，不跨缺口插值。优化后的 MANO 中位 3D 关节误差约为左 9.1 mm、右 11.9 mm；渲染只显示有可靠观测的连续段，左/右分别输出 659/618 个可见帧。

### 7. 可选：Basalt 世界坐标轨迹

只有需要世界坐标下的手末端 6D 轨迹时才运行。该步骤依赖 `third_party/basalt_runtime`。

```bash
python scripts/run_basalt_offline.py \
  --session "$EGO_SESSION" \
  --hand-pose-csv "$EGO_OUTPUT/mano_overlay_optimized/hand_end_effector_6d.csv" \
  --output-root "$EGO_OUTPUT/basalt_world"
```

输出：`camera_trajectory_world.csv`、`hand_trajectory_world.csv` 和 `world_hand_trajectory_overlay.mp4`。

## 前置条件

### 安装依赖

```bash
git clone https://github.com/zdh1213112/ego_hand_system.git
cd ego_hand_system
./scripts/setup_third_party.sh

conda env create -f environment.yml
conda activate ego-hand
```

所需本地资产：

```text
models/hand_landmarker.task
models/mano/MANO_LEFT.pkl
models/mano/MANO_RIGHT.pkl
third_party/MANO/
third_party/basalt_runtime/          # 仅 Basalt 步骤需要
```

MANO 模型需要从官方渠道获取并接受其许可；安装方法见 [models/README.md](models/README.md)。私有资产包可用：

```bash
./scripts/install_local_assets.sh --archive /path/ego_hand_assets.tar.gz
```

### 构建检查工具

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

## 运行与排错

- 每次新实验使用新的输出目录，例如 `output/recording_20260806_110653_v2`；不要覆盖已分析的结果。
- 首先查看 `mediapipe_stereo/stereo_annotated.mp4`。若手在这里就跳变，问题在检测、关联或三角化，而不是 MANO。
- 若双目点稳定、网格仍不贴手，查看 `mano_fit_optimized_final/summary.json` 和最终叠加视频，再调整 MANO 参数。
- 录制元数据表明本次使用 USB2.0。它不会使离线处理失败，但快速运动时更容易出现运动模糊和单侧漏检；后续采集建议使用 USB3.x 直连。
- 标定板极线验收只在录像中包含 `DICT_5X5_50` 标定板时运行：

```bash
./build/ego_epipolar_validate \
  --session "$EGO_SESSION" \
  --stride 5 \
  --output-csv "$EGO_OUTPUT/epipolar_per_frame.csv"
```

## 其他入口

- 实时双目 MANO：`scripts/run_ego_live.sh`，详细说明见 [docs/EGO_REALTIME.md](docs/EGO_REALTIME.md)。
- 双目深度、畸变和坐标系说明：[docs/STEREO_DEPTH_AND_DISTORTION_PIPELINE.md](docs/STEREO_DEPTH_AND_DISTORTION_PIPELINE.md)。
- Basalt 数据转换与世界坐标：[docs/BASALT_STEREO_INERTIAL_WORLD_TRAJECTORY.md](docs/BASALT_STEREO_INERTIAL_WORLD_TRAJECTORY.md)。
- MANO 运动稳定策略：[docs/MANO_MOTION_STABILITY_OPTIMIZATION_20260804.md](docs/MANO_MOTION_STABILITY_OPTIMIZATION_20260804.md)。

## 测试与坐标约定

```bash
conda run --no-capture-output -n ego-hand \
  python -m unittest discover -s tests -p 'test_*.py' -v
ctest --test-dir build --output-on-failure
```

- 三维坐标单位是米；
- `cam_0` 是左侧参考相机，`cam_1` 是右侧相机；
- 第一阶段 3D 输出位于左相机光学坐标系；
- KB 鱼眼畸变通过 OpenCV `cv::fisheye` 处理；
- 世界坐标仅在可选 Basalt 阶段产生。
