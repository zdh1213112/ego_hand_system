# EGO Hand System

EGO 双目手部重建与 MANO 拟合系统。输入是 Orbbec Ego 的左右视频、硬件时间戳、双目标定和 IMU；输出是左右手的米制 3D 关节、MANO 网格、21 自由度角度和可选的世界坐标轨迹。

- EGO 相机标定 YAML 解析；
- KB/Kannala–Brandt 鱼眼双目校正；
- 左右 MP4 的硬件 PTS 配对；
- 会话完整性、基线和同步质量报告；
- 原始/校正后的双目预览图。
- 使用 DICT_5X5_50 标定板定量统计校正后的垂直极线误差。
- 使用 MediaPipe Hand Landmarker 对校正后的左相机录像输出每只手的 21 点基线结果。
- 左右双路 MediaPipe、跨相机手部关联、时序轨迹和21点双目三角化。
- MANO 前处理：离群管理、短缺口补全、时序滤波、骨长约束和已校验 NPZ 输入契约。
- MANO 网格原始鱼眼画面叠加、双手实时角度仪表板和角度 CSV 导出。
- EGO `2bc5:1201` 真机双目实时流、设备标定读取、在线三角化和深度感知跟踪。
- GEN DAS EGO MCAP/H264 离线标准化、Double Sphere 双目校正和统一针孔数据集。
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

然后运行：

```bash
python scripts/check_third_party.py --require-mano --require-live --require-basalt
```

如果这是你有权使用的私有资产压缩包，也可以运行：

```bash
./scripts/install_local_assets.sh --archive /path/ego_hand_assets.tar.gz
```

## Python 环境

```bash
conda env create -f environment.yml
conda activate ego-hand
unset PYTHONPATH
```

RTX 50 系列实时 MANO 使用 CUDA 12.8 PyTorch。本机已验证组合为
`torch 2.11.0+cu128`；环境还需要 `typeguard==4.4.4`。可在已创建环境中执行：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128
python -m pip install typeguard==4.4.4
```

GEN 离线链路还使用 `mcap`、`mcap-protobuf-support`、`av==14.2.0`、
`protobuf`、`zstandard` 和 `lz4`，均已固定在 `environment.yml`。验证环境及
真实 MCAP H264 解码：

```bash
python scripts/check_gen_environment.py \
  --mcap record_data/20260804/DAS-Ego_example.mcap
```

## GEN DAS EGO 离线处理

首期默认使用头环中间双目 `camera2/camera3`。处理分为可独立检查的三步。

第一步将 MCAP 解码为统一原始数据集。每路 H264 使用持久解码器，并按 MCAP
`log_time` 配对，不把两路相同帧号视为同步：

```bash
python scripts/normalize_recording.py \
  --input record_data/20260804/DAS-Ego_example.mcap \
  --output output/normalized/das_example \
  --left-camera camera2 \
  --right-camera camera3
```

第二步根据标定自动选择 KB 或 Double Sphere 模型，输出标准针孔双目图和
`R1/R2/P1/P2/Q`：

```bash
python scripts/rectify_stereo_dataset.py \
  --input output/normalized/das_example \
  --output output/rectified/das_example \
  --camera-model auto \
  --focal-scale 1.0 \
  --video-codec h264 \
  --crf 18
```

第三步直接复用 MediaPipe、跨相机匹配和双目三角化：

```bash
python scripts/mediapipe_stereo_triangulate.py \
  --rectified-dataset output/rectified/das_example \
  --model models/hand_landmarker.task \
  --output output/mediapipe_stereo_das
```

标准化数据集将原始 H264 无转码封装到每路 `video.mkv`，并保留纳秒/微秒
时间戳和 GEN `T_b_c` 标定；校正数据集默认使用 H264/MKV 保存一一对应的
左右视频，不生成逐帧图片目录。需要像素无损中间结果时可改用
`--video-codec ffv1`，代价是文件明显增大。后续稳定化及 MANO 拟合命令不变。
如需把 MANO 网格画回原始 DS 鱼眼图，使用：

```bash
python scripts/render_mano_overlay_angles.py \
  --normalized-dataset output/normalized/das_example \
  --rectified-dataset output/rectified/das_example \
  --mano-fit output/mano_fit \
  --stereo-frames output/mediapipe_stereo_das/stereo_frames.csv \
  --output output/mano_overlay_das
```

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

依赖：OpenCV 4、yaml-cpp、CMake 和 C++17 编译器。

## 检查录制会话

```bash
./build/ego_session_inspect \
  --session "/path/to/Orbbec_Ego_<serial>_<time>" \
  --output output/session_check
```

程序按 `*_camera_left_pts.csv` 和 `*_camera_right_pts.csv` 中的设备硬件时间戳配对，不能将两个 MP4 的第 N 帧直接视为同一时刻。

输出：

```text
output/session_check/stereo_raw.jpg
output/session_check/stereo_rectified.jpg
```

## EGO 真机实时双目 MANO 跟踪

本地工作包需包含 `third_party/orbbec_sdk`。一键构建并启动：

```bash
cd /home/zdh/ego_hand_system
scripts/run_ego_live.sh
```

启动日志应优先显示 `connection=USB3.x`。如果显示 `USB2.0`，双路
1600×1300 MJPEG通常只能得到约8–12个完整帧对/秒，应更换直连USB3端口、
线缆并避开USB2集线器；这属于输入带宽限制，不是CUDA是否启用的问题。

保存实时标注视频：

```bash
scripts/run_ego_live.sh --record
```

启动脚本默认显示原始左鱼眼画面上的双手 MANO 网格、每手 21-DOF 面板和
独立3D预览。仅查看双目21点诊断画面时使用：

```bash
scripts/run_ego_live.sh --no-mano
```

实时程序从 EGO Flash 读取当前设备的 KB 双目标定，按 SDK `FrameSet` 和设备
时间戳获得左右帧，输出左相机光学坐标系下的米制3D。轨迹关联联合使用左图
速度预测、三维手掌中心、Z深度和 handedness 软约束；关节点根据极线误差、
重投影误差和视差做质量加权滤波。详见
[`docs/EGO_REALTIME.md`](docs/EGO_REALTIME.md)。关于“界面为什么像单目”、
MediaPipe world landmarks 与真实相机深度的区别、鱼眼去畸变/双目校正要求及
完整坐标流程，参阅
[`docs/STEREO_DEPTH_AND_DISTORTION_PIPELINE.md`](docs/STEREO_DEPTH_AND_DISTORTION_PIPELINE.md)。

## 标定板极线误差验证

```bash
./build/ego_epipolar_validate \
  --session "/path/to/Orbbec_Ego_<serial>_<time>" \
  --stride 5 \
  --output-csv output/epipolar_per_frame.csv
```

当前工具针对本项目录制的 `DICT_5X5_50` 标定板，通过左右同 ID 标记的四个角点统计校正后的垂直像素误差。正式验收建议同时保存标定板打印规格，并增加已知三维尺度测试。

## MediaPipe 左相机 21 点基线

使用独立 Conda 环境，不依赖系统 Python：

```bash
conda env create -f environment.yml
conda activate ego-hand
unset PYTHONPATH

python scripts/mediapipe_left_baseline.py \
  --session "/path/to/Orbbec_Ego_<serial>_<time>" \
  --model models/hand_landmarker.task \
  --output output/mediapipe_left
```

程序先使用 EGO YAML 中的 KB 标定参数完成双目校正，再在左校正图上运行最多两只手的 21 点检测。输出包括：

- `left_frames.csv`：逐帧检测数量；
- `left_landmarks.csv`：归一化坐标、校正图像素坐标、左右手类别与模型相对世界坐标；
- `left_annotated.mp4`：检测结果可视化；
- `summary.json`：版本、参数、检测率和速度。

`world_x_m/world_y_m/world_z_m` 是 MediaPipe 的手部模型相对坐标，不是相机坐标，也不能替代后续双目三角化结果。

## 双目21点三角化

```bash
conda activate ego-hand
unset PYTHONPATH

python scripts/mediapipe_stereo_triangulate.py \
  --session "/path/to/Orbbec_Ego_<serial>_<time>" \
  --model models/hand_landmarker.task \
  --output output/mediapipe_stereo
```

程序按左右硬件 PTS 配对，不按两个视频的相同帧号配对。左右校正图分别运行 MediaPipe，随后根据极线误差、视差、handedness软约束和几何有效点数量关联同一只手。输出：

- `stereo_frames.csv`：逐对同步、检测、匹配和3D有效点统计；
- `stereo_landmarks_3d.csv`：左右2D点、视差、极线误差、重投影误差及三维坐标；
- `stereo_annotated.mp4`：左右相机关联结果，相同颜色和 `track_id` 表示同一只手；
- `summary.json`：全片双目匹配率、3D有效率、误差和深度分布。

`x_left_camera_m/y_left_camera_m/z_left_camera_m` 是原始左相机光学坐标系下的米制3D结果，可作为后续 MANO 拟合输入。

## MANO前处理

```bash
python scripts/stabilize_hand_3d.py \
  --input output/mediapipe_stereo/stereo_landmarks_3d.csv \
  --output output/mano_preparation
```

该阶段先利用单帧手部空间范围、局部时序中值、骨架拓扑和优化残差剔除明显的双目深度离群点；再只填补不超过5帧的内部短缺口，保留长时间缺失。最后按观测几何质量进行零相位时序滤波，并使用每只手独立估计的稳定骨长约束结果。输出：

- `stabilized_landmarks_3d.csv`：带 `observed/interpolated/rejected_outlier/missing` 来源状态的稳定3D点；
- `mano_input.npz`：MANO拟合数据接口，包含原始观测、可接受观测、离群剔除、插值和有效性 mask；
- `stabilized_3d.mp4`：原始灰点与稳定骨架的正视图/俯视图；
- `preview_montage.jpg`：全片六个时间点的预览拼图；
- `summary.json`：补点数量、完整手实例、时序抖动和骨长误差对比。

只校验 NPZ 数据契约（不需要 MANO 模型）：

```bash
python scripts/check_mano_assets.py \
  --input output/mano_preparation/mano_input.npz \
  --input-only
```

MANO模型文件需用户从官方渠道接受许可证后提供。放置 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl` 后可检查：

```bash
python scripts/check_mano_assets.py \
  --model-dir models/mano \
  --mano-source third_party/MANO \
  --input output/mano_preparation/mano_input.npz
```

### MANO 参数拟合

本项目使用放置在 `third_party/MANO` 的外部 `otaheri/MANO` PyTorch 源码实现，不修改该外部目录。已固定 MediaPipe 21点到 MANO 21关节的语义映射，拟合目标包含置信度加权3D关节、左右校正图2D重投影、姿态/形状先验和时序平滑。

先用少量帧冒烟测试：

```bash
python scripts/fit_mano_sequence.py \
  --input output/mano_preparation/mano_input.npz \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --output output/mano_fit_smoke \
  --max-pairs 12
```

冒烟测试通过后运行全量：

```bash
python scripts/fit_mano_sequence.py \
  --input output/mano_preparation/mano_input.npz \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --output output/mano_fit
```

输出中每条轨迹包含 MANO 778顶点、21关节、faces、shape/pose/orientation/translation 参数、关节 CSV 和网格对比视频。

针对移动过程中的 MANO 整手翻转、平移和关节突变，项目现已加入观测质量自适应时序约束、窗口边界锚定、SO(3) 旋转限幅、姿态/刚体速度与加速度约束。验证结果和推荐参数见 [MANO 运动稳定性优化](docs/MANO_MOTION_STABILITY_OPTIMIZATION_20260804.md)。正式对比视频位于 `output/mano_overlay_motion_guard/mano_overlay_21dof.mp4`。

本录像的正式精修结果位于 `output/mano_fit_refined`，可直接查看：

```bash
xdg-open output/mano_fit_refined/mano_fit_both_hands.mp4
```

### 21自由度网格叠加与双手3D预览

将精修后的 MANO 778 顶点按 EGO KB 鱼眼内参直接投影回原始左相机画面。右侧同时显示左右手21自由度弧度条和独立 MANO 3D 预览：

```bash
python scripts/render_mano_overlay_angles.py \
  --session "/path/to/Orbbec_Ego_<serial>_<time>" \
  --mano-fit output/mano_fit_refined \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --stereo-frames output/mediapipe_stereo/stereo_frames.csv \
  --output output/mano_overlay_21dof
```

直接查看正式结果：

```bash
xdg-open output/mano_overlay_21dof/mano_overlay_21dof.mp4
```

21自由度定义为：

- 拇指：CMC屈伸、CMC张合、CMC对掌旋转、MCP屈伸、IP屈伸；
- 其余四指：MCP屈伸、MCP张合、PIP屈伸、DIP屈伸。

`mano_joint_angles_21dof.csv` 保存未滤波弧度、5帧局部中值弧度和对应角度；`mano_pose_axis_angle.csv` 保存全部15×3 MANO mean-relative 轴角分量；原有 `mano_joint_angles.csv` 继续保存20个几何角。两类角都尚未经过医学解剖轴标定。

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
ctest --test-dir build --output-on-failure
```

未放置许可 MANO 模型时，对应的官方模型前向测试会自动跳过。

## 当前约定

- 三维坐标单位统一为米；
- YAML 中 EGO 相机平移按毫米读取并转换为米；
- `cam_0` 为左/参考相机，`cam_1` 为右相机；
- `KB` 畸变使用 OpenCV `cv::fisheye` API；
- 第一阶段输出坐标系为左相机光学坐标系。

## Basalt 双目惯性 VIO 与世界坐标手部轨迹

项目现已支持把 EGO 双目、PTS、KB 鱼眼标定和约 1000 Hz IMU 转为 Basalt 输入，估计左相机的米制重力对齐世界轨迹，再将相机坐标下的手部末端 6D 位姿转换为世界坐标：

```text
T_world_hand = T_world_imu · T_imu_camera · T_camera_hand
```

进入环境后可一条命令运行当前录制：

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
