# EGO Hand System

EGO 双目三维手部重建项目。目前提供：

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

> Git 跟踪的仓库只包含源码与测试。本地工作包可以同时放置录像、运行输出和模型，
> 但 `.gitignore` 会阻止它们被误提交。请参阅 [`models/README.md`](models/README.md)。

## 仓库结构

```text
include/      C++ 头文件
src/          C++ 核心实现
tools/        会话检查与极线验证工具
scripts/      MediaPipe、3D 稳定、MANO 拟合与可视化脚本
tests/        C++/Python 回归测试
models/       本地运行模型，模型本体被 Git 忽略
docs/         管线与参考结果说明
output/       默认运行输出目录，内容被 Git 忽略
data/         本地录像数据，被 Git 忽略
third_party/  外部 MANO 源码，被 Git 忽略
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
