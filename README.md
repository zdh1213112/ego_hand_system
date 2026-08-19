# EGO Hand System

面向头戴式双目相机的离线手部重建与 MANO 拟合系统。当前主要支持两种输入：

- `orbbec`：Orbbec EGO 左右视频、硬件时间戳、KB 鱼眼标定和可选 IMU；
- `gen`：GEN DAS EGO MCAP/H264、KB 或 Double Sphere 双目标定。

两类数据在双目校正后可选择 MediaPipe+MANO、WiLoR，或同时运行两条路线：

```text
  Orbbec ─┐
          ├─> 标准化 ─> 双目校正 ─┬─> MediaPipe ─> MANO / 21-DOF
  GEN ────┘                     └─> WiLoR ─> MANO / 21-keypoints
```

两条路线从双目校正结果并行运行，可通过 `EGO_HAND_ROUTE` 选择。

## 效果展示

![Orbbec EGO 双手 MANO 网格、21-DOF 面板与 3D 预览](docs/images/offline_dual_hand_mano_21dof_overlay.png)

Orbbec EGO 离线处理结果：左侧在原始鱼眼画面上叠加双手 MANO 网格、骨架和
末端 6D 坐标轴；右侧同步显示左右手 21-DOF 数值及独立 MANO 3D 预览。

![WiLoR 左右目检测与 21 点投影结果](docs/images/wilor_stereo_dual_view.png)

同一组校正双目数据的 WiLoR 左右目独立结果：每路显示手框、左右手分类、置信度和
21 点投影骨架；逐帧数值结果同时保存为 JSONL 和 NPZ。

## 离线启动

下面的命令覆盖环境、源码、模型资产和完整运行。只需要把尖括号中的占位符替换成
你自己的路径；没有 WiLoR 模型时，把 `EGO_HAND_ROUTE` 改为 `mediapipe`。

### 一次性安装环境和公共依赖

```bash
cd <ego_hand_system目录>
./scripts/setup_python_environment.sh
conda activate ego-hand
unset PYTHONPATH
git submodule sync --recursive
git submodule update --init third_party/MANO third_party/basalt third_party/WiLoR
./scripts/install_mediapipe_model.sh
```

`environment.yml` 只负责创建稳定的 Python/Conda 基础环境；安装脚本会继续完成完整、
固定版本的运行时安装。它把 CUDA PyTorch、普通 PyPI 包和 chumpy 分开处理，避免普通
包误访问 PyTorch CUDA 下载源，也避免一次下载中断导致整个 Conda 环境创建失败。

MANO v1.2 模型需要旧版 chumpy 来反序列化。chumpy 的 GitHub 源码构建脚本与现代 pip
的隔离构建不兼容，因此安装脚本会明确执行固定提交的非隔离构建：

```bash
python -m pip install --no-build-isolation \
  git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da
```

项目在加载 MANO 时还会补上 chumpy 所需的 NumPy 2 兼容别名。安装完成后，脚本会
自动检查 WiLoR、MediaPipe、GEN、PyTorch/CUDA 和 OpenCV 包是否可用。如果网络中断，
重新执行 `./scripts/setup_python_environment.sh` 即可；已下载内容保存在 `/tmp/ego-hand-pip-cache`。
脚本也会清理 `ultralytics` 可能引入的 `opencv-python`，然后修复安装项目统一使用的
`opencv-contrib-python`，避免两个提供同名 `cv2` 模块的发行包互相覆盖。
当前官方 `detector.pt` 含有 `C3k2` 层，因此安装脚本固定使用与该权重一致的
Ultralytics 8.4.56；环境检查也会在版本或模型层不兼容时提前报出明确错误。

准备授权资产。MANO 的两个 pkl 从官方页面下载后安装，放到models/MANO。WiLoR checkpoint 和 detector
从官方 Hugging Face Space 下载：

```bash
cd <ego_hand_system目录>
mkdir -p models/wilor
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
  -P models/wilor/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
  -P models/wilor/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/model_config.yaml \
  -P models/wilor/
python scripts/check_third_party.py --require-mano --require-wilor
```

### GEN DAS EGO：完整运行

```bash
cd <ego_hand_system目录>
conda activate ego-hand

# 必须填写
export EGO_SOURCE=gen
export EGO_MCAP=<MCAP文件绝对路径>
export EGO_OUTPUT=<新输出目录绝对路径>

# 可选；以下均可不填写，右侧是默认值或示例值
export EGO_LEFT_CAMERA=camera2       # 默认 camera2
export EGO_RIGHT_CAMERA=camera3      # 默认 camera3
export EGO_HAND_ROUTE=parallel       # mediapipe | wilor | parallel
export EGO_DEVICE=auto               # auto | cuda | cpu
export EGO_MAX_PAIRS=0               # 冒烟测试可改为 60
export EGO_MAX_FRAMES=0              # GEN 解码帧数；0 表示全部
export EGO_NO_VIDEO=0                # 1 表示跳过诊断视频
export EGO_WILOR_CAMERAS=both        # left | right | both
export EGO_WILOR_BATCH_SIZE=4
export EGO_WILOR_FRAME_STRIDE=1
./scripts/run_offline.sh check
./scripts/run_offline.sh all
```

### Orbbec EGO：完整运行

```bash
cd <ego_hand_system目录>
conda activate ego-hand

# 必须填写
export EGO_SOURCE=orbbec
export EGO_SESSION=<Orbbec会话目录绝对路径>
export EGO_OUTPUT=<新输出目录绝对路径>

# 可选；以下均可不填写，右侧是默认值或示例值
export EGO_HAND_ROUTE=parallel       # mediapipe | wilor | parallel
export EGO_DEVICE=auto               # auto | cuda | cpu
export EGO_MAX_PAIRS=0
export EGO_NO_VIDEO=0
export EGO_WILOR_CAMERAS=both
export EGO_WILOR_BATCH_SIZE=4
./scripts/run_offline.sh check
./scripts/run_offline.sh all
```

### GEN 六目 WiLoR 实验

完整设计、算法改进、指标、左右手身份修复和排错记录见
[`docs/GEN_SIX_CAMERA_WILOR_FUSION.md`](docs/GEN_SIX_CAMERA_WILOR_FUSION.md)。

主流程仍是 `camera2+camera3` 双目。以下独立实验链路会一次读取
`camera0..camera5`，以 `camera2` 时间戳同步六路原始 DS 鱼眼视频，对每路运行左右手
双假设 WiLoR，以 `camera2/3` 为首选锚点并在缺失时动态选择其他相机对，再让外围
相机逐只手按重投影匹配，最后用原生 Double-Sphere 射线和 RANSAC 融合 3D 关节：

最终 Left/Right 身份由 `detector.pt` 的 `left/right` 类别严格锁定；双假设只用于估计
对应身份的姿态，几何误差和手掌翻转不能再交换两只手。汇总中的
`detector_handedness_mismatch_observation_count` 应为 `0`。

```bash
cd <ego_hand_system目录>
conda activate ego-hand
./scripts/run_multiview_wilor_experiment.sh \
  --mcap <MCAP文件绝对路径> \
  --output <新的六目实验输出目录> \
  --device cuda \
  --max-frames 60 \
  --batch-size 4
```
```bash
./scripts/run_multiview_wilor_experiment.sh \
    --mcap       /home/p3/data_sda1/ego_hand_system/recordings/20260818/DAS-Ego_20260818164752_none_none_689985_b5adb46c.mcap \
    --output     /home/p3/data_sda1/ego_hand_system/output/gen6_pose_full_v5_5090d \
    --conda-env  ego-hand \
    --device     cuda \
    --gpu-profile rtx5090d \
    --max-frames 0 \
    --batch-size 16
```

`--gpu-profile rtx5090d` 会使用 4 帧检测批次、16 个 WiLoR 假设批次、8 个并行
OpenCV 抗锯齿裁剪线程、FP16 autocast/TF32，并仅在每路相机结束后清理 CUDA 缓存。
WiLoR backbone 默认通过 `torch.compile` 合并 CUDA kernel；首次运行会有一次编译等待，
之后的完整序列会摊薄该开销。
由于严格 handedness 模式下每帧物理上最多只有一只左手和一只右手，该配置还会为
每个 detector 类别只保留最高置信度框，避免低阈值侧相机的重复假框成倍运行 WiLoR。
RTX 5060 保持默认的
`--gpu-profile compatible --batch-size 4`，继续使用原有逐帧 FP32 路径。可用
`--frame-batch-size` 和 `--preprocess-workers` 单独覆盖批次与裁剪线程数；显存不足
时先把 `--batch-size` 从 `16` 降为 `8`，再把帧批次从 `4` 降为 `2`。
如需保留所有候选框，可显式传入 `--max-detections-per-class 0`。
如当前 PyTorch/驱动组合无法编译，可用 `--compile-backbone 0` 关闭编译，其余 5090D
优化仍然保留。

确认 60 帧冒烟测试后，把 `--max-frames` 改成 `0`，同时使用新的输出目录运行完整
序列。脚本会生成 `fusion_multiview/diagnostic_6view.mp4`，其中六个画面按 3x2 排列；
绿色 `USED` 显示该相机实际贡献的内点关节，灰色 `INACTIVE` 表示未参与，红色
`OUTLIER/REJECTED` 显示离群候选或整帧拒绝原因。
短缺口会由邻近已确认姿态引导六路当前帧重新关联；恢复帧仍须至少两个相机的真实
观测通过 RANSAC，不直接使用关节插值代替识别。
`fusion_multiview/summary.json` 还会把六目结果和 `camera2+camera3` 双目结果投影到所有
可用视角，给出跨视角一致性对比。

六目结果确认无误后，可以直接导出与 WiLoR 训练数据一致的成对图片和 `.npy` 标签：

```bash
./scripts/run_multiview_wilor_label_export.sh \
  --experiment /home/zdh/ego_hand_system/output/gen6_pose_full_v3 \
  --fusion /home/zdh/ego_hand_system/output/gen6_pose_full_v3/fusion_handedness_strict_full \
  --output /home/zdh/ego_hand_system/output/gen6_pose_full_v3/wilor_training_labels \
  --conda-env ego-hand \
  --device cuda \
  --max-samples 0
```

导出结构：

```text
wilor_training_labels/dataset/
├── images/000000.jpg
├── labels/000000.npy
├── index.jsonl
├── rejected.jsonl
└── summary.json
```

`images` 和 `labels` 文件名严格一一对应。图片是 `camera2/3` 的共同针孔校正图像，标签内
的 `K` 与图片一致；不能把原始 DS 鱼眼图片和针孔 `K` 混用。每个标签会严格对照
`/home/zdh/tool/npy_decoder/000865.npy` 校验字段顺序、类型、dtype 和 shape，并验证
`vertices + trans` 经 `K` 投影后与 `joints_2d[778,2]` 一致。六目融合提供可靠 3D 和
身份，随后共享 MANO 时序拟合生成 778 顶点、21 关节、旋转矩阵、shape 和 translation。

路线含义：`mediapipe` 运行现有 MediaPipe→双目三角化→稳定化→MANO→渲染；`wilor`
只运行 WiLoR 左右目推理；`parallel` 顺序运行并保留两套结果。默认路线是
`mediapipe`，不设置 `EGO_HAND_ROUTE` 即保持旧行为。

## 分阶段运行与断点恢复

完整流程耗时较长，可以逐段执行：

| 阶段 | 命令 | 作用 |
|---|---|---|
| 配置检查 | `./scripts/run_offline.sh check` | 检查数据源、模型和 MANO 资产 |
| 输入准备 | `./scripts/run_offline.sh prepare` | Orbbec/GEN 标准化与双目校正；Orbbec 另含可选会话预览 |
| 双目重建 | `./scripts/run_offline.sh stereo` | MediaPipe、跨相机关联和 21 点三角化 |
| 3D 稳定 | `./scripts/run_offline.sh stabilize` | 离群管理、短缺口补全和骨长约束 |
| MANO 拟合 | `./scripts/run_offline.sh fit` | 稳定初值和低学习率精修两阶段拟合 |
| 最终导出 | `./scripts/run_offline.sh render` | 网格叠加、21-DOF 和手末端 6D CSV |
| 全部阶段 | `./scripts/run_offline.sh all` | 按顺序运行以上所有阶段 |

也可以通过环境变量选择阶段：

```bash
export EGO_STAGE=stereo
./scripts/run_offline.sh
```

每个阶段以 `summary.json` 或对应数据清单作为完成标志。再次运行时，已完成的阶段
会自动跳过；如果发现只有目录、没有完成标志，脚本会停止并提示使用新输出目录，
不会自动覆盖或删除可能有用的中间结果。

修改数据源、相机、`EGO_MAX_PAIRS` 或重要参数后，应使用新的 `EGO_OUTPUT`。

## 常用运行变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EGO_SOURCE` | 必填 | `orbbec` 或 `gen` |
| `EGO_SESSION` | Orbbec 必填 | Orbbec session 目录 |
| `EGO_MCAP` | GEN 必填 | GEN `.mcap` 文件 |
| `EGO_OUTPUT` | 必填 | 本次实验的统一输出根目录 |
| `EGO_LEFT_CAMERA` | `camera2` | GEN 左相机 ID |
| `EGO_RIGHT_CAMERA` | `camera3` | GEN 右相机 ID |
| `EGO_HAND_ROUTE` | `mediapipe` | `mediapipe`、`wilor` 或 `parallel` |
| `EGO_DEVICE` | `auto` | `auto`、`cuda` 或 `cpu` |
| `EGO_MAX_PAIRS` | `0` | 下游最大双目帧对数，`0` 表示全部 |
| `EGO_MAX_FRAMES` | `0` | GEN 每路最大解码帧数，`0` 表示全部 |
| `EGO_NO_VIDEO` | `0` | 设为 `1` 时跳过诊断视频，保留 CSV/JSON/NPZ |
| `EGO_WILOR_CAMERAS` | `both` | WiLoR 运行 `left`、`right` 或 `both` |
| `EGO_WILOR_BATCH_SIZE` | `16` | WiLoR 单个 worker 的手部 crop batch 大小 |
| `EGO_WILOR_FRAME_STRIDE` | `1` | WiLoR 每隔多少对帧处理一次 |
| `EGO_WILOR_FAST` | `0` | 设为 `1` 启用 FP16/编译加速，要求 CUDA |
| `EGO_CONDA_ENV` | `ego-hand` | 自动使用的 Conda 环境名 |
| `EGO_PYTHON` | 未设置 | 指定 Python 路径并绕过 Conda 自动选择 |

第一次验证新数据时建议使用独立的冒烟测试输出目录：

```bash
export EGO_OUTPUT=output/my_recording_smoke
export EGO_MAX_PAIRS=60

# GEN 还可以限制 MCAP 解码量
export EGO_MAX_FRAMES=80

./scripts/run_offline.sh all
```

冒烟测试通过后，将限制恢复为 `0`，并换一个新的正式输出目录。

## 输出目录

Orbbec 和 GEN 现在都先生成统一的标准化、校正数据集，再进入同一套后处理：

```text
$EGO_OUTPUT/
  session_check/                         # 仅 Orbbec，可选会话预览
  normalized/                            # 统一原始双目数据集（Orbbec/GEN）
  rectified/                             # 统一校正后的针孔双目数据集
  mediapipe_stereo/
    stereo_annotated.mp4
    stereo_frames.csv
    stereo_landmarks_3d.csv
    summary.json
  wilor_stereo/
    left/predictions.jsonl
    left/predictions.npz
    left/wilor_annotated.mp4
    right/predictions.jsonl
    right/predictions.npz
    right/wilor_annotated.mp4
    summary.json
  mano_preparation/
    mano_input.npz
    stabilized_landmarks_3d.csv
    summary.json
  mano_fit_optimized_initial_rigid/
    summary.json
    track_*.npz
  mano_fit_optimized_final/
    summary.json
    track_*.npz
  mano_overlay_optimized/
    mano_overlay_21dof.mp4
    mano_joint_angles_21dof.csv
    hand_end_effector_6d.csv
    summary.json
```

最先建议检查：

```bash
xdg-open "$EGO_OUTPUT/mediapipe_stereo/stereo_annotated.mp4"
xdg-open "$EGO_OUTPUT/mano_overlay_optimized/mano_overlay_21dof.mp4"
xdg-open "$EGO_OUTPUT/wilor_stereo/left/wilor_annotated.mp4"
xdg-open "$EGO_OUTPUT/wilor_stereo/right/wilor_annotated.mp4"
```

如果设置了 `EGO_NO_VIDEO=1`，请改看各阶段的 `summary.json` 和 CSV。

## 排错顺序

1. `check` 失败：先修正输入路径、Conda 环境或模型资产；
2. 双目视频中手身份跳变：检查 `mediapipe_stereo/stereo_annotated.mp4`，问题位于检测、左右关联或三角化；
3. 双目点稳定但网格不贴手：检查 `mano_fit_optimized_final/summary.json`；
4. GEN 无法解码：单独运行 `python scripts/check_gen_environment.py --mcap "$EGO_MCAP"`；
5. 某阶段留下不完整目录：保留它用于排错，并换一个新的 `EGO_OUTPUT` 重跑。

## 坐标与数据约定

- 三维坐标单位统一为米；
- `cam_0` 或所选左相机是参考相机，右相机是 `cam_1`；
- 双目 3D 和 MANO 初始输出位于原始左相机 OpenCV 光学坐标系；
- Orbbec KB 畸变使用 OpenCV `cv::fisheye`；
- GEN 支持 KB/Kannala–Brandt 和 Double Sphere；
- MediaPipe world landmarks 是手部模型相对坐标，不能代替双目相机坐标；
- 世界坐标仅由可选的 Basalt 阶段产生。

## 可选功能

### Orbbec 实时跟踪

实时模式不是当前离线主入口。需要 Orbbec SDK 本地运行时：

```bash
scripts/run_ego_live.sh
scripts/run_ego_live.sh --record
```

详见 [docs/EGO_REALTIME.md](docs/EGO_REALTIME.md)。

### Basalt 世界坐标轨迹

当前 Basalt 离线入口面向带 IMU 的 Orbbec session。完成离线渲染后运行：

```bash
python scripts/run_basalt_offline.py \
  --session "$EGO_SESSION" \
  --hand-pose-csv "$EGO_OUTPUT/mano_overlay_optimized/hand_end_effector_6d.csv" \
  --output-root "$EGO_OUTPUT/basalt_world"
```

详见 [docs/BASALT_STEREO_INERTIAL_WORLD_TRAJECTORY.md](docs/BASALT_STEREO_INERTIAL_WORLD_TRAJECTORY.md)。

## 构建与测试

C++ 会话检查和标定验证工具：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Python 回归测试：

```bash
conda run --no-capture-output -n ego-hand \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

更多说明：

- [双目深度、畸变和坐标流程](docs/STEREO_DEPTH_AND_DISTORTION_PIPELINE.md)
- [MANO 运动稳定策略](docs/MANO_MOTION_STABILITY_OPTIMIZATION_20260804.md)
- [21-DOF 仪表板与角度定义](docs/mano_21dof_dashboard_20260804.md)
- [参考结果](docs/RESULTS.md)
- [仓库内容与本地资产边界](docs/REPOSITORY_CONTENTS.md)
