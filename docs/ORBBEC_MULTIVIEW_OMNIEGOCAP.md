# Orbbec 四目手部、Basalt VIO 与 OmniEgoCap 输入

本流程对应录制 `EGO_Pro_AZLR86000F7_20260928_144343` 和检测器
`models/wilor/detector_1009.pt`。结果统一保存在本工程 `output/`，无需修改 OmniEgoCap 工程。

## 本次执行结果

最终输入为：

```text
output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap/input/omniegocap_input.npz
```

已通过目标工程 `EgoSequence.load()` 及 `make_features()`；全部特征为有限数值，
位姿旋转、缺失腕部占位和严格 30 Hz 时间间隔均通过检查。

| 项目 | 实际结果 |
| --- | --- |
| 原始同步四目帧 | 885 组 |
| 原始融合接受帧 | 648（820 个手部实例） |
| 融合重投影误差中位数 | 约 3.76 像素；不是三维精度或外部真值误差 |
| 最终输入 | 884 帧，30 Hz，0–29.433333 秒 |
| 左腕有效帧 | 305 |
| 右腕有效帧 | 257 |
| 双腕均有效帧 | 95 |
| 仅单腕有效帧 | 372 |
| 双腕均无有效观测帧 | 417 |
| 初始头部位置 | `(0,0,1.70)` 米，等于 IMU 位置加地面对齐平移 |
| 头部高度范围 | 约 1.470–1.702 米 |

最终有效腕部数量比原始融合手部数量少：257 个融合手部未通过构造腕部朝向所需的
掌面关键点检查，1 个通过掌面检查的手部观测超过有效 VIO 时间范围。原始融合结果
全部保留，未为了增加可见率放宽检查。没有有效腕部时，人体重建只能依靠头部和模型先验。

可视化检查文件：

- `input/poses_world.png`：世界坐标轨迹、高度、腕到头距离和可见标记。
- `multiview/fusion_multiview/diagnostic_4view.mp4`：885 帧四目诊断视频。
- `input/validation.json`：目标工程格式与特征验证结果。
- `provenance.json`：指定检测器、WiLoR 权重、配置和核心脚本的哈希及用户选择。

可复用已有推理和融合结果并重新验证输入：

```bash
cd /home/lff/xense_project/ego_hand_system
bash output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap/run_fusion_and_prepare.sh
```

完整首次推理命令见下节。若仅重新运行转换器，请指定新的 `--output` 目录；它会拒绝覆盖已有目录。
本次到生成并验证人体模型输入为止，尚未执行 OmniEgoCap 全身预测。

额外的旋转连续性检查发现两处相邻有效帧的腕部方向突变：右腕第 34→35 帧约
148.75°（约 1.17 秒），左腕第 103→104 帧约 64.76°（约 3.47 秒）。这说明格式和多目
重投影检查通过后，腕部朝向仍可能出现误差。使用者已明确选择**保留原始融合版本及这 4 个观测**。
最终应使用上面的 `omniegocap_input.npz`；选择和原文件哈希记录在 `selection.json`，异常记录在
`input/validation.json`。另有 `omniegocap_input_rotation_checked.npz` 保守候选文件供对照，
它不是本次选定的输入。可选后处理脚本为 `scripts/filter_omniegocap_wrist_rotations.py`；
本次不对选定输入应用该过滤。

## 当前执行的阶段

首先执行原有四目脚本的推理阶段；885 组原始图像全部参与，未使用默认的 60 帧限制：

```bash
cd /home/lff/xense_project/ego_hand_system
export PATH=/home/lff/miniforge3/bin:$PATH
export EGO_WILOR_ROOT=/home/lff/xense_project/WiLoR
SESSION=/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO_Pro_AZLR86000F7_20260928_144343
STAGE=output/EGO_Pro_AZLR86000F7_20260928_144343_multiview_det1009_inference

./scripts/run_multiview_wilor_experiment.sh \
  --session "$SESSION" --output "$STAGE" \
  --detector "$PWD/models/wilor/detector_1009.pt" \
  --conda-env wilor --device cuda --max-frames 0 \
  --gpu-profile compatible --batch-size 1 --frame-batch-size 2 \
  --preprocess-workers 2 --fusion-workers 2 \
  --no-video --stop-after-inference
```

检测模型为 YOLO26n，类别 0=left、1=right，使用项目内
`python_deps/wilor_runtime/ultralytics` 8.4.76。同目录的 scikit-image、pandas
用于解决原 Conda 环境与 NumPy 2 的兼容问题。没有更新全局 Conda 环境。
安装版本记录在 `requirements/wilor_camera3d_overlay.txt`。模型配置来自已有 WiLoR
预训练模型目录，复制到 `models/wilor/model_config.yaml`。

第一阶段只生成同步数据与四路双假设 WiLoR 预测，不生成最终三维融合结果。
推理完成以 `wilor_multiview/summary.json` 为准，
不能以部分 `predictions.jsonl` 的存在判断整段完成。

## 融合的选择

`--allow-single-hand-anchors 1` 可以保留通过几何检查的单手帧；另一只手在
OmniEgoCap 输入中标记为不可见。值为 0 时保留原脚本的双手筛选行为。
本次使用者已确认取 1。没有检测到的手不能伪造为可见手腕。

脚本会拒绝在同一输出目录混用不同参数。若修改融合选择，须使用新的输出目录，
并在确认第一阶段完整后复用同步数据与预测；不要覆盖已有 `run_config.json`。

## 转换成相机轨迹与双腕文件

融合完成后执行：

```bash
PYTHONPATH="$PWD/scripts" /home/lff/miniforge3/envs/wilor/bin/python \
  scripts/prepare_orbbec_omniegocap.py \
  --dataset FINAL_RUN/normalized_multiview \
  --fusion FINAL_RUN/fusion_multiview \
  --vio output/EGO_Pro_AZLR86000F7_20260928_144343_vio \
  --output output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap_intermediate
```

其中 `FINAL_RUN` 需要替换为真正完成融合的目录。输出：

| 文件 | 字段与含义 |
| --- | --- |
| `camera_trajectory.npz` | `T_world_cam (N,4,4)`：cam_0 光学系到 Basalt 世界的变换；`time_s (N,)`：相对首帧视频时间 |
| `wrists_camera.npz` | `wrist (885,2,4,4)`：cam_0 光学系中的左右腕变换；`visible (885,2)`；`time_s`；`joints_camera_m (885,2,21,3)` |
| `wrists_world.npz` | 同时刻 Basalt 世界系双腕变换和可见标记；地面尚未对齐 |
| `summary.json` | 来源、文件哈希、时间修正、质量检查、可见帧数和缺失标定信息 |

两只手的索引为 0=左、1=右。缺失腕部用单位矩阵占位并且 `visible=False`；
缺失三维关键点保留 NaN。腕姿来自融合后的腕点与掌面，不是单目模型的平移深度：

1. 取腕点 0、食指掌指关节 5、中指掌指关节 9、小指掌指关节 17。
2. 四点均有限且各有至少三路内点支持才建立可见手腕。
3. 局部 y 为手背法向，局部 x 为左手腕到中指方向、右手相反，局部 z=x×y。
4. 此朝向是依据掌面构建的近似解剖腕部朝向，符合目标工程的掌面构造方法；
   它不等同于完整 MANO 拟合的旋转参数，也不提供手指局部旋转。
5. `T_world_wrist = T_world_cam0 @ T_cam0_wrist`。融合已经将多目观测放到 cam_0，
   不重复应用相机间外参。

当前流程保留图像与 IMU 原始时间戳，时间修正固定为 0 ms；使用整数纳秒先做差，再转换成秒。
此前使用 2,637,831 ns 相机偏移的结果属于历史实验，需要重新运行零偏移 VIO 后再输入当前流程。
885 组视频中超过有效 VIO 时间范围的观测会标记不可见，
不会将轨迹末帧外推。最终输入使用严格 30 Hz 网格、旋转球面插值和位置线性插值；
手腕取最近样本，时间差不得超过 1/60 秒，不跨遮挡补造可见观测。

## 缺少的头部标定与地面信息

当前 `headring_poses.csv` 是 cam_0 的轨迹。OmniEgoCap 的 `head` 要求的是
SMPL-H **人体头部关节**轨迹，不能直接重命名替代。另需：

- `T_C0_H`：头部局部系到 cam_0 光学系的刚体变换，米制。头部局部轴是 x 左、y 上、z 前。
  计算 `T_W_H = T_W_C0 @ T_C0_H`。如果已知的是反方向矩阵，需要先取逆。
- `floor_z`：地面在原 Basalt 世界中的 z 高度。Basalt 世界虽然 z 向上，起始原点位于设备附近，
  并不是地面。头部、双腕、设备位姿统一减去这个高度，才能得到地面 z=0 的世界。

未提供两项参数时，转换脚本只导出中间文件，不会生成最终 `omniegocap_input.npz`。
如果使用者选择先用近似外参/地面跑实验，需要另行明确具体假设并记录到元数据。

### 本次已确认的 IMU 近似方案

使用者确认：头部关节位置暂取 IMU 位置；录像首帧头部直立正视前方；头环佩戴高度约
1.70 m；保留通过检查的单手帧。因此本次使用以下近似，不再将头部位置取为相机光心：

```text
p_W_H(t) = p_W_I(t)
R_W_H(0) 的 y 轴 = 世界 +Z
R_W_H(0) 的 z 轴 = 首帧 cam_0 镜头前向在世界水平面上的投影方向
R_W_H(0) 的 x 轴 = y × z（头部左方）
R_I_H = transpose(R_W_I(0)) × R_W_H(0)
R_W_H(t) = R_W_I(t) × R_I_H
floor_z = p_W_I(0).z − 1.70 = −1.70 m
```

其中 `trajectory.csv` 的首帧 IMU 位置为零。头部轨迹直接插值原始 IMU 位姿再应用固定旋转，
不对相机位置进行重命名。`T_I_H` 的平移为零，但旋转不是单位矩阵；IMU 与人体头部轴定义不同。
该旋转是首帧中立姿态假设得到的近似，不是测量出的安装外参。首帧近似不准确会使后续朝向产生固定偏差。

最终头部与可见手腕均减去 `floor_z`，即 z 坐标加 1.70 m，初始头部高度约为 1.70 m。
世界水平原点和朝向仍沿用原 VIO，后续高度变化和相对运动保留。1.70 m 为使用者提供的约数，
将头环高度近似为 IMU 初始高度，尚未测量地面平面或人体解剖关节位置。

本次转换参数为：

```bash
/home/lff/miniforge3/envs/wilor/bin/python scripts/prepare_orbbec_omniegocap.py \
  --dataset output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap/multiview/normalized_multiview \
  --fusion output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap/multiview/fusion_multiview \
  --vio output/EGO_Pro_AZLR86000F7_20260928_144343_vio \
  --head-at-imu --neutral-first-frame --floor-z -1.70 \
  --calibration-note '用户确认：头部位置取 IMU；首帧直立正视；初始 IMU 高度近似 1.70 m，未经安装外参/地面实测' \
  --output output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap/input
```

除通用中间文件外，此模式还保存 `imu_trajectory.npz` 和 `head_to_imu_approx.npy`。
元数据明确记录 `head_calibration_provided=false`，避免将此近似误认为真实安装标定。

提供这两项参数后的执行方式（文件和数值需替换为实际标定）：

```bash
/home/lff/miniforge3/envs/wilor/bin/python scripts/prepare_orbbec_omniegocap.py \
  --dataset FINAL_RUN/normalized_multiview \
  --fusion FINAL_RUN/fusion_multiview \
  --vio output/EGO_Pro_AZLR86000F7_20260928_144343_vio \
  --head-to-camera /path/to/T_C0_H.npy \
  --floor-z ACTUAL_FLOOR_Z \
  --calibration-note '标定来源、日期与方法' \
  --output output/EGO_Pro_AZLR86000F7_20260928_144343_omniegocap
```

最终文件含 `head (L,4,4)`、`wrist (L,2,4,4)`、`visible (L,2)`、
`device_pose`、`raw_time_s`、`device`、`meta`。位置单位米，左右腕使用同一地面对齐世界。
输出数组可以用 OmniEgoCap 的 `EgoSequence.load()` 检查。格式校验通过不代表标定正确或定位达到某种精度。

OmniEgoCap 接收的是头与双腕约束，预测全身人体运动；本流程本身不运行全身模型。
21 个手部关键点作为中间结果保留，不会自动变成 OmniEgoCap 的手指动作输出。
