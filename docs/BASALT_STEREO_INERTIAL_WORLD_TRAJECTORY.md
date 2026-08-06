# EGO 双目惯性 Basalt VIO 与手部世界轨迹

## 本阶段解决的问题

此前的手部末端 6D 位姿位于“当前左相机坐标系”。佩戴者头部或 EGO 相机移动时，即使手在房间中没有移动，相机坐标下的手位置也会发生变化，因此不能直接形成世界固定轨迹。

本阶段加入 Basalt 0.1.7 双目惯性 VIO：使用左右 KB 鱼眼图像、双目外参、相机到 IMU 外参、硬件时间戳和约 1000 Hz IMU，估计每帧左相机在重力对齐世界坐标中的位姿，再将现有 MANO 手部末端 6D 位姿转换到该世界坐标。

## 数据流

```text
EGO 左右视频 + PTS + 相机 KB 标定 + IMU CSV/标定
                         │
                         ▼
            prepare_basalt_dataset.py
     同步配对、缩放灰度图、时间偏移、IMU预校准
                         │
                         ▼
                 EuRoC 兼容数据目录
                         │
                         ▼
              Basalt stereo-inertial VIO
                         │
                         ▼
              每帧 T_world_imu / T_world_camera
                         │
      当前相机坐标下手部 T_camera_hand
                         │
                         ▼
          fuse_basalt_hand_trajectory.py
                         │
                         ▼
       T_world_hand、世界固定移动轨迹、叠加视频
```

## 坐标系与变换

- `I`：IMU 坐标系。
- `C`：左相机光学坐标系。
- `H`：手部末端坐标系。
- `W`：Basalt 重力对齐世界坐标系。

Basalt 输出 `T_W_I`，标定提供 `T_I_C`，原 MANO 结果提供 `T_C_H`。最终使用：

```text
T_W_C = T_W_I · T_I_C
T_W_H = T_W_C · T_C_H
```

正式 CSV 同时保存两套参考系：

1. `world_*`：保留 Basalt 的重力对齐方向，只把原点平移到第一帧左相机光心；`+Z` 为重力反方向（上方）。这是推荐的世界轨迹坐标。
2. `first_camera_*`：位置和方向都相对第一帧左相机。这一套更容易直观比较首末帧，但坐标轴不一定与重力对齐。

“第一帧”只定义局部世界的原点和航向，不提供地理绝对位置。没有 GNSS、外部定位或闭环地图时，VIO 会随时间累积漂移。

## EGO 数据适配

- 原始双目为 1600×1300 KB 鱼眼；默认转换为 800×650，Basalt 仍直接使用 `kb4` 投影模型，不先生成针孔去畸变图。
- 双目不是按视频相同帧号配对，而是按左右硬件 PTS 顺序匹配。
- 相机到 IMU 时间偏移写入图像时间戳；本设备标定值约为 `-12.358 ms`。
- YAML 中相机/IMU平移以毫米读取，写给 Basalt 前转换为米。
- 当前 EGO IMU 样本使用厂商标定矩阵预校准：

```text
a_cal = M_acc · a_raw - AccBias
w_cal = M_gyr · w_raw - GyrBias
```

预校准后 Basalt 静态 IMU 校准数组置零，避免重复校准。

## 一条命令运行

先进入项目并激活环境：

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

已有 `mano_overlay_trajectory/hand_end_effector_6d.csv` 时：

```bash
python scripts/run_basalt_offline.py \
  --session recordings/Orbbec_Ego_AZER764008C_20260805_171119
```

也可以明确指定手部位姿和输出目录：

```bash
python scripts/run_basalt_offline.py \
  --session recordings/Orbbec_Ego_AZER764008C_20260805_171119 \
  --hand-pose-csv output/recording_20260805_171119/mano_overlay_trajectory/hand_end_effector_6d.csv \
  --output-root output/recording_20260805_171119/basalt_world
```

中间 EuRoC 图像缓存约 0.5 GB。重复执行会复用完整数据集和 VIO 轨迹，不会自动删除或覆盖已有数据；需要换参数时请指定新的 `--output-root`。

## 输出

- `dataset/trajectory.csv`：Basalt 的 `T_W_I` 相机时刻轨迹。
- `dataset/calibration.json`：适配 EGO 的 Basalt KB4/IMU 标定。
- `dataset/ego_timestamp_map.csv`：EGO 帧号、PTS 与 Basalt 纳秒时间戳映射。
- `world_hand_trajectory/camera_trajectory_world.csv`：左相机世界 6D 位姿。
- `world_hand_trajectory/hand_trajectory_world.csv`：左右手末端世界 6D 位姿和相对首帧位移。
- `world_hand_trajectory/world_hand_trajectory_overlay.mp4`：把世界固定历史轨迹重新投影到当前左相机画面。

## 当前录制的验收结果

录制 `20260805_171119`：

- 1182 对同步双目帧，39.37 秒。
- 40260 对 IMU 样本，约 1002 Hz。
- Basalt 全量处理约 10.6 秒（无 GUI，8 线程）。
- 相机世界累计路径约 0.55 m；首末位移约 0.029 m。
- 相机帧间位移 P95 约 1.36 mm，未发现爆炸跳变。
- 两条手轨迹均输出 1182 帧；手部轨迹仍会继承 MediaPipe/MANO 自身的瞬时误差。

当前录制没有外部真值，所以上述统计证明的是时序完整、尺度为米制且轨迹没有发散，不能单独证明绝对定位精度。建议后续增加静止测试、已知长度往返测试、闭环回到起点测试，并与 AprilTag/光学动捕对比。

## Basalt 运行时与源码

- 源码：`third_party/basalt`。
- 本机离线运行时：`third_party/basalt_runtime`，固定为 0.1.7，只保留 `basalt_vio`、`libbasalt.so` 和版本说明。
- Basalt 使用 BSD-3-Clause 许可证。运行时二进制和大体积生成数据由 `.gitignore` 排除，GitHub 仓库应保留下载/构建说明而不是提交平台相关二进制。

## 下一阶段

1. 用 EGO 实时双目帧和 IMU 接入 Basalt 在线队列，而不是先落盘成 EuRoC。
2. 对 VIO 和手部检测采用统一硬件时间线，并处理端到端延迟。
3. 对手部世界轨迹增加基于 VIO 协方差与 MANO 观测质量的融合滤波。
4. 加入闭环或外部锚点，降低长时间漂移并得到可复现的房间坐标。
