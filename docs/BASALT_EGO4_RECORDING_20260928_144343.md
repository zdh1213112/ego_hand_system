# Orbbec EGO Pro 四目惯性定位与稀疏地图结果

> 本文保留历史实验数值，其中图像曾追加约 +2.638 ms 的偏移。2026-11-08 起项目统一保留图像与 IMU 导出时间戳，修正固定为 0 ms；当前运行入口拒绝非零偏移旧结果，需要重新准备并运行 VIO。

输入目录：

`/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO_Pro_AZLR86000F7_20260928_144343`

结果目录：

`output/EGO_Pro_AZLR86000F7_20260928_144343_vio`

## 已完成的运行

- 原始四路：left、right、right_side、left_side，对应厂商 cam_0、cam_1、cam_2、cam_3；每路 885 帧，1600×1200，约 29.467 秒。
- 实际 VIO：四目 KB4 + IMU，图像缩放为 800×600，Basalt Monado 分支，2 线程；884 个位姿，覆盖 29.433684 秒，VIO 运行耗时约 20.15 秒。
- IMU：29446 对加速度/陀螺仪样本，约 1000 Hz。使用厂商 `M_acc/M_gyr` 和偏差预校准，Basalt 静态校准数组置零，避免重复校准。
- 将相机时钟转换到 IMU 时钟：`t_basalt_ns = t_camera_us × 1000 + 2637831`；偏移来自本设备标定，约 +2.638 ms。最后一组相机帧校正后超出 IMU 覆盖，已排除，没有补造惯性数据。
- 按真实时间戳配对，最大相机间时间差为 11 µs；保留每路原始时间及配对偏差。
- 已保存 119 个有效边缘化快照，用 Basalt NFR 流程完成特征检测、匹配、三角化、优化、异常观测剔除与再次优化。
- 地图：126 个关键帧、2688 个唯一三维路标；额外提供 10 米范围内的 2678 点预览版本。10 米版本仅用于查看，完整地图保留所有路标。

## 主要文件

| 文件 | 内容 |
|---|---|
| `headring_poses.csv` | 每帧头环参考点的世界位置、四元数、ZYX 欧拉角；含原始 cam_0 时间戳 |
| `headring_poses.tum` | 相同位姿的 TUM 格式 |
| `trajectory.csv` | Basalt 原始 IMU 位姿轨迹，EuRoC 格式 |
| `headring_trajectory.png` | 位移、姿态变化和轨迹图 |
| `map/sparse_map.ply` | 全部 2688 个唯一三维路标，ASCII PLY |
| `map/sparse_map_10m.ply` | 距平均 VIO cam_0 位置 10 m 以内的查看子集 |
| `map/sparse_landmarks.csv` | 路标 ID 与三维坐标 |
| `map/keyframe_headring_poses.csv` | 全局地图优化后的关键帧头环参考位姿 |
| `map/keyframe_imu_poses.csv` | 全局地图优化后的关键帧 IMU 位姿 |
| `sparse_map_preview.png` | 地图三维图及俯视图 |
| `dataset/frame_map.csv` | 解码帧号、原始时间戳、配对偏差 |
| `dataset/calibration.json` | 本次实际使用的四目 KB4/IMU 标定 |
| `dataset/config.json` | 本次实际运行配置 |
| `input_adapter/metadata.json` | 外参组合、单位转换、时间偏移、尾帧排除记录 |
| `marg_data/` | Basalt NFR 后续建图缓存 |
| `vio.log` / `mapping.log` | 原始运行日志 |
| `summary.json` / `map/summary.json` | 机器可读结果统计 |

## 坐标约定

`T_A_B` 将 B 坐标下的点变换到 A 坐标。

厂商没有提供头环几何中心的独立安装标定，因此当前“头环位姿”以厂商参考相机 **cam_0 光学坐标系** 定义；原始 IMU 位姿同时保留。

```text
T_world_cam0 = T_world_imu × T_imu_cam0
```

世界坐标为 Basalt 初始化的局部重力对齐坐标，长度单位米；不是 NOKOV/房间绝对坐标。需要真正的头环中心位姿时，再提供固定安装变换 `T_imu_head`，计算 `T_world_head = T_world_imu × T_imu_head`。

具体初始化：原点为首个有效相机时刻的初始 IMU 位置；起始加速度方向被旋转到世界 +Z，以此估计竖直向上。
世界 X/Y 位于估计水平面，绕竖直轴的方向由程序初始化约定选择，没有北向或房间方向的绝对参考。
世界坐标固定；IMU 和各相机的局部坐标随头环运动。相机光学坐标采用 X 向图像右、Y 向图像下、Z 向镜头前方。
首帧 cam_0 世界位置约 `(0.026953, 0.050004, 0.001180)` 米，因相机与 IMU 存在安装偏移，因此不为零。

四目共享一条刚体运动估计，但每目位姿都能通过 `T_world_cam_i = T_world_imu × T_imu_cam_i` 换算。
已额外导出 `camera_poses/cam_0_world_poses.csv` 至 `cam_3_world_poses.csv`，每份 884 帧；
这些是同一刚体运动的不同传感器参考点，不是四次独立 VIO。
已验证 cam_0 与 `headring_poses.csv` 一致，cam_0/cam_1 的米制间距在所有时刻保持约 0.099932 米。
坐标解释图为 `coordinate_frames_explained.png`；换算脚本为 `scripts/export_basalt_camera_poses.py`。

CSV 四元数顺序：`qw,qx,qy,qz`；TUM 顺序：`qx,qy,qz,qw`。
VIO 每帧轨迹与 NFR 优化关键帧轨迹分别保存，不把低频全局优化结果冒充每帧重新估计的位姿。

## 数值检查与精度边界

VIO cam_0 参考轨迹累计长度约 **1.870 m**，首末位移约 **0.374 m**；帧间平移 P95 约 **8.98 mm**，最大估计速度约 **0.510 m/s**。
这些是运动统计，不是误差指标；录制没有外部定位真值，不能由此声称达到毫米或厘米定位精度。

已检查：位姿时间严格递增、四元数归一化、四目外参与项目独立厂商标定读取器一致；VIO 日志没有重置或求解失败。

地图属于优化后的**稀疏特征点云**，没有 RGB 纹理和连续表面，也不是可直接重载的完整重定位数据库。
录制有手部和其他运动物体，地图可能残留错误路标；完整点云含远处低视差路标，10 米预览并不是精度筛选或真值验证。
上游可视化接口重复返回了同一路标的多个观测，导出时按路标 ID 去重；2688 是唯一路标数，不能把重复观测数当成地图点数。

## 重复运行

以下命令均在工程根目录执行。输出必须使用新目录，避免覆盖已有定位或建图结果。

```bash
/home/lff/miniforge3/envs/wilor/bin/python scripts/run_orbbec_headring_vio.py \
  --session /home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO_Pro_AZLR86000F7_20260928_144343 \
  --output output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat \
  --image-scale 0.5 --num-threads 2 --save-marg-data

python scripts/build_basalt_mapper_export.py

third_party/basalt_monado/build/basalt_mapper_export \
  output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat/dataset/calibration.json \
  output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat/dataset/config.json \
  output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat/marg_data \
  output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat/map

/home/lff/miniforge3/envs/wilor/bin/python scripts/plot_basalt_headring_result.py \
  --result output/EGO_Pro_AZLR86000F7_20260928_144343_vio_repeat
```

无界面地图导出器源码为 `tools/basalt_mapper_export.cpp`，直接调用本次 Basalt 构建的 NFR API；构建脚本复用已有 Ninja 编译与链接参数，不修改第三方源码。
