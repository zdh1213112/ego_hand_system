# EGO-4 原版 Basalt / Monado VIO 与 NOKOV 对比

> 历史实验记录：下文数值使用当时 +1.802 ms 图像时间偏移。2026-11-08 起项目统一固定图像—IMU 修正为 0 ms，已取消相机时间修正选项。当前结果请查看 [零偏移单会话报告](../output/basalt_comparison_20261104_195844/comparison_report.html) 与 [零偏移跨会话报告](../output/basalt_calibration_comparison_195732_195844/report.html)；下文历史数值不代表当前报告。

实验日期：2026-11-04。本次基于本地源码、指定录制及 NOKOV 头环刚体实际运行。

结果目录：[完整实验](../output/archive/camera_time_policy_before_20261108/basalt_comparison_20261104_195844/)。
查看 [HTML 报告](../output/archive/camera_time_policy_before_20261108/basalt_comparison_20261104_195844/comparison_report.html)、
[同步对比视频](../output/archive/camera_time_policy_before_20261108/basalt_comparison_20261104_195844/comparison_video.mp4)、
[结构化结果](../output/archive/camera_time_policy_before_20261108/basalt_comparison_20261104_195844/comparison_summary.json)。

| 方法 | VIO 墙钟中位数/s | 吞吐/帧组每秒 | 后段位置 RMSE/cm | 后段姿态 RMSE/° | 全程 SE(3) ATE/cm |
|---|---:|---:|---:|---:|---:|
| 原版双目 | 18.65 | 86.2 | 19.33 | 2.44 | 8.10 |
| Monado 双目 | 18.90 | 85.1 | 13.09 | 2.88 | 5.31 |
| Monado 四目 | 36.89 | 43.6 | 9.05 | 2.59 | 3.71 |

每组完整运行三次，表中耗时和误差取中位数；全部 1608 帧均输出位姿。
共同数据 53.80 秒，960×600 灰度 KB4 鱼眼，1000 Hz 原始 IMU。
每组 TBB 并行上限 2，包含其他输入/输出流水线线程，非整个进程仅有两个线程。
VIO 耗时包含 PNG 读取、边缘化缓存写盘与轨迹导出，排除编译、视频转换和 NFR。
离线吞吐不等同于在线传感器到位姿的单帧延迟。

位置误差需要区分两种配准口径。后段指标只用前 2 秒确定世界旋转和平移，
固定尺度为 1，评价 t≥20 秒的漂移。全程 ATE 则对整段位置做 SE(3) 配准，
固定尺度为 1；这会移除一部分整体偏移。二者不可混用。

## 真值、同步和安装关系

录制：`EGO-4B_AZLCA610091_20261104_195844`。
真值：`/home/lff/xense_project/NOKOV/nokov-rigid-capture/sessions/session_20261104_195852_461949`。
使用 `orbbec_ego4` 刚体，90 Hz、5060 帧、毫米、xyzw，不使用手部资产评价头环位姿。

厂家文件提供相机间、相机与 IMU 的外参；未提供 NOKOV 刚体原点到 IMU 的安装关系。
NOKOV 元数据也明确未硬件同步。不能直接比较这两个不同刚体原点，或使用主机接收时间
代替观测时刻。

本次用前 20 秒原始 IMU 与动捕估计时钟偏移和安装关系，没有用 VIO 预测拟合它们。
原始 NOKOV 时钟单位由 90 Hz 下 11/12 tick 的间隔推断为毫秒。
估计时间关系：`t_imu = t_nokov_raw_ms × 1e-3 + 0.161781812 s`。
旋转来自陀螺仪与动捕角运动；安装平移来自加速度计与动捕刚体加速度关系，
重力模长约束为 9.81 m/s²。t≥20 秒保留为标定验证与误差评价部分。

这是带有同段传感器标定条件的结果，不是独立安装标定、硬件同步下的最终精度认证。
`nokov_calibration.json` 保留参数、角运动/加速度验证残差及标记点刚体一致性。
`comparison_summary.json` 还保留 ±5 ms 时间偏移和安装平移变化的敏感性结果。

## 建图

使用各版本自身 NFR 稀疏建图，统一 10 次优化、过滤异常点、再 10 次优化，TBB 上限 2。
最终 LM 对照结果如下；建图耗时各测一次，只表示本机本次配置的近似值。

| 方法 | 耗时/s | 唯一地图点 | 重投影 RMSE/px | 中位数/px |
|---|---:|---:|---:|---:|
| 原版双目 | 63.59 | 853 | 1.150 | 0.786 |
| Monado 双目 | 62.79 | 941 | 41.654 | 0.738 |
| Monado 四目 | 124.81 | 1493 | 1.034 | 0.658 |

Monado 双目 NFR 存在严重离群观测。共同 GN 配置的两次重投影 RMSE 分别约 6.91、203.71 px；
切换 LM 后仍有离群值，不能因退出码为 0、点更多或地图外观相似就认为质量合格。
原版与四目此次的 LM 重投影约 1 px。无独立三维表面真值，不能据此宣称地图几何精度。
GN 保留在 `map` / `map_exploratory`，LM 在 `map_lm`。

原版首次建图的 SIGSEGV 来自 Ubuntu oneTBB 节点的 `aligned_storage` 没有采用
32 字节对齐值类型的实际对齐要求。GDB 与局部 oneTBB 头文件兼容修复均保留。
修复只重编映射模块，不修改两个 Basalt 源码仓库或已经测量的 VIO 二进制。
导出器直接遍历唯一 landmark 数据库：原版旧 `get_current_points()` 的输出 ID 恒为 1，
不能对那个占位 ID 去重后统计地图点数。

## 输入与版本控制

录制器因四路总帧数不同将记录标为校验失败；各路自身视频与 PTS 数量一致。
采用真实时间戳交集和 IMU 覆盖范围，得到共同 1608 帧，相机间最大差 23 μs，
最长帧间隙 233.336 ms。双目复用四目数据的 cam0/cam1 图像、时间戳，输入字节相同。
按厂家 KB4 标定缩放内参，毫米外参转米；相机—IMU 时间偏移加到图像时间戳，
IMU 厂家校正只应用一次。原始录制、NOKOV 文件均保持只读。

- 原版：`6d8637b9d68ea18a1a63c1baa72818779e156932`。
- Monado：`df6e970c8da7636eb401a09e3317fbeaaf829b9a`。
- 原版头文件按 vcpkg 锁定 `aa441ba3e51050c47ba1902537792a2e4db7e43d`。
- 两者共同采用本机 Eigen 5.0.1、Sophus、OpenCV、TBB、GCC 13.3，Release -O3 -march=native，float VIO。
- 共同后端为 ABS_QR、平方根边缘化、3 状态、7 关键帧、最多 7 次迭代、LM 开启；recall 关闭。

## 复现

现有环境需要已配置的 `third_party/basalt_monado/build`、系统开发依赖和可导入
OpenCV/NumPy/SciPy/Matplotlib/PyYAML 的 Python。新实验写入新目录：

```bash
bash scripts/reproduce_basalt_ego4_comparison.sh output/basalt_comparison_repeat
```

该入口会下载原版锁定头文件、编译、准备共同输入、串行运行三次 VIO、运行 GN/LM NFR、
标定动捕关系、输出图表和视频。各阶段也可单独调用同名 Python 入口；每次运行的实际命令和日志
随结果保存。三项几何指标不变量测试及六项视频输入适配测试通过。
