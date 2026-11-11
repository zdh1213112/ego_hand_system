# Basalt for Monado：视频输入的多目头环定位复现

目标：输入刚性安装在头环上的 2/4/6 路同步视频，输出头环在局部世界中的位置和朝向。
先跑离线轨迹，再考虑 Monado 在线集成。离线运行不需要安装 Monado。

本文依据检查过的 Monado 分支 `main` 源码；当前 Ubuntu 24.04 机器已通过独立构建脚本完成 `basalt_vio` 编译，`--help` 启动验证通过，运行时使用系统 fmt v9 和 yaml-cpp。
新增加的视频转换入口已经通过四目、六目合成视频的编码/解码测试，尚未用真实头环数据验证定位精度。

## 1. 先选输入模式

| 输入 | 模式 | 需要的标定 | 输出参考系 |
|---|---|---|---|
| 多路同步视频，无 IMU | VO，`--use-imu 0` | 每目内参、相机间米制外参 | 相对初始机体坐标，不自动对齐重力 |
| 多路同步视频，加 IMU | VIO，`--use-imu 1` | 上述参数，加相机/IMU 外参、IMU 噪声与时间关系 | 局部重力对齐世界 |

已知且有实际共同观测的米制相机基线可以约束 VO 的尺度。视野完全互不重叠、只有旋转或没有足够视差时，不能承诺得到可靠米制平移。先选有重叠且能看到静态背景的双目跑通，再加入其他相机。

要求相机刚性固定，曝光时刻同步，时间戳来自同一时钟或已标定到共同时间线。相同 FPS、相同视频帧号或共同的录像开始时间均不能代替曝光同步。手和人应避免长期占据大部分画面。

## 2. 独立获取正确分支并构建

现有 `third_party/basalt` 和 `install_basalt_runtime.sh` 对应原版 Basalt，不能当成已安装了 Monado 分支。
使用独立目录保留现有依赖：

```bash
cd /home/lff/xense_project/ego_hand_system
git clone --recursive --branch main \
  https://gitlab.freedesktop.org/mateosss/basalt.git \
  third_party/basalt_monado
cd third_party/basalt_monado
git rev-parse HEAD > REPRODUCTION_COMMIT.txt
./scripts/install_deps.sh
bash ../../scripts/build_basalt_monado.sh
```

依赖安装脚本会调用系统包管理器和 sudo。构建脚本使用 Release，并启用离线工具；`library` 配置只生成给 Monado 加载的共享库，不生成这里需要的离线工具。
构建脚本明确选择 Ubuntu/Debian 的系统 fmt 和 yaml-cpp，同时为本次构建清理 Conda 搜索环境，修正已缓存的包路径。
如果内存不足，运行 `BASALT_BUILD_JOBS=1 bash ../../scripts/build_basalt_monado.sh`。完整工程默认包含可视化，因此安装依赖后再配置。
需要复现同一源码时，使用保存的提交号 checkout，并执行 `git submodule update --init --recursive`。

如果此前在 Conda `(base)` 中构建，并在链接阶段遇到 `undefined symbol: fmt::v9::vformat`，
同时链接命令中出现 `miniforge3/lib/libfmt.so.12`，就是系统 fmt v9 头文件与 Conda fmt v12 库混用。
直接运行上述构建脚本修正缓存并重新构建；仅执行 `conda deactivate` 不会清除缓存里的 `fmt_DIR`。
脚本也修正 Conda yaml-cpp 带入的运行库搜索路径，无需卸载 Conda 或修改系统库。

`Eigen version ..` 是 OpenGV 的旧检测脚本读取 Eigen 5 版本信息失败的状态提示；
`unused variable` 和 CMake 的旧版本兼容提示也不是此处的链接错误。
先检查真实的 `FAILED`、`undefined symbol` 或 `error`，不要仅凭这些提示更换 Eigen 或隐藏所有警告。

先用 README 给出的 `MOO09_short_1_updown` 样例检验安装：

```bash
cd /home/lff/xense_project/ego_hand_system
mkdir -p data/basalt_demo output/basalt_demo
curl -L --fail \
  'https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO09_short_1_updown.zip?download=true' \
  -o data/basalt_demo/sample.zip
unzip data/basalt_demo/sample.zip -d data/basalt_demo
```

下载并确认解压目录后，在 `output/basalt_demo` 中运行以下命令，轨迹将写到当前目录：

```bash
cd /home/lff/xense_project/ego_hand_system/output/basalt_demo
../../third_party/basalt_monado/build/basalt_vio \
  --dataset-path ../../data/basalt_demo/MOO09_short_1_updown \
  --dataset-type euroc \
  --cam-calib ../../third_party/basalt_monado/data/msd/msdmo_calib.json \
  --config-path ../../third_party/basalt_monado/data/msd/msdmo_config.json \
  --show-gui 0 --use-imu 1 --save-trajectory euroc
```

这一步使用样例自己的标定。不得把样例标定用于你的头环。

## 3. 准备自己的视频、时间戳与标定

新入口为 `scripts/prepare_basalt_video_dataset.py`，不依赖 MANO 或手部检测。

每一路视频配一份 CSV，每行对应一个解码后的视频帧，顺序与视频播放顺序相同：

```csv
timestamp_ns
1000000000
1033333333
1066666666
```

这里的数值只是格式示例。使用真实曝光时间戳，不用这个等间隔示例替代硬件时间戳。
时间戳必须非负、严格递增；不同视频可以有丢帧，但每路 CSV 必须与该路解码帧一一对应。
对于变帧率、B 帧和 MCAP/H264，须确认时间戳对应的是解码显示顺序。

标定文件必须是该分支可读取的完整原生 Basalt JSON，可参照其 `data/euroc/euroc_ds_calib.json` 的格式，但所有实际相机参数必须换成自己测得的值：

- `value0.intrinsics`：按视频顺序填写 N 目；KB 鱼眼为 `kb4`，参数为 `fx,fy,cx,cy,k1,k2,k3,k4`；Double Sphere 为 `ds`，参数为 `fx,fy,cx,cy,xi,alpha`。
- `value0.resolution`：按相同顺序填写 N 个 `[width,height]`。当前入口要求所有相机分辨率一致，并与原始视频匹配，不自动缩放或去畸变。
- `value0.T_imu_cam`：N 个相机到参考机体的变换 `T_B_Ci`，即 `p_B = T_B_Ci p_Ci`，平移单位米，四元数字段为 `qx,qy,qz,qw`。
- `value0.vignette`：同样需要 N 项；其余字段保留完整序列化结构，并按实际传感器配置。
- VIO 中 B 为 IMU；VO 中 B 可以选头环刚体或某个相机。名字虽然仍叫 `T_imu_cam`，VO 不需要虚构 IMU 测量。
- 不能把相机到机体的变换直接填成其逆变换，也不能把毫米直接当成米。
- 图像与 IMU 直接使用导出时间戳，仅转换时间单位，不向相机 CSV 追加时间偏移；`cam_time_offset_ns` 必须为 0。项目入口已取消时间修正选项，旧的非零偏移轨迹需重新运行。

项目中的 GEN 标准化流程已能导出 `cameras/cameraN/video.mkv`、`timestamps.csv` 和每目 `calibration/cameraN.json`。
这些每目 JSON 不是原生 Basalt 标定；其中 DS 参数完整保存在 `distortion=[fx,fy,cx,cy,xi,alpha]`，应使用这里的六个值，不要用可能不一致的 K 覆盖。
VO 可将 `T_base_camera` 用作 `T_B_Ci`；VIO 还需要真实的 base 到 IMU 变换，不能默认相同。

## 4. 先用两目，再扩展到四/六目

当前机器的系统 Python 缺少 OpenCV；已检查 `/home/lff/miniforge3/envs/wilor/bin/python` 可以导入 OpenCV。
以下命令在工程根目录运行，`/path/to/...` 均须替换为真实数据路径：

```bash
cd /home/lff/xense_project/ego_hand_system
/home/lff/miniforge3/envs/wilor/bin/python scripts/prepare_basalt_video_dataset.py \
  --videos /path/to/cam0.mp4 /path/to/cam1.mp4 \
  --timestamps /path/to/cam0.csv /path/to/cam1.csv \
  --calibration /path/to/headring_basalt_2cam.json \
  --config third_party/basalt_monado/data/default_config.json \
  --output output/headring_vo_2cam_trial \
  --max-frames 300 \
  --basalt-bin third_party/basalt_monado/build/basalt_vio
```

四目时，扩展为：

```bash
/home/lff/miniforge3/envs/wilor/bin/python scripts/prepare_basalt_video_dataset.py \
  --videos /path/to/cam0.mp4 /path/to/cam1.mp4 /path/to/cam2.mp4 /path/to/cam3.mp4 \
  --timestamps /path/to/cam0.csv /path/to/cam1.csv /path/to/cam2.csv /path/to/cam3.csv \
  --calibration /path/to/headring_basalt_4cam.json \
  --config third_party/basalt_monado/data/default_config.json \
  --output output/headring_vo_4cam_trial \
  --max-frames 300 \
  --basalt-bin third_party/basalt_monado/build/basalt_vio
```

六目同理；标定数组项数与视频数量、排列顺序必须完全对应。建议 cam0 选视野较稳定且与其他相机有重叠的一路。
无 IMU 时入口选择 `patch` 光流并关闭特征 recall，使用 `--use-imu 0`，避免依赖 IMU 预测。其余参数来自指定分支的配置。

默认要求各路时间戳完全相等。如果已知存在小量曝光偏差，可显式设置如 `--max-delta-us 1000`，仅作为实验配对阈值，不是精度保证。
脚本以第一路为参考，按真实时间戳选帧，不复用帧；输出保留每目原始时间与偏差。写入 EuRoC 时同组文件名统一，**这不会修正原始曝光时刻、运动畸变或时钟漂移**。
输出目录必须不存在；换参数使用新目录。

## 5. 有 IMU 时改为 VIO

在上述命令中添加 `--imu /path/to/imu.csv`，并使用真实 IMU 标定齐全的 Basalt 标定文件：

```csv
timestamp_ns,wx,wy,wz,ax,ay,az
990000000,0.001,0.002,0.003,0.01,0.02,9.81
```

角速度单位 rad/s，加速度单位 m/s²；加速度必须包含传感器测得的重力分量，不预先扣重力。
此处同样只是格式示例。IMU 数据覆盖首帧到末帧，并至少延伸到末帧之后一个样本。
相机与 IMU 采用共同时间线。厂商预校准与 Basalt 内部静态校准不能重复应用。
脚本会自动运行 `--use-imu 1`，保留所给配置中的视觉前端设置。

## 6. 输出与头环坐标

输出目录中包含：

```text
mav0/cam0/data.csv + data/*.png
mav0/cam1/data.csv + data/*.png
... cam2 至 camN
mav0/imu0/data.csv
calibration.json
config.json
frame_map.csv
summary.json
basalt.log           # 指定 --basalt-bin 时
trajectory.csv       # Basalt 成功完成时
```

纯视觉模式的 IMU CSV 只有表头，没有伪造数据。`trajectory.csv` 使用 `--save-trajectory euroc`，列顺序是：

```text
timestamp_ns, tx, ty, tz, qw, qx, qy, qz
```

轨迹表示 `T_W_B`，不是默认表示某个相机或头环中心。
如果 B 是 IMU、H 是定义好的头环中心，已知固定安装变换 `T_B_H`（头环到 IMU），则：

```text
T_W_H = T_W_B × T_B_H
T_W_Ci = T_W_B × T_B_Ci
```

标定采用左乘、列向量约定。VO 若直接以头环为 B，则轨迹就是头环位姿。
世界原点和初始航向由初始化决定；需要 NOKOV/房间统一坐标时，还要标定 `T_room_W`，再左乘轨迹。得到轨迹文件不代表已经达到毫米或厘米精度。

## 7. 验收与排错

1. 静止录制：检查位置漂移和姿态抖动，排查时间线、模糊和动态物体。
2. 约 1 m 已知平移：检查运动方向和尺度；尺度倍数异常优先检查外参单位。
3. 绕头环中心转动：考虑传感器到头环中心的杆臂位移，检查变换方向。
4. 回到起点：量化末帧闭合误差；VO/VIO 本身允许累积漂移，不应当作自动回环已启用。
5. 有 NOKOV 真值：先做时间同步与刚体安装变换，再统一局部世界坐标，计算 ATE/RPE。
6. 对比双目、四目、六目：检查丢组比例、特征数、耗时与真值误差；增加相机不保证精度单调改善。

若不能生成轨迹，查看 `basalt.log`；若配对组数少，查看 `frame_map.csv` 和原始时间线；若全部图像都能读但轨迹发散，先查标定模型、外参方向、单位和背景纹理，再调算法参数。

## 原始资料

- 项目：https://gitlab.freedesktop.org/mateosss/basalt
- 离线 CLI：https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/src/vio.cpp
- 多目 EuRoC 读取：https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/include/basalt/io/dataset_io_euroc.h
- 纯视觉路径：https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/doc/Vo.md
- 标定：https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/doc/Calibration.md
