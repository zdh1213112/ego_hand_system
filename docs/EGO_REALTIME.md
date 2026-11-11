# EGO 实时双目手部跟踪

## 当前实时链路

```text
EGO COLOR_LEFT / COLOR_RIGHT MJPG
  -> OrbbecSDK FrameSet 与设备时间戳
  -> C++采集/输出双线程 + Python有界队列，默认只保留最新完整双目帧对
  -> KB 鱼眼双目校正
  -> 左右 MediaPipe 21点并行推理
  -> 极线约束的跨相机关联
  -> 校正图稀疏LK前后向匹配与指尖亚像素细化
  -> 双目三角化（左相机光学坐标系，米）
  -> 深度/重投影/极线质量门控
  -> 2D运动 + 3D位置/深度的轨迹ID关联
  -> 质量加权在线滤波、在线骨长约束与最多5帧短时预测
  -> CUDA MANO warm-start 拟合（固定shape，优化pose/orient/transl）
  -> 原始左鱼眼网格 + 双手21-DOF面板 + 3D预览
  -> 实时窗口、3D/角度CSV和可选录像
```

当前双目实时入口支持以下 EGO 设备：

- USB VID/PID：`2bc5:1201`（左右流 `1600x1300 @ 30 FPS MJPG`）；
- USB VID/PID：`2bc5:1204`（EGO-4B，左右流 `1920x1200 @ 30 FPS MJPG`）；
- Linux UVC 后端：`LibUVC`；
- 标定：启动时从设备 Flash 的 `OB_RAW_DATA_ALIGN_CALIB_YAML` 读取；
- 坐标：统一输出为原始左相机光学坐标系下的米制3D。

程序会拒绝其他 VID/PID，避免错误地把普通 Orbbec 深度相机当作 EGO。
`1204` 的标定包含四目，但当前实时入口仅用 `cam_0/cam_1` 做双目 3D；
侧右 `cam_2`、侧左 `cam_3` 尚未进入实时融合。

旧款 `1201` 启动日志如果显示 `connection=USB2.0`，双路 `1600x1300 MJPG` 的完整帧对可能
只能达到约8–12 FPS。应优先将设备直连USB3端口，避免USB2集线器、扩展坞或
仅支持USB2的数据线。设备当前没有提供低分辨率MJPEG profile；1600×1200的
60 FPS profile是H264/H265，需要单独实现有状态硬件视频解码，不能用MJPEG
逐帧解码器直接替换。

## 项目内运行依赖

本地运行包应包含：

```text
third_party/orbbec_sdk/
  include/libobsensor/       OrbbecSDK C/C++ 头文件
  lib/libOrbbecSDK.so*       SDK 动态库
  lib/extensions/            SDK 运行扩展
  OrbbecSDKConfig.xml        EGO流配置与低延迟队列配置
  99-obsensor-libusb.rules   Linux USB权限规则
  install_udev_rules.sh      权限安装脚本
```

该目录由 `.gitignore` 排除。发布到 GitHub 时，应让使用者从设备资料包或
Orbbec 官方 SDK 获取同版本文件，并确认其分发许可。不要把未知许可状态的
供应商二进制直接公开提交。

## 首次安装 USB 规则

```bash
cd /home/p3/data_sda1/ego_hand_system
sudo /path/to/OrbbecSDK/shared/install_udev_rules.sh
```

安装后重新拔插 EGO。

## 启动

先确认 GPU 环境：

```bash
conda activate ego-hand
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
cd /home/p3/data_sda1/ego_hand_system
export ORBBEC_SDK_ROOT=/home/p3/data_sda1/EGO_glove/OrbbecSDK_v2.9.0_202610120915_6f302abc_linux_x86_64
scripts/run_ego_live.sh
```

按 `Esc` 或 `Q` 退出。保存实时预览：

```bash
scripts/run_ego_live.sh --record
```

默认启用 MANO。需要回到左右校正图21点诊断视图时：

```bash
scripts/run_ego_live.sh --no-mano
```

CPU跟不上时可隔帧推理：

```bash
scripts/run_ego_live.sh --process-every 2
```

无界面短测试：

```bash
scripts/run_ego_live.sh --max-frames 60 --no-display --record
```

输出：

- `output/ego_live/ego_camera_calibration.yaml`：从当前真机读取的标定；
- `output/ego_live/live_landmarks_3d.csv`：实时原始/滤波3D、深度质量和预测状态；
- `output/ego_live/live_mano_21dof.csv`：每帧每只手21个运动学自由度（弧度/角度）；
- `output/ego_live/live_mano_21dof.mp4`：默认 MANO 模式使用 `--record` 时生成；
- `output/ego_live/live_stereo_annotated.mp4`：`--no-mano --record` 时生成。

## 深度如何提高双手实时跟踪

现有程序不是将 MediaPipe handedness 直接当作永久ID，而是联合使用：

1. 左校正图手掌中心与速度预测；
2. 三角化手掌中心的三维欧氏距离；
3. Z深度差；
4. handedness 作为软惩罚；
5. 两个长期轨迹槽位和最多90帧的丢失恢复窗口。

当双手在图像中交叉但前后深度不同，3D距离和Z深度会抑制轨迹互换。
每个关节点还根据极线误差、重投影误差、视差和检测置信度计算
`depth_quality`。质量差或深度突跳的观测不会直接覆盖轨迹，而是短时使用
速度预测，灰色骨架表示预测点。

### 稀疏极线亚像素细化

左右 MediaPipe 仍负责提供21个关节的语义初值。程序在校正灰度图上使用稀疏
Lucas–Kanade 前后向匹配细化右目位置，并以左目 y 作为水平极线初值；指尖
使用更大的局部窗口。修正只有同时满足以下条件才会被采用：

- LK光度误差通过门限；
- 左到右、右到左的回查误差通过门限；
- 相对水平极线的垂直残差通过门限；
- 相对 MediaPipe 右目初值的水平移动没有超过门限；
- 修正后视差仍为合理的正值。

未通过的点保留原始 MediaPipe 像素，不会被强制移动；但其深度质量和右目
MANO 2D权重会降低，指尖失败时惩罚更强。实时CSV新增
`refinement_used/refinement_quality`，退出时阶段耗时新增
`refine_triangulate`。

默认参数适合当前1600×1300校正图：

```bash
scripts/run_ego_live.sh \
  --refine-window 17 \
  --refine-tip-window 25 \
  --refine-max-x-shift-px 18 \
  --refine-max-y-residual-px 3 \
  --refine-max-fb-error-px 2
```

需要对照旧行为时可禁用：

```bash
scripts/run_ego_live.sh --no-stereo-refine
```

在参考录像完整398对回归中，跨相机匹配保持395对、轨迹保持2条；修正接受率
为22.9%，有效3D点由11,018增至11,126（75.1%增至75.8%），五个指尖平均
有效率由68.8%增至69.6%。极线误差中位数由2.828 px降至1.752 px，重投影
误差中位数由1.414 px降至0.876 px；处理速度由15.11变为15.06双目对/秒。
该结果说明细化生效且没有牺牲轨迹连续性或实时吞吐，但仍需真机快速运动、
互相遮挡和边缘视场测试。

## 实时 MANO 与21自由度

每只手首次出现时用双目3D掌部点做刚体初始化，随后固定该手的10维 MANO
shape，并以上一帧的 pose/orient/transl 为初值执行少量 CUDA 优化。优化同时
使用滤波后的米制3D关节点、左右校正图2D重投影、姿态先验和时序项。这样网格
不是简单套在 MediaPipe 2D骨架上，而是受双目深度约束。

拇指输出 CMC屈伸/外展/对掌、MCP屈伸、IP屈伸共5项；其余每指输出 MCP
屈伸/MCP外展、PIP屈伸、DIP屈伸共4项，总计每手21项。观测不足7个有效3D点
时不接受新的拟合，保留上一状态并在角度CSV中写 `observed=0`，画面网格变灰。

### 对齐的目标版本

实时版以 `output/mano_overlay_21dof/mano_overlay_21dof.mp4` 为视觉基准。该视频
来自 `output/mano_fit_refined`：`track 0 = Right`、`track 1 = Left`，离线使用
24帧窗口、4帧重叠和60次姿态优化，角度使用半径2帧中值平滑。

实时版无法使用未来帧或每帧60次优化，但现在保持相同的MANO shape profile、
左右手模型、颜色、原始鱼眼投影和21-DOF定义，并使用5帧因果中值角度平滑。
如果累计 handedness 从初始误判翻转，实时拟合器会销毁错误的镜像MANO状态并
按正确左右手重建；不会再出现两条轨迹都永久使用橙色Left模型的问题。

## 当前边界与调参

当前流水线并行运行左右 MediaPipe，并在同一GPU上更新两只手 MANO；实际帧率取决于手数、
遮挡和 GPU。默认稳定帧每手3次优化，首次20次。可用以下参数权衡速度与精度：

```bash
# 更快
scripts/run_ego_live.sh --mano-iterations 2

# 更精细但更慢
scripts/run_ego_live.sh --mano-iterations 5 --mano-initial-iterations 30
```

稳定帧如果拟合损失仍高于默认阈值，会自动追加最多2次MANO更新；因此
`--mano-iterations 2` 不再表示质量差的帧永远只计算两次。后台采集线程会丢弃
已经过时的帧，而不是让它们在管道中排队，因此界面中的 `drop` 增长是主动的
低延迟策略，不是双目同步失败。

如果新细化在低纹理或运动模糊场景带来额外开销，可先比较：

```bash
scripts/run_ego_live.sh --no-stereo-refine
```

退出时检查 `Stage averages` 中的 `detect`、`refine_triangulate` 和 `mano`，不要
仅根据窗口观感判断瓶颈。

默认 `--capture-queue 1` 优先降低延迟；USB2成批到帧时若更看重连续性，可试：

```bash
scripts/run_ego_live.sh --capture-queue 2
```

实时 warm-start 不等同于离线全序列精修：离线版本可以前后看完整序列并执行
数百次优化，实时版本只能使用当前及历史帧。严重互相遮挡、快速运动或指尖双目
不一致时，局部手指仍可能偏离。后续若需要更高帧率，应将采集、MediaPipe和
MANO拆为异步流水线，或接入训练好的 MANO 参数回归网络。

## EGO-4B 四目 30 FPS 实测与改造目标（2026-11-03）

测试机器：i9-14900K、RTX 5090 D v2、OrbbecSDK 2.9.0、EGO-4B
`AZLCA610091`。四路 MJPG 在原生标定分辨率 `1920x1200` 下均稳定输出
约 30 FPS。单独的双目桥接连续接收 120 对帧，实测 29.99 对/秒，
没有丢帧。设备报告 `USB2.1`，但当前压缩码流的实测采集吞吐已达到 30 FPS。

现有 Python 实时链路仍达不到每秒 30 次完整 3D/MANO 更新：

| 测试 | 完整输出吞吐 | 处理阶段吞吐 | 主要耗时 |
| --- | ---: | ---: | --- |
| 双目 3D、无显示、180 帧、无手场景 | 16.3 FPS | 27.4 FPS | 每对解码 7.7 ms、校正 4.4 ms、检测 23.4 ms |
| 双目 3D、无显示、180 帧、有手场景 | 15.1 FPS | 22.3 FPS | 每对解码 8.6 ms、校正 3.6 ms、检测 31.7 ms |
| 双目 3D + MANO、无显示、120 帧 | 11.5 FPS | 19.5 FPS | 仅 3 次有效 MANO 拟合：342、50、41 ms |

完整输出吞吐包含相机/进程启动开销；处理阶段吞吐只统计处理时间，
没有计入等待采集和显示。这组有手测试只产生 3 次有效 MANO 拟合，
还不能代表持续双手观测的稳定速度。应先检查主双目的手部检测和匹配质量，
再做更长时间的双手基准。仅比较窗口 FPS 会掩盖丢帧和姿态更新率。

要实现每个相机时刻都有**真实更新**的四目 3D 关键点与 MANO 姿态，下一阶段需要：

1. 将 C++ 桥接扩展为四路带设备时间戳的压缩帧输出；用有界最新帧队列保持低延迟，
   并用当前设备 Flash 的 `cam_0..cam_3` 标定。现有桥接只输出 `cam_0/cam_1`。
2. 把四路逐帧全图 MediaPipe 改为 GPU 批量手框/关键点推理和 ROI 跟踪；
   检测到遮挡时再动用侧目。必须在真实双手动作中测四路检测的 P95 耗时和召回率。
3. 用四相机 KB 射线、多视角身份关联及鲁棒三角化生成 3D 关键点，
   再做因果滤波；不能把四路鱼眼像素直接套用双目极线匹配。
4. 将逐帧 CUDA MANO 迭代优化换成单次前向的姿态回归，或将耗时的精修放在
   后台并明确标记预测帧。当前 41–50 ms 的稳定拟合一次就超过 33.3 ms 的整帧预算。
5. 将预览和录像移到独立线程，避免图像绘制占用推理预算；端到端验收要求
   连续双手动作至少 300 帧、四路设备时间戳对齐、真实 3D/MANO 更新率 ≥30 Hz，
   同时记录丢帧、延迟 P50/P95、每阶段耗时和重投影误差。

`1204` 设备识别、主双目标定和无显示模式已修复。下方的四目 WiLoR 入口
实现了四路采集、主双目逐帧推理、侧目按需推理及 KB 多视角三角化；
跨相机长期身份跟踪、因果滤波与 30 Hz 双手输出仍需继续完善。

### 当前 WiLoR 权重的实时预算

用户当前检测器是 `models/wilor/detector0.pt`，姿态权重是
`models/wilor/wilor_final.ckpt`。使用项目现有双假设 WiLoR 路径，
在旧四目录像的一组 `1600x1200` 图像上重复测试，排除首次预热后：

| 输入 | 检测与裁剪中位数 | WiLoR 中位数 | 总中位数 | 假设数量 |
| --- | ---: | ---: | ---: | --- |
| 主双目 | 27.2 ms | 30.0 ms | 54.6 ms | 左目 4、右目 2 |
| 四目 | 32.8 ms | 30.4 ms | 63.1 ms | 4、2、0、2 |

这些数值不包含相机采集、MJPG 解码、四目同步、3D 融合和显示；
录像分辨率也低于当前设备的 `1920x1200`。因此现有权重和代码直接逐帧运行
达不到 30 Hz。四目实验的侧相机检出数较少，不能据此推断完整双手可见时的成本。

针对“主双目每帧、侧目按需”的目标，应先做以下优化并逐项计时：

1. 每隔若干帧运行 `detector0.pt` 全图检测，其间用经验证的跟踪框送入 WiLoR；
   丢失或遮挡时立即重新检测。这样必须量化遮挡恢复延迟和误检。
2. 对 detector 已确定左右身份且置信度足够的框只运行对应 WiLoR 假设；
   模糊帧才运行双假设。当前离线路径对每个框都运行两个假设。
3. 将主双目的手部 crop 合成一次 GPU 批次，侧目只在主视角质量不足时加入；
   异步重叠相机采集、CPU JPEG 解码、GPU 推理与结果显示。
4. 以完整连续动作重新计时，要求每秒至少 30 次**非预测**的 3D 与 MANO
   更新，同时验收精度、手身份和 P95 延迟。仅让窗口以 30 FPS 刷新不算达标。

## 四目实时 WiLoR 开发入口

新入口使用当前 `detector0.pt` 和 `wilor_final.ckpt`，与上面的 MediaPipe
入口相互独立。四目 MJPEG 始终采集；`cam_0/cam_1` 逐帧送入 WiLoR，
`cam_2/cam_3` 在主双目 3D 失败时按需推理。默认先用 `cam_2`，仍缺失时再用
`cam_3`；也可设置周期扫描。两目使用批量 KB 射线三角化，更多视角时选择
有效点最多且跨视角重投影一致的两目候选。仅通过重投影、深度和同步门限的
3D 点标为有效。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
export ORBBEC_SDK_ROOT=/home/p3/data_sda1/EGO_glove/OrbbecSDK_v2.9.0_202610120915_6f302abc_linux_x86_64
scripts/run_ego_live_wilor.sh --max-frames 300
```

在桌面会话中，上述命令默认打开 `EGO-4B Live WiLoR` 窗口：左侧显示
`cam_0` 主相机和左右手框；白色关节是同一帧双目三角化 3D 点重投影回
该相机的结果。可用 `--preview-camera cam_1` 改为另一台主相机。右侧
分别列出左、右手每根手指的三个关节弯曲角，共每手 15 个：拇指
CMC/MCP/IP，其他手指 MCP/PIP/DIP。角度由同一帧的双目 3D 点计算，
是相邻骨段之间的**无符号几何角**；`0°` 表示骨段成一直线，它不区分
屈曲与过伸，也不是独立的解剖学关节旋转。缺少有效 3D 点时显示 `--`。
窗口下方显示左右手的 WiLoR/MANO 有阴影表面网格，不再显示骨架或三维坐标
网格。下方面板采用与 `cam_0` 对齐的正面投影。MANO 顶点从当前帧 WiLoR
推理结果取得，并通过刚体配准对齐到同帧的双目三角化关节；两只手使用
共同的米制坐标和固定比例，不对网格逐关节拉伸。双目关节缺失或配准
误差过大时不绘制该手网格。上方重投影、右侧角度和下方网格配准都读取
`hands[].joints_3d_base_m_display` 双目 3D 显示点。显示点采用逐帧因果
滤波，与当前原始三角化结果的距离最多为腕点 2.5 mm、其他关节 3.5 mm；
无效关节仍保持无效。顶部区分 3D 数据更新率和预览刷新上限，并显示延迟
与跳帧数。两帧批处理默认
每批刷新一次预览，约 15 次/秒；这不改变 JSONL 的逐帧更新。
显示滤波不等待未来帧；采集、两帧批处理和模型推理仍会产生延迟。
若优先减少批处理等待，可试 `--frame-batch 1`，但实际吞吐可能低于 30 FPS。
点击窗口后按 `q` 或 `Esc`，或在终端按 `Ctrl+C`，可结束采集并打印统计。
无桌面显示时自动关闭窗口；也可主动加 `--no-preview` 用于无界面运行。
如果已经启动了旧进程，需要先在终端按 `Ctrl+C`，再重新运行命令。

默认输出到 `output/ego_live_wilor/<启动时间>/predictions.jsonl`。每行包含
四路设备时间戳和帧序号、逐相机手框、WiLoR 2D/相对 3D/MANO 参数、
`hands[].joints_3d_base_m`、有效点数、是否调用侧目、是否刷新检测器、
延迟与阶段耗时。无效 3D 为 `null`；手的部分关节点无效时对应坐标也为 `null`。
原始字段保持不变；新增 `camera_hands[].joints_2d_stabilized`、
`camera_hands[].joints_3d_camera_relative_stabilized` 和
`hands[].joints_3d_base_m_stabilized` 以及低延迟的
`hands[].joints_3d_base_m_display`。因果滤波不补填当前无效关节，
观测中断超过 200 ms 后重置；原始字段仍可供下游按需使用。
程序每 10 个已处理帧报告最新帧队列丢帧量。输出中仅写有推理结果的帧，
不会伪造对被丢弃的相机帧的 3D/MANO 更新。

### 接入 Ego Visualizer 的实时 Rerun 界面

先运行 `ego-visualizer live --socket /tmp/ego-visualizer-live.sock`，再为当前入口
增加 `--live-socket /tmp/ego-visualizer-live.sock --no-preview`。显示端接收原始
`cam_0` MJPEG、同一帧的融合双手 3D、Orbbec 加速度/角速度原始采样，
以及后台计算的 cam_0/cam_1 标定双目深度；不重新编码相机图像。可用
`--live-cameras cam_0 cam_1 cam_2 cam_3` 显示全部四目，用
`--no-live-depth` 关闭后台双目深度。深度是 CPU 估计的毫米制轴向深度，
不是硬件深度传感器输出。socket 输出使用独立线程和
单帧队列，查看器断开或刷新跟不上不会阻塞 WiLoR 推理；终端退出统计会
报告实际发送帧和显示侧主动跳过的帧数。Rerun 的 3D 场景使用 cam_0
光学坐标（X 右、Y 下、Z 前）。世界坐标视图只在收到有效 VIO 位姿后显示。
2026-11-06 已接入持续运行的 Basalt VIO 和延迟的 OmniEgoCap 人体预测；
使用下方联合入口启动。相机位姿和地图来自实际双目/IMU 估计，初始化期间
保持空白。人体是滑动窗口推理，目前吞吐明显低于相机采集。

默认每 12 个**已处理帧**重新运行全图检测；其余帧沿用并移动手框。
跟踪帧的框会标记 `proposal_source=tracker`。快速运动或遮挡可能造成短暂
错误，需按任务调整 `--detector-every`；设置为 `1` 可每帧检测。
`--hypotheses dual` 可在左右手分类不可靠时对每个框运行两个 WiLoR 假设，
代价是更高延迟。默认每次批量处理两对双目帧，增加约一帧的等待时间以提高
吞吐；`--frame-batch 1` 降低延迟。`--crop-device cpu` 用于对照 CPU 裁剪路径。
要主动发现仅出现在侧目的手，可设置 `--side-scan-every 30`；默认 `0` 只在主
双目 3D 失败时调用侧目。WiLoR MANO 参数保留在每个相机的输出中。网格顶点
用于实时预览和可视化传输，不写入 JSONL；当前每只手优先使用主相机的一套 MANO
网格，以双目关节配准位置，不对多个相机的 MANO 网格做逐顶点融合。

当前真机实测：四目同步时间戳跨度约 10 µs；未过滤异常大框时，
`detector0.pt` 会把近乎整张左目画面误报为手，因此默认丢弃面积超过
画面 65% 的框。双手持续可见的 240 个已处理帧中，初版每帧全图检测
约 10.5 FPS，其中 234 帧得到两手有效 3D；批量双目三角化和每 6 帧
检测一次后，两手帧的处理时间中位数约 60.5 ms。CUDA 手框裁剪、后台
双目 JPEG 解码及两帧批处理后，在无遮挡区间连续 80 帧达到 30.0 FPS，
两手 3D 均有效。旧版包含 60 帧侧目调用的完整 300 帧测试为 22.5 FPS，
P95 延迟 238 ms。侧目多视角融合优化后的离线重放耗时中位数为 2.48 ms，
重放的 60 帧中有效手数与原算法一致，关节点 3D 中位差 0.8 mm。

最终默认配置的真机 300 帧验收：连续相机帧 34 至 333 全部输出，无跳帧，
相机时间跨度 9.967 秒，更新速率 30.00 Hz；300/300 帧均有双手 3D，
主双目各输出 600 份 MANO 参数。输出延迟 P50/P95 为 76.3/96.7 ms，
融合结果的重投影误差中位数为 1.75 px。该段主双目始终有效，侧目调用
次数为 0。另以 `--side-scan-every 12` 主动扫描侧目时，120 帧测试中
侧目调用 10 次，120/120 帧仍有双手 3D，但相机跨度为 145 帧，跳过
25 帧，实际更新速率 24.79 Hz，延迟 P50/P95 为 80.6/194.1 ms。
穿插实际遮挡并触发侧目后的完整 30 Hz 验收尚未完成。


## 连续 Basalt VIO / WiLoR / 人体联合入口（2026-11-06）

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
bash scripts/run_ego_live_multimodal.sh
```

在有桌面环境的终端运行会打开轻量 Rerun 界面。SSH 无桌面可保存同一显示流：

```bash
bash scripts/run_ego_live_multimodal.sh --headless --max-frames 240 --body-max-windows 1
```

默认输出 `output/ego_live_multimodal/<时间>/`；`--output` 可指定新的目录。
启动采集后先保持相机静止约 2 秒，镜头朝向有纹理的场景，等待 VIO 初始化
后再缓慢移动。避免镜头贴近或被空白桌面占满。
`Ctrl+C` 停止本次启动的进程组。不要同时运行其他 Orbbec 采集程序。
启动器首先读取设备 flash 标定，然后让 WiLoR 单独打开四目相机；原始
cam_0/cam_1 JPEG 和 IMU 通过独立有界 socket 发往 VIO，不二次打开设备。
默认打开左右两栏：左侧为彩色 22 关节人体骨架、跟随手腕的实时手指骨架、VIO 和活动地图，
不显示人体网格及关节名称；右侧为 WiLoR 的 cam_0 原生图像叠加、手指角度和 MANO 网格。
右侧 cam_0 图像默认最多 10 Hz；
相机采集、WiLoR 推理和 Basalt VIO 仍各自以原始速率运行。显示队列只保留最新一帧，
Rerun 界面内存上限默认为 128 MiB 的记录数据，达到上限时丢弃最早历史。
首页不绘制深度和 IMU，默认不运行软件双目深度计算。`--display-width`、
`--display-preview-fps`、`--display-pose-fps`、`--display-map-fps` 可进一步降低显示负载。
`--display-depth-hz 2` 可单独启用后台深度计算；需要同时查看原始多目、深度和 IMU
时，可使用 `ego-visualizer live` 的完整模式。未降采样的推理与 VIO 数值结果仍保存在
`predictions.jsonl` 和 `vio/*.jsonl`，原有离线命令继续可用。

人体默认 1 秒窗口、6 步 DDIM 采样、关闭优化；预测出的窗口以原速度播放，
不再按推理耗时放慢。人体姿态仍比当前相机画面延迟，手指骨架从当前 WiLoR
结果更新，并将手腕接到最近的人体骨架手腕上。`--body-ddim-steps 30` 可恢复
原模型的 30 步采样，`--body-optimization` 恢复较慢的优化；
`--body-window-s`、`--body-stride-s` 可调整窗口和刷新间隔。
`--no-body` 可单独查看手部/VIO，`--max-frames 0` 持续运行。

脚本自动验证/安装锁定 SHA256 的项目内 Basalt headers、Sophus、cereal，
并编译 `ego_live_bridge` 和 `ego_basalt_live`，不替换原 Basalt 动态库。
默认 Python 为 `~/miniforge3/envs/ego-hand/bin/python`，可通过
`EGO_HAND_PYTHON`、`ORBBEC_SDK_ROOT`、`EGO_BUILD_JOBS` 配置。

- `wilor/predictions.jsonl`：真实处理过的手部帧，保留原始设备时间戳。
- `vio/vio.jsonl`：仅相机位姿，供 OmniEgoCap 匹配。
- `vio/world.jsonl`：相机位姿和活动稀疏地图更新，供 Rerun `live-world`。
- `body.jsonl`：22 个米制世界坐标人体关节、源时间戳及窗口计算耗时；实时界面只接收骨架。
- `vio/summary.json`、`run_summary.json`、`logs/`：统计与各子进程日志。
- `live.rrd` 和 `live.summary.json`：仅 `--headless` 模式保存已限频的显示流；用
  `../ego-visualizer/.venv/bin/ego-visualizer view /完整路径/live.rrd` 在桌面回看。

`tools/ego_basalt_live.cpp` 在一次会话中只创建一个光流/VIO 估计器，持续
向 Basalt 的双目和 IMU 队列送数据。`ego_live_basalt_stream.py` 解码/缩放
灰度图，应用 flash IMU 标定一次，使用标定的图像到 IMU 时间偏移一次；
输出时恢复 cam_0 原设备时间戳。启动时固件缓存的过期 IMU 被过滤，等待
当前有效 IMU 覆盖后初始化。严重 IMU 丢失会报错，不伪造轨迹。
旧 `ego_live_basalt.py` 是未验收的重复离线窗口原型，不是联合入口。

世界坐标是米制、右手系、Z 向上，`T_world_cam0=T_world_imu*T_imu_cam0`。
稀疏地图是当前 VIO 活动路标，不是带闭环的全局地图。未提供实测相机到头部
外参时，Rerun 的 head 轨迹代表 cam_0；OmniEgoCap 则沿用初始近似头部外参，
起始相机高度默认按本次坐姿实测约 1.25 m，可通过 `--camera-height` 修改。要评估人体
绝对位置精度，需要测量真实高度和相机到头部外参。

Rerun 按原设备时间戳对齐异步结果，最多匹配相差 50 ms 的手与位姿；
快 VIO/慢神经网络和慢 VIO/快手部结果都保留历史缓冲。默认人体先积累 1 秒窗口
再运行模型，延迟结果回写到采集时刻。不要把时间轴上对齐的结果解释成
零延迟，也不要将 VIO 适配器耗时解释为整条神经网络链路延迟。

2026-11-06 流畅模式真机 450 帧测试：WiLoR 23.10 Hz，显示端发送 410 帧且
没有 IMU 异常、发送超时或断管；人体两次 60 帧推理耗时 10.2/6.6 秒。
追加 1500 帧、65.30 秒测试：WiLoR 22.96 Hz，显示端发送 1388 帧，
查看器 0 次异常，采集端 0 次发送超时/断管；人体 9 次推理，中位数 6.65 秒。
运行约 40 秒时的桌面画面时间轴与采集经过时间同步，双目画面与彩色人体骨架
仍在更新。上述发送帧数不等于桌面实际渲染帧数；人体还需先积累 2 秒窗口，
首个窗口计算约 9.88 秒，不能保证所有场景都满足 5–10 秒延迟。

改版后的 100 秒、3000 帧真机运行中，WiLoR 保持 30.00 Hz，实时合成画面发送
964 帧，VIO 输出 2974 次位姿，人体输出 18 次，查看器无异常。桌面截图中
合成画面帧号 2231 与 Rerun 当前时间约 73.2 秒一致，没有重现录屏中的累计
数十秒落后。该次相机前没有手，MANO 网格显示为不可用；手进入视野后仍需
人工检查双手网格的视觉效果。人体为模型估计，并非人体光学真值。

2026-11-06 低延迟版真机 550 帧测试：默认 1 秒窗口、6 步 DDIM，人体 15 次预测；
首轮之后推理耗时中位数 1.29 秒，预测帧相对最新匹配输入的滞后中位数 1.28 秒，
两膝角中位数约 107°。左侧手指骨架已在桌面画面中确认显示。滞后统计不含 WiLoR
从相机取帧到结果输出的约百毫秒处理时间，也不代表桌面渲染的精确端到端延迟。

联合真机验收统计与限制见
`work/20261106/live_vio_resume/ACCEPTANCE.md`（相对 EGO_glove 项目根目录）。


初始化会积累 2 秒真实 IMU。若其角速度均值/方差和加速度方差满足静止候选
门限，则用平均重力方向初始化姿态，用平均角速度初始化陀螺偏置，并估计
重力方向上可观测的加速度残余偏置。运动起步不满足门限时保留 Basalt 默认
初始化。只设置估计器初始状态，不改写 flash 标定，也不锁死后续相机轨迹。
记录 `vio/backend_events.jsonl.initialization.json` 及 summary 的 `initialization`。
单独 VIO 入口 `--initialization-s` 可改变累积时长（0–5 秒，默认 2 秒）。

弱纹理近距离桌面仍是已知失败场景：同一段真实静止输入，零偏置版本在
约 9.93 秒内漂移 14.01 m；短窗口初始化降到 0.535 m。积累 2 秒后，严格
相同的后续 7.93 秒输入中，基线漂移 9.003 m、修复版漂移 0.232 m。这是
漂移改善，仍不满足高精度静止跟踪；地图点数量和有限刚体矩阵不代表位置
准确。运动场景无外部真值时，也不能由这些统计推导毫米级精度。

启动脚本兼容 `data_sda1/ego_hand_system` 和 `data_sda1/EGO_glove/ego_hand_system`
两种目录布局，会查找相邻目录中的共享 SDK、Ego Visualizer 和 OmniEgoCap。
仍可通过 `ORBBEC_SDK_ROOT`、`--visualizer`、`--omni-root` 显式指定依赖位置。
