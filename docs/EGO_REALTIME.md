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
cd /home/p3/data_sda1/ego_hand_system
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
窗口下方只显示一个共同 3D 场景：左右手使用 `cam_0` 为基准的米制
坐标、固定 1050 px/m 的斜视正交投影及相同原点，保留实际大小和相对位置；
网格间距为 10 cm。上方重投影、右侧角度、下方场景都读取同一帧的
`hands[].joints_3d_base_m_display` 双目 3D 显示点，不使用手掌归一化。
显示点采用逐帧因果滤波，与当前原始三角化结果的距离最多为腕点 2.5 mm、
其他关节 3.5 mm；无效关节仍保持无效。顶部显示
最近 3 秒的实际输出速率、延迟和该窗口内的跳帧数。两帧批处理默认
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

默认每 12 个**已处理帧**重新运行全图检测；其余帧沿用并移动手框。
跟踪帧的框会标记 `proposal_source=tracker`。快速运动或遮挡可能造成短暂
错误，需按任务调整 `--detector-every`；设置为 `1` 可每帧检测。
`--hypotheses dual` 可在左右手分类不可靠时对每个框运行两个 WiLoR 假设，
代价是更高延迟。默认每次批量处理两对双目帧，增加约一帧的等待时间以提高
吞吐；`--frame-batch 1` 降低延迟。`--crop-device cpu` 用于对照 CPU 裁剪路径。
要主动发现仅出现在侧目的手，可设置 `--side-scan-every 30`；默认 `0` 只在主
双目 3D 失败时调用侧目。WiLoR MANO 参数保留在每个相机的输出中，当前没有
把各相机的 MANO 网格融合为单个世界坐标网格。

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
