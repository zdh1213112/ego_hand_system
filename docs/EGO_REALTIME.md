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

支持的设备是本项目随附 SDK 中的 EGO：

- USB VID/PID：`2bc5:1201`；
- 左右流：`1600x1300 @ 30 FPS MJPG`；
- Linux UVC 后端：`LibUVC`；
- 标定：启动时从设备 Flash 的 `OB_RAW_DATA_ALIGN_CALIB_YAML` 读取；
- 坐标：统一输出为原始左相机光学坐标系下的米制3D。

程序会拒绝其他 VID/PID，避免错误地把普通 Orbbec 深度相机当作 EGO。

启动日志如果显示 `connection=USB2.0`，双路 `1600x1300 MJPG` 的完整帧对可能
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
cd /home/zdh/ego_hand_system
sudo third_party/orbbec_sdk/install_udev_rules.sh
```

安装后重新拔插 EGO。

## 启动

先确认 GPU 环境：

```bash
conda activate ego-hand
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
cd /home/zdh/ego_hand_system
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
