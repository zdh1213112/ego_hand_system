# NOKOV/XINGYING 光学动捕手部模型生成 WiLoR 标签

## 结论

可以使用 NOKOV/XINGYING 的光学动捕结果生成 WiLoR 训练标签，但
`Left\\Right Hand(24)` 手部模型本身不是 MANO，也不是 WiLoR 的 `.npy` 标签。
它首先需要导出每帧的 3D marker/手部关节数据，然后完成点位映射、坐标变换、
MANO 拟合和相机投影。

官方参考：
[`Left\\Right Hand(24)`](https://docs.nokov.com/xingying/XingYing4.6-CN/jiu-chuang-jian-markerset/er-body-ren-ti/left-right-hand-24/)

相关导出参考：
[`C3D 文件`](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-san-shu-ju-dao-chu/er-c3d-wen-jian)、
[`TRC 文件`](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-san-shu-ju-dao-chu/wu-trc-wen-jian)、
[`SDK`](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-qi-cha-jian-sdk/shi-san-sdk)。

官方页面明确了以下内容：

- 左手和右手分别使用 24 点模板；反光 marker 贴在手指各关节处。
- 创建模板前需要在实时 3D 视图中确认所有 marker 都被识别，并确认 marker 总数。
- 创建模型时需要冻结一帧。官方还规定了建模姿态和坐标方向：Y 轴向上时手指朝
  `+Z`，Z 轴向上时手指朝 `+Y`。
- 该页面主要描述建模和点位布局，没有在页面正文给出每个点的语义编号。因此必须
  以实际导出的文件为准确认 marker 名称、编号、单位和时间戳。XINGYING 4.6 另有
  C3D、TRC 和 SDK 导出/访问文档，可以作为适配入口。

## 总体架构

```text
XINGYING 24-marker / hand model
        │
        ├─ marker 编号与左右手身份固定
        ├─ 单位、坐标轴、世界坐标 → RGB 相机坐标
        ├─ 24 点 → MediaPipe/WiLoR 21 点（按实际命名点映射，多余点仅作辅助约束）
        ├─ MANO 时序拟合，得到 pose / shape / translation
        ├─ MANO_RIGHT.pkl 生成 778 vertices 和 21 joints
        ├─ K、trans 投影得到 778 个 mesh 像素点和 bbox
        └─ images/*.jpg + labels/*.npy + index.jsonl
```

光学动捕可以作为高质量的 3D 教师标签；它不能代替 RGB 图像。要训练真实图像上的
WiLoR，动捕数据必须和相机 RGB 帧同步。只有动捕轨迹而没有对应图像时，只能生成
合成图像标签，不能称为真实 RGB 训练样本。

## 最关键的难点：EGO RGB 与动捕的时空对齐

EGO 是头戴式相机，不能把“动捕世界坐标 → 相机坐标”的外参当成一个固定矩阵。
必须同时解决：

```text
时间：哪个 XINGYING 帧对应哪个 EGO RGB 帧？
空间：在这一帧，头环相机的姿态在哪里？
```

如果只解决时间、不跟踪头环姿态，头部一转动，3D 手部投影就会整体漂移；如果只
解决空间、不同步时间，快速张合手指时标签也会错位。因此在没有完成下面两项标定
之前，不应批量生成 WiLoR 标签。

### 1. 时间同步

#### 首选：共享硬件触发或 TimeCode

让 EGO 录制和 XINGYING 记录同一个触发信号/时间码，保存每个 RGB 帧的设备时间戳和
每个动捕帧的时间戳。不要只用两个设备的帧号或 nominal FPS 对齐，因为启动延迟和
时钟漂移会随着录制时长累积。

#### 没有硬件同步时：可重复的动作事件

在正式采集开头录制一个两边都能观察到的事件，例如“手掌快速张开—握拳—张开”或
LED 闪烁，并从动捕手部速度峰值与 RGB 中手部运动峰值估计初始时间偏移。之后用
连续轨迹拟合时间关系：

```text
t_mocap = a * t_ego + b
```

其中 `b` 是起始偏移，`a` 表示时钟速率差。若录制很长，必须保留 `a`，不能只估计
一个固定 `b`。每个 EGO 帧应在相邻动捕帧之间插值（位置线性插值，头环旋转用 SLERP），
而不是简单取最近帧。

建议保存如下对齐表：

```text
ego_frame_index,ego_timestamp_ns,mocap_timestamp_ns,
mocap_frame_index,time_error_us,interpolation_alpha
```

本项目已经按硬件时间戳处理 EGO 双目：GEN 数据在标准化后的
`cameras/*/timestamps.csv` 和 `stereo_pairs.csv` 中保存纳秒时间戳，Orbbec 会从
PTS 文件生成同样的时间戳表。因此标签转换应读取这些时间戳，不要使用视频解码后的
顺序帧号。

初始验收可要求时间残差小于约 2--5 ms；最终阈值应以手部运动速度和 RGB 帧率为准，
并用重投影误差再次确认。

### 2. 空间同步：给 EGO 头环建立动态位姿

#### 推荐方案：在头环上创建一个 NOKOV 刚体

在头环/相机支架上固定至少 3 个不共线的反光 marker，并在 XINGYING 中创建独立的
`head_rigidbody`。采集时每个动捕帧得到头环刚体在动捕世界坐标中的位姿：

```text
T_world_head(t)
```

还要标定一次头环坐标到 EGO 每个相机坐标的固定安装外参：

```text
T_head_camera_left
T_head_camera_right
```

于是每个 RGB 帧的相机位姿为：

```text
T_world_camera(t) = T_world_head(t) @ T_head_camera
p_camera(t) = inverse(T_world_camera(t)) @ p_world_mocap(t)
```

这里的矩阵方向必须在代码中明确记录。当前项目的 `T_base_camera` 约定是“相机坐标
转到 base 坐标”；不要把它与 XINGYING 的刚体 pose 方向直接混用。

#### 如何标定 `T_head_camera`

建议使用一个被 EGO RGB 看到、同时在 NOKOV 世界中有 3D 坐标的标定板/标定点：

1. 头环固定或缓慢移动，确保 RGB 相机看到标定板；
2. 记录标定板点在 EGO 图像中的像素位置；
3. 记录同一批点在 NOKOV 世界坐标中的位置，以及同帧 `T_world_head`；
4. 用 PnP/手眼标定求解 `T_head_camera`；
5. 用多组头部姿态做非线性优化，最小化所有帧、所有点的重投影误差。

不能只用头环外壳尺寸或界面坐标轴估计这个外参。左右两个 RGB 相机必须分别标定，
并且使用录制时实际安装的相机支架；重新拆装头环后需要重新标定。

#### 没有头环刚体时的可行性

如果没有 `head_rigidbody`，只有手部在 NOKOV 世界中的 3D 轨迹，则只有以下情况
才可用：

- 头环在整个录制期间完全固定；或
- EGO 已经输出可靠的相机/头部轨迹，并且该轨迹已与 NOKOV 世界完成外参标定。

否则无法把动捕世界坐标正确变换到每一帧的头戴 RGB 相机，最多只能做相对运动演示，
不能生成可信的图像训练标签。

### 3. 用 RGB 重投影做联合校验

完成初始同步和头环外参后，把动捕 3D 手部点投影到 EGO 图像，并与 RGB 中的
MediaPipe/WiLoR 21 点进行对比。可进一步微调时间偏移和外参：

```text
min_{delta,T_head_camera}
    median || project(K, inverse(T_world_head(t+delta) @ T_head_camera)
                    @ p_mocap(t+delta)) - u_rgb(t) ||
```

这里的 RGB 21 点只用于标定和质量检查，不应替换最终的动捕标签。应在不同头部姿态、
不同手势和左右手上都验证，而不是只看一帧叠加效果。

如果使用当前 WiLoR 导出格式，建议统一在**双目校正后的针孔图像**上做这个检查：

- 原始 KB/DS 鱼眼图像：使用原始畸变模型投影；
- 校正图像：使用校正输出的 `P1[:3,:3]` 作为 `K`；
- 不能把针孔 `K` 直接用于原始鱼眼图像，也不能校正图片后仍使用原始鱼眼投影点。

这样最终的 `images/*.jpg`、`K`、`vertices + trans` 和 `joints_2d` 才处于同一个图像
坐标系，能够通过现有的 `check_wilor_training_dataset.py`。

建议的质量门槛：

- 时间偏移/漂移拟合后残差稳定，不能出现随录制时间单调增大的错位；
- 手部关节重投影 median 先控制在 5 px 左右，p95 控制在 10--15 px 左右，再根据
  图像分辨率收紧；
- 头环快速转动时，腕部和指尖不能出现同方向的整体拖影；
- 左右相机分别通过检查后，才进入 WiLoR 标签导出。

## 推荐采集流程

```text
头环加 head_rigidbody
      ↓
EGO 与 XINGYING 同步录制校准动作
      ↓
标定 T_head_camera_left/right
      ↓
估计 t_mocap = a*t_ego+b，并插值动捕轨迹
      ↓
逐帧用动态 T_world_head(t) 转到 EGO 相机坐标
      ↓
RGB 重投影检查和剔除异常帧
      ↓
MANO 拟合 → WiLoR labels/*.npy
```

一次正式采集建议包含三段：

1. 5--10 秒静止/张手，用于确认外参、单位、左右手和 marker 映射；
2. 5--10 秒大幅度转头但手尽量保持，用于验证动态头环位姿；
3. 正式手势动作，用于验证时间同步、快速指尖运动和遮挡恢复。

只有第一段对齐而第二段失败，说明缺少动态头环跟踪；只有慢动作对齐而快速动作失败，
通常说明时间偏移或时钟漂移没有处理。

## 24 点如何对应 21 点

本项目的 21 点顺序与 MediaPipe 一致：

```text
0 wrist
1-4   thumb: CMC/MCP/IP/tip
5-8   index: MCP/PIP/DIP/tip
9-12  middle: MCP/PIP/DIP/tip
13-16 ring: MCP/PIP/DIP/tip
17-20 pinky: MCP/PIP/DIP/tip
```

官方示意图显示每只手有 24 个 marker，但页面正文没有给出每个 marker 的名称和
MediaPipe 语义对应关系。因此不能假设“前 21 个点就是 WiLoR 的 21 点”，也不能仅
凭图片推断哪个点是 wrist。实际转换应按 XINGYING 导出的命名点建立映射：

1. 用静态张手帧确认 24 个点的名称/编号和左右手规则；
2. 将指根、指间、指尖和腕部中心映射到 MediaPipe 的 21 点；
3. 无法一一对应的掌部点作为额外拟合约束或质量检查，不强行塞入 21 点；
4. 如果 marker 贴在皮肤表面而不是关节中心，需要估计固定的 marker 偏移，不能把
   marker 坐标直接当成关节中心。

最终映射不能只凭截图确定，必须拿到一小段带 marker 名称的实际导出数据。建议先做
一帧张手静态姿态和一段连续动作，检查 24 个点是否在每一帧保持同一语义。

## 推荐的 XINGYING 数据入口

按离线建库的可靠性排序：

1. **TRC（首选）**：官方说明 TRC 包含已命名点和未命名点的逐帧 `XYZ`，并带帧、时间
   和时间戳，最适合写一个确定性的 Python 转换器。需要保留每个点的名称和原始帧号。
2. **C3D（推荐备选）**：官方说明 C3D 导出会包含 Markerset 的命名点，适合保留原始
   marker 轨迹、帧率和元数据；可在 Python 侧解析后转成统一的 NPZ 中间格式。
3. **SDK（实时采集）**：官方 SDK 支持 Windows、Linux、Python 等入口；适合与 RGB
   相机在线同步或直接接收命名点，但第一版离线建库建议先保存成文件再转换，便于复现。
4. **BVH/FBX**：更偏向人体骨骼/动画重定向，可能丢失 24 个原始 marker 的逐点信息，
   不作为本项目第一选择。

TRC 中如果出现 `RCalc_标记点名称`，表示 XINGYING 用反算点补齐了丢失的原始 marker；
这类点应标记为 `inferred`、降低 confidence，不能与同时刻真实可见 marker 等权。

## 两条转换路线

### 路线 A：XINGYING 已导出手部关节变换或 MANO 参数

这是优先路线。若 SDK/插件可以直接导出每帧关节旋转、腕部平移、手型参数，或者直接
导出 MANO 参数，则只需要：

1. 将旋转统一为 WiLoR 需要的完整旋转矩阵；
2. 将平移转换到目标 RGB 相机坐标系；
3. 使用项目中相同的 `MANO_RIGHT.pkl` 重放 778 个顶点和 21 个关节；
4. 进行投影、bbox 计算和标签写出。

这条路线避免了从 24 个表面 marker 反推关节中心，误差最小。

### 路线 B：只有 24 个 marker 的 3D 坐标

需要先把 24 点整理为与本项目一致的 21 点观测，再复用
`scripts/fit_mano_sequence.py` 的时序 MANO 拟合。输入至少要包含：

- `positions`: `(hand, frame, 21, 3)`，单位为米；
- `valid`、`observed`、`confidence`；
- 左右手身份和每帧时间戳；
- 相机同步后的 2D 点（如果有 RGB 图像，建议同时提供）；
- 手部骨长统计和相机内参。

仅使用 3D marker 可以拟合出 MANO，但 shape 和部分关节旋转约束会弱于“3D + RGB
投影”联合拟合。因此应保留 marker 残差和 MANO 拟合残差，超阈值帧不进入训练集。

## 坐标系和左右手规范

这是最容易造成标签整体错误的部分，必须单独做标定。

### 世界坐标到 RGB 相机坐标

XINGYING 输出通常是动捕世界坐标，WiLoR 标签需要相机坐标。使用：

```text
p_camera = R_camera_from_mocap @ p_mocap + t_camera_from_mocap
```

其中 `R`、`t` 应通过同一场景中的标定物或同步点云求得，不能只根据软件界面上的
坐标轴猜测。还要确认单位是毫米还是米；本项目内部统一使用米。

### 右手 canonical 空间

本项目使用 `wilor_right_canonical_v1`，左右物理手都由 `MANO_RIGHT.pkl` 解码：

- 物理右手：直接进入 MANO；
- 物理左手：在转换到相机坐标后反射 X 轴，图像水平翻转一次；
- 左手的 `K` 也必须同步变换；训练读取器不能根据 `side=0` 再翻转一次。

现有实现位于 [`scripts/mano_conventions.py`](../scripts/mano_conventions.py)，包括
`mirror_left_points`、`horizontally_flipped_intrinsics` 和左右手几何恢复函数。

注意：先完成“动捕世界坐标 → RGB 相机坐标”，再做 WiLoR 左手 canonical 化；不要
直接对 XINGYING 世界坐标盲目取负号。

## WiLoR 标签的固定格式

本项目的导出器和校验器要求每个 `labels/xxxxxx.npy` 的字典键顺序严格为：

```text
("bbox", "vertices", "joints_3d", "joints_2d", "side", "trans", "K", "mano")
```

| 字段 | 类型和形状 | 含义 |
|---|---|---|
| `bbox` | `float64 (4,)` | `[x1,y1,x2,y2]`，对应 RGB 图像 |
| `vertices` | `float32 (778,3)` | MANO 局部顶点，平移单独放在 `trans` |
| `joints_3d` | `float32 (21,3)` | MANO 21 点局部坐标 |
| `joints_2d` | `torch.float32 (778,2)` | 778 个 mesh 顶点的图像投影 |
| `side` | `float32` 标量 | 物理身份：左手 `0`，右手 `1` |
| `trans` | `float32 (3,)` | 相机坐标系平移 |
| `K` | `float32 (3,3)` | 与配对图像完全一致的内参 |
| `mano.global_orient` | `float32 (1,3,3)` | 完整 MANO 根关节旋转矩阵 |
| `mano.hand_pose` | `float32 (15,3,3)` | 15 个手部关节旋转矩阵 |
| `mano.betas` | `float32 (10,)` | MANO shape 参数 |

投影必须满足：

```text
P_camera = vertices + trans
P_image  = K @ P_camera
joints_2d = P_image.xy / P_image.z
```

因此光学动捕只提供 3D 点时，`vertices` 不能直接由 21 点“补齐”；必须通过同一份
MANO 模型生成 778 个顶点，并用 MANO 参数重放校验。

## 推荐的代码落地方式

当前项目的 `export_multiview_wilor_training_dataset.py` 绑定了 GEN 多视角
`fusion/normalized_multiview` 输入，不能直接读取 XINGYING 文件。建议新增一条独立
适配链路，同时复用已有 MANO 和校验逻辑：

```text
inspect_nokov_hand_export.py
    检查 marker 名称、数量、单位、时间戳和左右手

convert_nokov_markers_to_mano_input.py
    24→21 映射、坐标变换、时间同步、质量筛选，输出 mano_input.npz

fit_mano_sequence.py
    3D/2D + 时序约束拟合 MANO_RIGHT

export_nokov_wilor_training_dataset.py
    读取 RGB、K、相机位姿和 MANO 结果，输出 images + labels

check_wilor_training_dataset.py
    检查 schema、投影误差、bbox、MANO_RIGHT 重放误差
```

第一版适配器建议以 TRC/CSV 规范化为输入，字段如下：

```text
frame,timestamp_ns,hand,marker_id,marker_name,x,y,z,visible,residual
```

若从 C3D 读取，则先转换成同样的中间表，不让 C3D/TRC 解析逻辑渗透到 MANO 和
WiLoR 导出阶段。

以及：

- XINGYING 导出格式（CSV/C3D/BVH/JSON/SDK 回调中的一种）；
- marker 单位和坐标轴方向；
- 左右手的 marker ID/name 是否固定；
- RGB 相机视频、分辨率、内参 `K`；
- 动捕坐标到 RGB 相机的外参 `R,t`；
- 两套系统的时间戳或固定延迟。

没有这些信息时，可以展示 3D 骨架，但不能生成可验证的 WiLoR 训练标签。

## 质量门槛和验收

每个样本进入训练集前建议检查：

- 24 点数量、marker 语义和左右手身份没有跳变；
- MANO 与 marker 的 3D 残差、骨长变化在阈值内；
- MANO 顶点全部位于相机前方，且至少规定比例投影到图像内；
- 778 点投影与 `joints_2d` 的最大误差小于 `1e-3` 像素；
- 使用同一 `MANO_RIGHT.pkl` 重放的 778 顶点和 21 关节误差小于 `1e-5 m`；
- `images/*.jpg` 与 `labels/*.npy` 一一配对，并按人物/动作/采集 session 划分训练集、
  验证集，避免相邻帧泄漏。

最终应运行：

```bash
PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/check_wilor_training_dataset.py <dataset> \
  --reference /home/zdh/tool/npy_decoder/000865.npy \
  --mano-source third_party/MANO \
  --mano-model-dir models/mano
```

## 当前结论和下一步

技术路线可行，而且比单纯用 MediaPipe/WiLoR 预测结果更适合作为高质量教师标签；
但当前仓库还缺少 XINGYING 导出格式的适配器，不能仅凭截图或官方模板图直接生成
标签。下一步应先导出一小段实际数据（建议 100 帧，含左右手和一帧静态张手），
放入项目后再确定 24→21 映射和坐标外参，随后实现转换脚本并跑完整的标签校验。
