# GEN 六目 WiLoR 手部姿态融合：设计、改进与验证记录

更新日期：2026-08-19

## 1. 文档目的

本文记录 `/home/zdh/ego_hand_system` 从 GEN `camera2+camera3` 双目 WiLoR 扩展到
`camera0..camera5` 六目手部姿态识别的完整过程，包括遇到的问题、算法改进、最终参数、
真实数据指标和运行方法。

当前目标是提高双手 21 点三维姿态识别的覆盖率、稳定性和身份一致性。六目系统不会修改
WiLoR 网络权重，而是在 WiLoR 逐相机预测之后进行跨相机关联、三角化和质量控制；第
20 节进一步说明如何把已经验收的六目结果导出为 WiLoR 图片/NPY 训练样本。

## 2. 最终结论

六目方案已经产生可量化收益。在 1104 帧完整序列上，严格左右手身份版本达到：

| 指标 | 最终结果 |
|---|---:|
| 处理帧数 | 1104 |
| 接受帧数 | 1053 |
| 接受率 | 95.38% |
| 输出手数量 | 2106 |
| 获得多目增强的手 | 2103 / 2106（99.86%） |
| 选中视角数中位数 | 6 |
| 有效关节内点视角数中位数 | 6 |
| 六目重投影误差中位数 | 3.51 px |
| 两目锚点跨视角误差中位数 | 4.13 px |
| 六目跨视角误差中位数 | 3.52 px |
| 相对两目锚点改善 | 14.75% |
| 两目与六目 3D 差异中位数 | 19.73 mm |
| 最终身份与 detector 不一致观测数 | 0 |

六个相机在最终接受结果中的实际手观测贡献次数为：

| 相机 | 贡献手观测数 |
|---|---:|
| camera0 | 2091 |
| camera1 | 2086 |
| camera2 | 2103 |
| camera3 | 2105 |
| camera4 | 2102 |
| camera5 | 2095 |

这说明最终结果不是“仍然只用 camera2/3”：外围四路已经稳定参与六目约束。并不是每个
关节都必须得到六路支持；RANSAC 会逐关节排除遮挡、误检或不一致视角。

## 3. GEN 六目输入

完整测试使用：

```text
recordings/20260818/DAS-Ego_20260818164752_none_none_689985_b5adb46c.mcap
```

六路输入均为 1600×1300 H264，MCAP 内包含每路 `/compressed` 和 `/camera_info`：

```text
/robot0/sensor/camera0/compressed
...
/robot0/sensor/camera5/compressed

/robot0/sensor/camera0/camera_info
...
/robot0/sensor/camera5/camera_info
```

每个 `camera_info` 提供：

- Double Sphere（DS）内参；
- 图像尺寸；
- 相机到 GEN base/rig 坐标系的 `T_base_camera`；
- 六目统一外参。

完整序列同步统计：

| 指标 | 数值 |
|---|---:|
| 每路有效解码帧 | 1104 |
| 参考相机 | camera2 |
| 时间差绝对值中位数 | 46 µs |
| 时间差绝对值 p95 | 80 µs |
| 最大时间差 | 116 µs |
| 配对上限 | 1500 µs |

同步误差远小于一帧周期，因此六目时间对应关系满足手部三角化要求。

## 4. 最终处理链路

```text
GEN MCAP: camera0..camera5
              |
              v
   六路 H264 无重编码 remux
              |
              v
以 camera2 为参考的微秒级时间同步
              |
              v
六路 detector.pt + WiLoR 双姿态假设
              |
              v
detector.pt 严格锁定 Left / Right 身份
              |
              v
camera2/3 首选锚点；失败时动态选择备用锚点对
              |
              v
外围相机逐只手进行 DS 重投影关联
              |
              v
原生 Double-Sphere 射线 + 多视角 RANSAC
              |
              v
短缺口：邻近已确认姿态引导当前帧六路重新关联
              |
              v
GEN base 坐标系 21×3 三维关节 + 六目诊断视频
```

## 5. 关键代码

| 文件 | 作用 |
|---|---|
| `scripts/normalize_multiview_recording.py` | 六路视频 remux、标定保存和时间同步 |
| `scripts/wilor_multiview_inference.py` | 模型只加载一次，依次运行六路 detector 和 WiLoR 双假设 |
| `scripts/camera_models/double_sphere.py` | GEN DS 投影和与项目参数约定一致的解析反投影 |
| `scripts/fuse_multiview_wilor.py` | 通用原生 DS 射线三角化、RANSAC 和序列化基础函数 |
| `scripts/fuse_multiview_wilor_guided.py` | detector 身份锁定、锚点引导、动态锚点、逐手关联和时序恢复 |
| `scripts/render_multiview_wilor.py` | 3×2 六目诊断视频、内点/离群/未参与状态显示 |
| `scripts/run_multiview_wilor_experiment.sh` | 一键运行、完成标记、配置指纹和断点复用 |
| `tests/test_double_sphere.py` | DS 投影/反投影往返和双目几何测试 |
| `tests/test_multiview_wilor.py` | 六目同步、三角化和 detector 引导关联测试 |

需要直接阅读核心算法时，可按以下函数定位：

| 核心函数 | 算法职责 |
|---|---|
| `double_sphere.project()` / `unproject()` | DS 正向投影和解析反投影 |
| `fuse_multiview_wilor._ray_in_base()` | 像素射线转换到 GEN base 坐标系 |
| `fuse_multiview_wilor._intersect_rays()` | 多射线最小二乘交会 |
| `fuse_multiview_wilor.triangulate_ransac()` | 两射线枚举、内点选择和全内点重拟合 |
| `fuse_multiview_wilor._triangulate_hand()` | 21 个关节逐点融合和手级质量统计 |
| `fuse_multiview_wilor_guided._ordered_hand_pairs()` | 锚点左右手排列及 detector 身份过滤 |
| `fuse_multiview_wilor_guided._candidate_error()` | 3D 初值回投到单目候选的关联误差 |
| `fuse_multiview_wilor_guided._match_camera()` | 外围相机左右手联合匹配 |
| `fuse_multiview_wilor_guided._evaluate_anchor_assignment()` | 锚点方案评分、外围补充和最终重建 |
| `fuse_multiview_wilor_guided._recover_from_reference()` | 短缺口的当前帧观测恢复 |

## 6. Double Sphere 几何修正

GEN 标定保存 `[fx, fy, cx, cy, xi, alpha]`。项目原投影约定为：

```text
d1  = ||p||
z1  = alpha * d1 + (1 - alpha) * z
d2  = sqrt(x² + y² + z1²)
u_n = x / (xi * d2 + z1)
v_n = y / (xi * d2 + z1)
u   = fx * u_n + cx
v   = fy * v_n + cy
```

这里的系数使用方式与常见论文实现的参数排列不完全相同。第一版直接套用标准 DS 逆公式，
单位射线方向出现约 `2.9e-4` 的误差；宽基线多目会把该角度误差放大为明显 3D 偏差。

最终实现根据上述实际投影方程重新推导解析逆解。投影→反投影单元测试目前在浮点精度范围
内通过。六目融合始终在原始 DS 像素上建立射线，不把原始鱼眼像素错误地当作针孔像素。

每个像素首先转换为相机坐标系单位射线，再通过 `T_base_camera` 旋转到 GEN base 坐标系；
相机中心直接取 `T_base_camera[:3, 3]`。

## 7. 六路 WiLoR 推理

每路先运行 `detector.pt` 得到物理手框。为避免 WiLoR 的单目 handedness 假设影响姿态，
每个物理框仍计算 Left 和 Right 两套 WiLoR 姿态，但同时保留 detector 原始类别：

```json
{
  "detection_index": 0,
  "detector_is_right": 1,
  "is_right": 0,
  "joints_2d": []
}
```

其中：

- `detector_is_right=0`：detector 判定为左手；
- `detector_is_right=1`：detector 判定为右手；
- `is_right`：当前 WiLoR 姿态假设，不再拥有最终身份决定权。

模型和 detector 只加载一次，六个视频顺序推理，避免 8 GB 显存同时保存六份模型。

侧视角黑手套置信度偏低，因此使用逐相机阈值：

| 相机 | detector 阈值 |
|---|---:|
| camera0 | 0.2 |
| camera1 | 0.3 |
| camera2 | 0.3 |
| camera3 | 0.3 |
| camera4 | 0.1 |
| camera5 | 0.1 |

低阈值只负责提高候选召回率。假框不能仅凭低置信度直接进入最终结果，必须继续通过身份、
跨视角关联和 RANSAC。

## 8. 从第一版融合到最终方案

### 8.1 第一版：所有相机同时枚举左右手

第一版要求一个相机必须同时检测到两只手才可参与，并枚举每路两个检测框的交换组合。

主要问题：

- 外围相机只看到一只手时完全失去贡献；
- 低阈值带来的背景假框会占据“第二只手”；
- 六路组合数随可用相机增加而增大；
- 几何误差可能在手掌翻转时选择全局交换的左右手方案；
- 60 帧测试仅接受 13 帧，覆盖率 21.7%。

### 8.2 锚点引导和逐只手关联

最终空间关联先用稳定的 `camera2/3` 建立两只手的三维初值，再把三维关节投到外围相机。
每个外围相机、每只手独立寻找误差最小的候选：

```text
candidate_cost = median(||project(X_base) - observed_joints_2d||)
```

候选误差超过 55 px 时不加入。左右手候选不能使用同一个物理检测框。

该策略允许：

- camera0 只贡献左手；
- camera5 只贡献右手；
- camera1/4 漏掉另一只手时仍然保留可见手；
- 背景框在进入最终三角化之前就被拒绝。

### 8.3 动态锚点

首选锚点为 `camera2+camera3`。当它不可用或不能形成有效三维初值时，只搜索至少包含
`camera2` 或 `camera3` 的备用相机对，例如：

```text
camera1+camera2
camera2+camera4
camera0+camera3
camera3+camera5
```

为减少计算量，首选锚点产生质量合格结果后立即采用；只有失败时才枚举备用锚点。

### 8.4 多视角 RANSAC

每个语义关节独立处理：

1. 将各相机 2D 关节反投影为 GEN base 坐标系射线；
2. 枚举两条射线产生候选三维点；
3. 将候选点投回所有可用 DS 相机；
4. 重投影误差不超过 20 px 的视角作为内点；
5. 使用全部内点射线做最小二乘交会；
6. 至少 12 个关节通过质量门控，手姿态才可输出。

最终质量门限：

| 参数 | 默认值 |
|---|---:|
| 初始锚点阈值 | 60 px |
| 外围关联阈值 | 55 px |
| RANSAC 内点阈值 | 20 px |
| 最大重投影中位数 | 15 px |
| 最大重投影 p95 | 40 px |
| 最少有效关节 | 12 |

初始锚点允许更宽松的 60 px，只用于产生候选；最终输出仍执行 20/15/40 px 的严格质量
控制。放宽初值不会直接放宽最终结果。

### 8.5 短缺口时序恢复

空间融合失败但距离最近已确认帧不超过 3 帧时，使用邻近已确认三维姿态投影到当前六路，
重新寻找当前帧的 detector/WiLoR 候选。

时序恢复不是关节插值。恢复帧仍然必须满足：

- 使用当前帧真实图像产生的 WiLoR 2D 关节；
- 至少两个相机提供观测；
- 重新执行 DS 三角化和 RANSAC；
- 最终质量门限不变；
- 手腕位移不超过与帧间隔成比例的上限。

因此时序只提供“去哪里找”的先验，不替代当前帧识别。

### 8.6 DS 像素反投影的实际解析式

多目三角化的基础不是针孔相机的 `K^-1 [u,v,1]`，而是 GEN 标定文件中定义的 Double
Sphere（DS）模型。设归一化像素：

```text
mx = (u - cx) / fx
my = (v - cy) / fy
r2 = mx^2 + my^2
```

当前代码使用与本项目实际正向投影参数排列严格互逆的解析式：

```text
q     = (1 - xi * sqrt(1 + (1 - xi^2) * r2)) / (1 - xi^2)
scale = (alpha*q + (1-alpha)*sqrt(q^2 + (1-2*alpha)*r2))
        / (q^2 + (1-alpha)^2*r2)

ray_camera = normalize([
    scale * mx,
    scale * my,
    (q * scale - alpha) / (1 - alpha)
])
```

根号内部、分母和有效视场都会检查；无效像素不会产生射线。相机到 base 的外参记为
`T_base_camera = [R_bc, t_bc]`，则：

```text
camera_center_base = t_bc
ray_direction_base = R_bc @ ray_camera
```

相反，把 base 坐标中的点投回该相机时：

```text
X_camera = R_bc.T @ (X_base - t_bc)
```

然后再执行 DS 正向投影。这里必须成对使用同一套正反模型，否则单相机看似只有很小的
角度误差，经过六目宽基线交会后会变成明显的三维偏移。

对应实现：`scripts/camera_models/double_sphere.py`。

### 8.7 多射线最小二乘交会

一条单位射线由相机中心 `c_i` 和方向 `d_i` 表示。三维点 `X` 到射线的垂直残差为：

```text
r_i = (I - d_i d_i^T) (X - c_i)
```

因此多射线交会求解：

```text
X* = argmin_X sum_i ||(I - d_i d_i^T)(X - c_i)||^2
```

令 `P_i = I - d_i d_i^T`，法方程为：

```text
A = sum_i P_i
b = sum_i P_i c_i
A X = b
```

实现会检查 `cond(A)`；条件数大于 `1e8` 时认为射线近乎平行或几何退化，不输出该点。
得到 `X` 后还会验证点位于相机射线前方，并通过真实 DS 模型回投计算像素误差。

### 8.8 逐关节 RANSAC 的完整过程

系统不会一次性假设某一路相机整只手都正确，而是对 21 个语义关节分别做 RANSAC。这样
某相机的食指被遮挡时，它的手腕、拇指等正确关节仍然可以贡献结果。

对某个关节的算法为：

```text
best = None
for 每一对可用射线 (i, j):
    X_candidate = intersect(rayi, rayj)
    errors = 把 X_candidate 用各相机 DS 模型投回后的像素误差
    inliers = {k | errors[k] <= 20 px}

    优先选择内点数最多的候选；
    内点数相同时，选择内点重投影中位数更小的候选。

X_final = 使用 best.inliers 的全部射线重新做最小二乘交会
再次投影并生成最终内点集合
```

至少两路内点才能产生一个三维关节。一只手至少有 12 个有效三维关节，随后才检查该手
所有内点的重投影中位数和 p95。这里有一个重要改进：

- `median/p95` 质量门控只统计 RANSAC 内点；
- 被 RANSAC 拒绝的外围视角保留在 `all_view_*` 诊断字段中；
- 离群相机不会因为仍被计入 p95 而把一个本来正确的三维解再次拒绝。

早期实现把离群视角也混入最终 p95，导致 RANSAC 虽然正确找到了内点，整手仍会被离群值
否决。分离“输出质量”和“全视角诊断”后，RANSAC 才真正发挥作用。

### 8.9 锚点左右手排列与评分函数

锚点相机中，每个物理检测框都有两套 WiLoR 姿态假设。首先只保留同时具有 Left/Right
两套假设的完整物理检测，并按 detector 置信度排序，最多保留 3 个框。

严格身份模式下，锚点排列满足：

```text
Left 位置的物理框：detector_is_right 必须为 0（或旧格式缺失值 None）
Right 位置的物理框：detector_is_right 必须为 1（或旧格式缺失值 None）
Left 和 Right 不得是同一个物理 detection_index
```

当前六目推理文件总会写出 0/1；允许 `None` 只是为了读取早期预测格式。若输入真的缺少该
字段，程序仍能运行，但不能声称该观测经过 detector 身份验证。

对锚点排列先用 60 px 阈值产生左右手三维初值，再关联其余四路，最后用 20 px RANSAC
重建。一个完整排列的代价不是单一重投影误差，而是：

```text
assignment_cost =
      100 * missing_joint_count
    + sum_over_hands(median_reproj_px + 0.2 * p95_reproj_px)
    + sum_over_hands(anatomy_penalty)
    - 1.5 * extra_joint_support
    - 4.0 * matched_side_views
    + fallback_anchor_penalty
```

其中：

```text
extra_joint_support = sum(max(joint_inlier_view_count - 2, 0))
matched_side_views  = sum(number_of_selected_views_for_hand - 2)
```

缺失关节受到强惩罚，多相机一致支持则降低代价。备用锚点每缺少一个首选相机再增加 `5`
的轻微惩罚，使系统在质量相近时优先选择稳定的 `camera2/3`，但不会为了坚持首选锚点而
接受一个明显更差的几何解。

解剖惩罚使用手部 20 条标准骨边。有效边少于 12 条时直接罚 `500`；边长低于 6 mm 或
高于 90 mm 的部分按以下方式惩罚：

```text
anatomy_penalty =
    1000 * sum(max(0.006 - edge_length_m, 0))
  + 1000 * sum(max(edge_length_m - 0.090, 0))
```

它不是用固定手型约束真实动作，而是排除明显由错框、错关节对应产生的折叠或超长骨架。

### 8.10 外围相机的左右手联合匹配

三维锚点姿态投到一个外围相机后，每个候选框的几何误差定义为至少 12 个有效关节的
二维欧氏误差中位数：

```text
E(hand, detection) = median_j ||project_DS(X_j) - y_detection,j||_2
```

超过 55 px 的候选直接无效。系统随后联合枚举该相机的 `(left_candidate,
right_candidate)`，候选也可以是 `None`，并最小化：

```text
camera_match_cost = E_left + E_right
                    - 0.25 * 55 * matched_hand_count
```

其中 `None` 的代价为 55。每匹配到一只几何一致的手会获得支持奖励，但同一
`detection_index` 绝不允许同时供给左右手。严格模式还要求候选 detector 类别与目标手
一致。这一联合选择比“左右手各自独立取最小值”多解决了一个关键冲突：两只手相近时，
不能让它们同时抢到同一个高置信度物理框。

### 8.11 动态锚点的实际搜索顺序

动态锚点并不是任意六路两两盲搜。实际伪代码如下：

```text
preferred = (camera2, camera3)
candidate = evaluate(preferred)

if candidate 通过最终质量门限:
    直接采用 candidate
else:
    for pair in 其余相机对:
        要求 pair 至少包含 camera2 或 camera3
        要求两路都至少有两个完整物理手框
        evaluate(pair)
    从通过质量门限的结果中选 assignment_cost 最小者
```

这个限制来自设备布局：`camera2/3` 是主要手部视角，外围相机更适合补充遮挡而不是单独
承担身份初始化。首选结果合格即停止也显著减少候选排列和重复三角化；60 帧版本优化后
约 27 秒完成融合，避免每帧把所有锚点组合全部算完。

### 8.12 时序恢复的实际流程

时序恢复只处理空间融合拒绝的帧，且只参考距离不超过 3 帧的最近主接受帧：

```text
reference = nearest_primary_accepted_frame(current, max_gap=3)
seed_3d   = reference 的左右手三维关节

for camera in 六路相机:
    把 seed_3d 投到 current frame
    在当前帧候选中按 strict detector 身份匹配
    匹配阈值放宽到 90 px

要求每只手至少由两个当前相机匹配
使用当前帧 2D 观测重新执行 20 px RANSAC 三角化
执行同样的 12 joints / 15 px median / 40 px p95 门限
检查 wrist displacement <= 0.12 m * frame_gap
```

90 px 只扩大当前帧的候选搜索区域；它不属于最终三角化内点阈值。恢复输出中的三维点
完全由当前帧观测重建，不复制参考帧，也不做线性插值。

### 8.13 身份约束是贯穿算法的不变量

最终实现把下面的不变量应用到所有入口，而不是最后给结果改一个名字：

```text
对于带有 detector 类别的最终 Left 观测：detector_is_right == 0
对于带有 detector 类别的最终 Right 观测：detector_is_right == 1
```

约束位置包括：

1. 首选锚点左右手排列；
2. 备用锚点左右手排列；
3. 四路外围相机候选匹配；
4. 时序恢复的当前帧候选匹配；
5. 输出审计和 summary 计数。

这也是修复手掌翻转交换问题的核心：WiLoR 的双假设仍用于比较同一个物理框在两种 MANO
手型下的姿态质量，但它不能把 detector-right 的物理框变成最终 Left。

### 8.14 算法层级与各自权限

最终系统把原先混在一起的三个问题拆开：

| 层级 | 输入 | 决定内容 | 无权决定的内容 |
|---|---|---|---|
| detector | 单目图像 | 物理手框、Left/Right 身份、置信度 | 最终 3D 位置 |
| WiLoR 双假设 | 每个物理手框 | 21 点/手网格的单目姿态候选 | 修改 detector 身份 |
| 六目几何 | 标定、各路 2D 候选 | 跨视角关联、离群点、3D 关节、质量 | 跨类别交换左右手 |
| 时序恢复 | 邻近可靠 3D + 当前帧候选 | 当前帧搜索先验 | 复制或插值出无图像证据的姿态 |

这种权限划分比单纯调阈值更重要。它保证某一层的不确定性不会越权污染另一个层面的语义。

## 9. 左右手交换问题与最终修复

### 9.1 问题表现

旧融合在手掌翻转时会出现：

- 黄色右手变成蓝绿色左手；
- 蓝绿色左手变成黄色右手；
- 六个相机经常整段一起交换；
- detector 框本身的 Left/Right 类别仍然正确。

### 9.2 根因

每个检测框都有两套 WiLoR handedness 假设，但旧融合只按几何代价选择物理框属于哪只手，
没有使用 `detector_is_right`。手掌翻转、两手相近或两套几何代价接近时，全局交换仍可得到
相似重投影误差，因此几何优化错误地获得了修改身份的权限。

旧完整结果审计发现：

- 1063 个接受帧中有 572 帧出现最终身份与 detector 相反；
- 有 6706 个实际参与融合的相机手观测与 detector 类别不一致。

### 9.3 修复原则

```text
detector.pt 决定“这是谁”
WiLoR 决定“这只手是什么姿态”
六目几何决定“哪些观测可信、三维点在哪里”
```

严格模式规则：

- detector-left 只能进入最终 Left；
- detector-right 只能进入最终 Right；
- 锚点、外围相机和时序恢复全部执行相同约束；
- 几何误差不再允许跨类别交换；
- 身份与几何冲突时拒绝该帧，不输出交换身份的结果。

命令行参数：

```text
--detector-handedness strict
```

该参数已经成为默认值，也由一键脚本显式传入。完整 1104 帧修复后，所有有效观测中的
detector/fusion 身份不一致数为 `0`。

本次 `fusion_handedness_strict_full` 目录恰好生成于 summary 新增身份计数字段之前，因此
该目录的 `summary.json` 中还没有这一行；随后已使用 `accepted.jsonl` 内每个视角保存的
`detector_is_right` 对全部 1053 个接受帧独立审计，结果为 `0`。以后由当前代码生成的
summary 会直接写入该字段。

## 10. 改进过程指标

### 10.1 算法改进记录

下面按实际开发顺序记录“现象—根因—修改”，便于以后回归时判断某段代码为什么存在：

| 阶段 | 发现的问题 | 核心修改 | 结果/意义 |
|---|---|---|---|
| 双目基线 | `camera2/3` 稳定，但遮挡时缺少补充证据 | 保留为首选锚点和对照基线 | 提供稳定身份初始化与量化参照 |
| 六目直接全局枚举 | 外围单手视角不能参与，组合爆炸，60 帧只接受 13 帧 | 改为锚点初始化、外围逐相机补充 | 搜索空间从全局组合拆成局部匹配 |
| DS 几何校正 | 套用参数排列不同的通用 DS 逆式，射线有系统角度误差 | 从本项目正向投影重新推导解析逆式，并做往返测试 | 消除宽基线下被放大的系统 3D 偏差 |
| RANSAC 质量统计 | 离群视角已被 RANSAC 排除，却仍进入 p95 导致整手拒绝 | 输出门限只统计内点，全视角误差单独诊断 | 让离群剔除真正影响最终判定 |
| 外围召回 | camera4/5 黑手套框置信度较低 | detector 阈值降到 0.1，但仍需身份、几何和 RANSAC 三重验证 | 提高召回而不让低置信假框直接成为结果 |
| 单手外围视角 | 某路只看见一只手时整路被丢弃 | 左右手可分别匹配 `None`，但联合禁止共用物理框 | camera0/1/4/5 可只贡献当前可见手 |
| 锚点引导 | 背景假框、双手近距离时容易错关联 | 用锚点 3D 回投后的关节中位误差关联候选 | 关联依据从框顺序变成跨视角几何一致性 |
| 动态锚点 | `camera2/3` 偶发漏框会导致整帧失败 | 失败时搜索至少包含 camera2 或 camera3 的备用锚点 | 60 帧覆盖从锚点引导阶段继续提高到 38/60 |
| 短缺口恢复 | 少数连续动作帧因瞬时漏检形成空洞 | 最近可靠 3D 只作搜索先验，当前帧重新三角化 | 开发序列达到 57/60，且不是插值伪标签 |
| 首选锚点短路 | 每帧穷举全部可用锚点计算慢 | `camera2/3` 通过最终质量即停止，失败才回退 | 60 帧融合约 27 秒 |
| 可视化语义 | 所有相机都显示框，无法区分是否真正参与 | 增加 `USED/INACTIVE/OUTLIER/REJECTED` | 可以直接检查每路对最终 3D 的实际贡献 |
| 左右手身份锁定 | 手掌翻转时几何最优解会把两手整体交换 | detector 类别成为全链路硬约束，WiLoR 双假设只负责姿态 | 完整序列身份不一致观测从 6706 降为 0 |

这些修改中，阈值降低只是候选召回手段；真正带来稳定性的部分是 DS 几何正确性、锚点引导
关联、逐关节 RANSAC、内点质量统计和身份权限隔离。

### 10.2 阶段指标

开发和回归阶段使用前 60 帧快速测试，记录如下：

| 方案 | 接受帧 | 覆盖率 | 典型有效视角 | 重投影中位数 |
|---|---:|---:|---:|---:|
| 第一版全局枚举 | 13 / 60 | 21.7% | 3 | 5.49 px |
| 锚点引导 + 动态锚点 | 38 / 60 | 63.3% | 4 | 5.50 px |
| 增加短缺口时序恢复 | 57 / 60 | 95.0% | 3–4 | 4.06 px |
| 严格 detector 身份测试 | 60 / 60 | 100% | 6 | 3.09 px |

前三行来自算法开发裁剪，最后一行来自最终目标序列的前 60 帧严格身份回归，不能把四行
当作同一批帧上的消融实验；它们用于记录覆盖率问题如何逐步被发现和解决。下方 1104 帧
旧/新身份版本使用同一批完整数据，才是严格身份修复的直接对照。

完整 1104 帧的旧/新身份版本比较：

| 指标 | 未锁定身份 | 严格 detector 身份 |
|---|---:|---:|
| 接受帧 | 1063 | 1053 |
| 接受率 | 96.29% | 95.38% |
| 重投影中位数 | 3.64 px | 3.51 px |
| 有效视角中位数 | 6 | 6 |
| 身份反向帧 | 572 | 0 |
| 身份不一致有效观测 | 6706 | 0 |

严格身份少接受 10 帧。这是有意的质量选择：身份证据与几何证据冲突时，系统宁可留下
短暂缺口，也不把右手输出为左手或把左手输出为右手。

### 10.3 六目相对双目究竟提升了什么

完整严格身份结果中，可比较的 2106 只多目手显示：

```text
camera2/3 双目跨视角误差中位数：4.13 px
六目融合跨视角误差中位数：      3.52 px
相对改善：                       14.75%
双目与六目三维结果差异中位数：  19.73 mm
```

六目的收益不是把 WiLoR 网络权重本身训练得更好，而是在推理后利用额外视角完成：

- 被遮挡关节的补充观测；
- 错误单目预测的 RANSAC 剔除；
- 更宽基线的三维约束；
- 对 detector/WiLoR 瞬时漏检的跨视角补偿；
- 更可靠的质量判断和拒绝机制。

因此这里的“提升手势识别”准确说是提升最终多视角 3D 姿态的稳定性和一致性，不代表单独
拿某一路图像输入 WiLoR 时网络精度已经改变。

## 11. 可视化说明

输出视频为 3×2 六目布局：

```text
camera0 | camera1 | camera2
camera3 | camera4 | camera5
```

颜色和状态：

| 显示 | 含义 |
|---|---|
| 黄色框/骨架 | 右手，`detector.pt` 类别 1 / `right` |
| 蓝绿色框/骨架 | 左手，`detector.pt` 类别 0 / `left` |
| `USED n/42` | 该相机有 n 个双手关节作为最终 RANSAC 内点 |
| `INACTIVE` | 当前相机没有参与该帧融合 |
| 红色 `OUTLIER` | 有候选框，但没有关节通过最终内点检验 |
| 红色 `REJECTED` | 整帧未通过最终输出质量门限 |

`ACCEPTED`/`USED` 是质量状态，不负责定义左右手；黄色/蓝绿色只由严格 detector 身份决定。

当前修复后完整视频：

```text
output/gen6_pose_full_v3/fusion_handedness_strict_full/diagnostic_6view.mp4
```

规格为 1440×780、30 FPS、1104 帧、36.8 秒。

## 12. 一键运行

### 12.1 先跑冒烟测试

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand

./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording.mcap \
  --output /path/to/new_gen6_smoke_output \
  --conda-env ego-hand \
  --device cuda \
  --max-frames 60 \
  --batch-size 4
```

### 12.2 完整运行

冒烟测试确认后使用新的输出目录：

```bash
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording.mcap \
  --output /path/to/new_gen6_full_output \
  --conda-env ego-hand \
  --device cuda \
  --max-frames 0 \
  --batch-size 4
```

`--max-frames 0` 表示处理全部同步帧。

## 13. 只重跑融合，不重复 WiLoR

如果六路预测已经存在，仅需修正身份或调整融合参数，不必再次运行 GPU 推理：

```bash
cd /home/zdh/ego_hand_system

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/fuse_multiview_wilor_guided.py \
  --dataset /path/to/output/normalized_multiview \
  --predictions /path/to/output/wilor_multiview \
  --output /path/to/new_fusion_output \
  --anchor-cameras camera2 camera3 \
  --detector-handedness strict \
  --max-frames 0

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/render_multiview_wilor.py \
  --dataset /path/to/output/normalized_multiview \
  --fusion /path/to/new_fusion_output \
  --output /path/to/new_fusion_output/diagnostic_6view.mp4
```

输出目录必须是新目录，程序不会覆盖旧融合结果。

## 14. 输出目录

```text
OUTPUT/
  run_config.json
  normalized_multiview/
    manifest.json
    multiview_frames.csv
    calibration/camera0.json ... camera5.json
    cameras/camera0/video.mkv ... camera5/video.mkv
  wilor_multiview/
    summary.json
    camera0/predictions.jsonl ... camera5/predictions.jsonl
  fusion_multiview/
    accepted.jsonl
    rejected.jsonl
    summary.json
    diagnostic_6view.mp4
```

`accepted.jsonl` 每帧包含：

- `sync_index`；
- 实际锚点相机；
- 左右两只手；
- GEN base 坐标系 `joints_base_m`，形状为 21×3，单位米；
- 每关节内点视角数；
- 每相机的 detector 类别、框、2D 关节和内点关节数；
- 重投影质量；
- 两目锚点与多目三维差异。

## 15. 断点复用与配置保护

一键脚本以各阶段的 `manifest.json` 或 `summary.json` 作为完成标记：

- 已完成的六路标准化可复用；
- 已完成的六路 WiLoR 可复用；
- 已完成的融合可复用；
- 不完整目录不会被自动删除或覆盖。

`run_config.json` 保存 MCAP 路径、大小、时间、相机列表、模型参数、逐相机阈值和融合算法
版本。如果在同一输出目录更换 MCAP 或算法版本，脚本会停止并要求使用新目录，防止结果
混用。

## 16. 验收与排错

### 16.1 建议验收项

每次新数据先检查：

1. `normalized_multiview/manifest.json` 中六路帧数一致；
2. 同步 `abs_delta_us_p95` 明显小于帧周期；
3. `fusion_multiview/summary.json` 接受率满足场景要求；
4. `detector_handedness_mismatch_observation_count == 0`；
5. 左右手数量合理；
6. 有效关节视角中位数至少为 3；
7. 重投影中位数建议低于 10 px；
8. 视频中手掌翻转前后颜色保持不变；
9. 红色 OUTLIER 不应绘制为有效骨架；
10. 快速运动、交叉和遮挡片段人工抽查。

### 16.2 左右手又发生交换

检查 summary：

```text
parameters.detector_handedness == "strict"
detector_handedness_mismatch_observation_count == 0
```

如果旧输出没有这些字段，它是在身份锁定修复前生成的，需要复用已有 predictions 重新运行
融合。例外是本文记录的 `fusion_handedness_strict_full`：它已经启用 strict 并完成独立
零冲突审计，只是生成时间早于 summary 字段的加入。不要把旧 `fusion_multiview` 视频
当作修复后结果。

### 16.3 外围相机出现背景框

低阈值 detector 出现候选背景框是允许的，但最终应显示为红色 OUTLIER 或不被选择。若背景
框被显示为 USED，应检查：

- DS 标定是否来自同一 MCAP；
- 时间同步是否异常；
- `T_base_camera` 是否变化；
- 是否错误关闭 strict handedness；
- 是否放宽了关联或 RANSAC 阈值。

### 16.4 接受率下降

严格身份可能比纯几何少接受少量帧。不要首先放宽最终质量门限，应按顺序检查：

1. detector 是否在 camera2/3 稳定识别左右手；
2. 备用相机是否提供相同 detector 类别；
3. 时间同步；
4. DS 投影/反投影；
5. 最后才调整初始锚点或外围关联阈值。

## 17. 测试

当前相关测试共 9 项：

```bash
cd /home/zdh/ego_hand_system

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python tests/test_double_sphere.py

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python tests/test_genrobot_mcap.py

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python tests/test_multiview_wilor.py
```

覆盖内容包括：

- GEN DS 投影公式；
- DS 投影/反投影往返；
- 双目正视差和米制三角化；
- MCAP H264/标定基础约定；
- 多相机帧不重复同步；
- 原生 DS 多视角三角化；
- 引导关联拒绝背景候选；
- strict handedness 禁止 Left/Right 交换。

## 18. 当前边界

- 这是 WiLoR 后处理和多视角融合，不是重新训练后的六目神经网络；
- strict 身份依赖 `detector.pt` 类别正确；如果 detector 本身持续判错，融合会保持该错误身份；
- 快速运动、严重模糊或六路同时遮挡仍可能产生拒绝帧；
- 当前输出核心是 21 点三维姿态，不是离散手势类别（如 OK、握拳、数字手势）分类器；
- 六目姿态结果已可通过独立导出脚本进入训练标签流程；训练导出不会反过来改变本节描述的
  六目识别结果；
- 如需连续 MANO 网格，可在身份锁定后的六目 3D 上增加独立 MANO 时序拟合，但不应重新
  让 MANO 或几何模块决定左右手身份。

## 19. 推荐使用的结果

本次完整验证应查看：

```text
output/gen6_pose_full_v3/fusion_handedness_strict_full/summary.json
output/gen6_pose_full_v3/fusion_handedness_strict_full/accepted.jsonl
output/gen6_pose_full_v3/fusion_handedness_strict_full/rejected.jsonl
output/gen6_pose_full_v3/fusion_handedness_strict_full/diagnostic_6view.mp4
```

旧目录：

```text
output/gen6_pose_full_v3/fusion_multiview/
```

是在严格 handedness 修复之前生成的，包含手掌翻转时左右手交换问题，仅用于历史比较，
不应作为当前六目姿态识别结果。

## 20. 接入 WiLoR 训练标注

本节记录设计约束和本次结果。面向实际操作的独立手册见
[`GEN_SIX_CAMERA_WILOR_LABEL_EXPORT_GUIDE.md`](GEN_SIX_CAMERA_WILOR_LABEL_EXPORT_GUIDE.md)。

六目识别确认无左右手交换后，使用：

```bash
cd /home/zdh/ego_hand_system

./scripts/run_multiview_wilor_label_export.sh \
  --experiment /home/zdh/ego_hand_system/output/gen6_pose_full_v3 \
  --fusion /home/zdh/ego_hand_system/output/gen6_pose_full_v3/fusion_handedness_strict_full \
  --output /home/zdh/ego_hand_system/output/gen6_pose_full_v3/wilor_training_labels_right_v1 \
  --conda-env ego-hand \
  --device cuda \
  --max-samples 0
```

新增标注链路：

```text
严格六目融合 21×3（GEN base）
        |
        v
转换到 camera2 原始光学坐标
        |
        v
camera2/3 共同针孔校正坐标 + 六目支持度权重
        |
        v
两只物理手都在 MANO_RIGHT 规范空间进行共享 shape、逐帧 pose 的时序拟合
        |
        v
每个接受帧、每只手、每个 camera2/3 视图生成一个训练样本
        |
        v
images/xxxxxx.jpg + labels/xxxxxx.npy
```

选择 `camera2/3` 校正图作为训练图片是数据契约要求，不是退回双目识别。物理左手样本在
校正后水平翻转，`K` 和全部标签一起进入右手规范空间；训练端不能再次翻转。标签的 3D 初值
仍来自六目 RANSAC；只是 `000865.npy` 使用单个针孔矩阵 `K`，无法表达原始 DS 鱼眼。
如果保存原始鱼眼图片却写入针孔 `K`，即使数组 shape 正确，标签投影语义也是错误的。

每个 `.npy` 保持以下字段顺序与类型：

```text
bbox          numpy.float64  (4,)
vertices      numpy.float32  (778, 3)
joints_3d     numpy.float32  (21, 3)
joints_2d     torch.float32  (778, 2)   # 实际为 778 个网格顶点投影
side          numpy.float32  scalar     # 0 Left, 1 Right
trans         numpy.float32  (3,)
K             numpy.float32  (3, 3)
mano.global_orient  numpy.float32 (1, 3, 3)
mano.hand_pose      numpy.float32 (15, 3, 3)
mano.betas          numpy.float32 (10,)
```

最终校验同时检查：

1. 图片与标签同名且数量相等；
2. 字段顺序、Python 类型、dtype、shape 与 `000865.npy` 一致；
3. bbox 位于对应图片内；
4. `vertices + trans` 全部位于相机前方；
5. `K @ (vertices + trans)` 的投影与 `joints_2d` 最大误差不超过 `1e-3 px`；
6. summary、index 和磁盘实际文件数量一致；
7. 不论 `side` 是 0 还是 1，都只用 `MANO_RIGHT.pkl` 重放 778 顶点和 21 关节。

下面是旧版 side-specific 导出的历史结果，只用于数量和质量门限参考；它不是当前
`wilor_right_canonical_v1` 正式训练数据，必须用上面的新输出目录重新生成：

| 指标 | 数值 |
|---|---:|
| 成对图片/NPY | 4117 / 4117 |
| 左手样本 | 2037 |
| 右手样本 | 2080 |
| camera2 样本 | 2059 |
| camera3 样本 | 2058 |
| 多目融合来源样本 | 4117（100%） |
| 被质量门限拒绝 | 95 |
| MANO 重投影拒绝 | 83 |
| 单相机内点关节不足拒绝 | 12 |
| 对照 `000865.npy` 全量校验 | 通过 |
| 778 顶点投影最大误差 | 0.0 px |

正式数据位置：

```text
/home/zdh/ego_hand_system/output/gen6_pose_full_v3/wilor_training_labels_right_v1/dataset
```
