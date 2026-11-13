# EGO + NOKOV 双手原图标签流程

只需要执行命令时，请看 [精简命令清单](EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)；本页保留详细原理、参数和验收记录。

目标：NOKOV 同时采集左右手；同一相机帧同时出现双手时，原图只保存一张，左右手各有自己的三维姿态、MANO 参数及标签。物理身份固定为 `side=0` 左手、`side=1` 右手，不能按画面中的左右位置分配。

当前标签链路直接使用原相机坐标、真实内参和 KB 畸变模型，不生成或读取双目矫正参数，也不对图片执行去畸变。**KB 投影是在原图上描述镜头畸变，不是消除畸变。** 下文所说的外参旋转和平移仅变换三维点，图片像素保持原样。

## 当前联合实现：受限修正 v2（20261112）

此前 `multiview_surface_markers` 联合结果保留作历史对照，不能作为当前已验证的高质量标签。当前默认仍关闭联合；启用后使用 `baseline_anchored_joint_v2`，其实现如下。下文 20261111 的试验数字属于旧实现。

1. 先用原 NOKOV 三维关节目标拟合双手基线。左右手各有一套 MANO 时序参数；两手分别优化，不加入接触约束。
2. 每手在同一时刻只存在一套共享姿态。用每台相机自己的 K、KB D、外参，将估计表面 marker 投影到该相机原图，与实际 RGB 匹配像素比较。图片不去畸变；同一姿态接受所有有有效观测的视角约束，不为每台相机生成不同姿态。
3. 优化损失为原三维关节误差 + 原时序/姿态约束 + RGB 投影项 + 相对基线的关节位置约束。手形冻结；只更新全局旋转、手指姿态及位移。RGB 残差以 4 px 归一化，稳健项按每个视角归一化后平均四路，乘 `0.006 m × 0.25`；基线项为 `0.006 × mean(((J-J_base)/0.002 m)^2)`。基线项是软约束，不等于硬限制 2 mm。
4. 验收候选：该帧须有至少 6 个训练 RGB 观测且覆盖至少两台相机；同一批 RGB 点的投影距离中位数至少下降 0.01 px；三维关节到 NOKOV 的中位数/P95 相对基线增量分别不超过 0.1/0.2 mm；所有关节和所有网格顶点的位移均不超过 3 mm。没有足够观测、候选不合格、RGB 留出帧均使用基线参数。
5. 再检查相邻帧的修正量变化，任何关节超过 1 mm 的修正跳变会触发回退，并重复检查至稳定。完整基线仍可能有自身抖动，此检查只限制新增修正的跳变。留出 RGB 不参与训练或候选验收。
6. 按原来的关节误差门槛导出四路标签，执行 MANO 重建/投影检查并生成视频。拟合 `summary.json` 的 `refinement_safety` 给出接受/回退数量及各项原因，原因计数可重叠。`multiview_strategy` 用于区分历史旧算法和当前算法。

候选表面附着点仍未经独立标定，所以 v2 是保护基线的受限试验，并不保证能提升标签精度。若数据不支持可靠修正，允许全部回退；不能把输出数量或格式检查通过作为精度提升证明。首个问题片段（128 帧）两手全部回退，关节与基线最大差小于 0.0001 mm。

整段验收结果：左右手各 1,612 帧，均为 0 帧接受、1,612 帧回退；网格和关节与原基线逐点完全相同。此结果仅表明当前候选在当前门限下没有可信增益，不能宣称四目精度提升。已导出 488 张原图、976 份标签和 4 路视频；976 份全部通过 MANO 重建与 KB 投影检查。

输出目录：`/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/joint_safe_v2_192916/pipeline/`，视频位于 `export/videos/`；上级 `verification.json` 记录完整回退统计与位移核对。

## 导出每一步做什么

当前目标输出为 cam0～cam3 各自原图对应的双手姿态标签。四路共享同一套 NOKOV 三维参考和双手 MANO 拟合结果，再分别转换到各自的物理相机坐标，使用各自 K/D 投影、筛选和保存。合并目录中的 `camera` 字段标识标签所属相机；不要求四路同时通过筛选。各步骤的输入、处理与结果如下：

| 步骤 | 输入 | 具体处理 | 输出 |
| --- | --- | --- | --- |
| 1. 原始数据检查 | 同一次录制的 EGO 与 NOKOV 目录 | 检查视频、时间戳、标定及双手 marker 身份，确认两设备确实同期采集 | 可处理的录制清单与检查结果 |
| 2. 时间同步 | EGO PTS/IMU、NOKOV 时间戳与头部刚体运动 | 估计两设备时间对应关系，将双手三维观测关联到每个相机同步帧 | `multiview_frames.csv` 和同步结果 |
| 3. 空间对齐 | 双手三维点、头部刚体、相机内外参 | 将左右手统一到标准化相机基坐标，核对原图 marker 投影；先用 `align_native_dual_nokov.py` 初始对齐，再用 `native_fourview_alignment.py` 共同修正空间变换，见第 2、2.1 节 | `aligned_nokov21.npz`，含 `[2,F,21,3]` 双手参考及有效掩码 |
| 4. 准备拟合输入 | 已对齐双手参考、相机标定 | 保留左右手身份和缺点掩码，生成拟合所需坐标及各相机观测记录 | `mano_input_cam01.npz`、`nokov_observations_cam01/` |
| 5. 双手 MANO 拟合 | 双手三维参考 | 默认优化双手手形、姿态与位移，使用 NOKOV 三维监督和时序约束；显式联合模式冻结基线手形，增加真实 RGB marker 项 | `track_0.npz` 左手、`track_1.npz` 右手，包含网格、关节、参数和拟合误差 |
| 6. 原图标签导出 | MANO 轨迹、原始帧、相机 KB 标定 | 将三维几何转换到各物理相机坐标，用鱼眼投影生成二维位置和包围框；一帧保存一张图并关联两只手 | `images/`、每手一份的 `labels/`、`index.jsonl`、`frames.jsonl` |
| 7. 质量筛选 | 拟合误差、有效观测和画面内投影 | 默认按每手三维误差 median ≤ 10 mm、P95 ≤ 20 mm 等门限筛选；双手模式中任一手不合格即排除该相机帧 | 保留样本及 `rejected.jsonl` 中的原因 |
| 8. 严格检查与合并 | 每段录制的图片和双手标签 | 验证鱼眼投影、MANO 参数重建及双手关联；合并时保留录制来源和对应标定 | 统一的 `export/merged/`、统计和检查日志 |
| 9. 原图预览与验收 | 已检查的双手数据集 | 在同一张原图上同时叠加两只手的框、关节和网格顶点，检查身份与贴合情况 | `export/preview/` 中的抽样预览 |

步骤 6 的包围框由拟合网格投影生成，供姿态训练裁剪使用；人工原图检测标签及 YOLO 训练由用户另行维护。步骤 7 的筛选实际在导出写入前执行，表中分列是为了说明保留与排除的依据。

双手完整样本中，**一张图片对应两份手实例标签**，`frames.jsonl` 将它们关联起来。最终图片保持原始鱼眼几何；三维点之间的物理坐标变换直接使用相机外参。以上检查证明数据自洽性，模型标签的实际精度仍受同步、标定、marker 定义和 MANO 拟合影响。

## 当前是否四目优化

本文按四目路线给出完整命令：第 1、2 节准备原始数据并完成初始空间对齐，第 2.1 节用四路真实原图 marker 共同优化空间变换，第 3 节一次拟合双手并导出四路各自的标签，第 6 节生成视频。无需先额外跑一次两目拟合。

**默认模式仍仅用 NOKOV 三维约束拟合 MANO。显式开启下节配置后，新增四路 RGB marker 约束共同优化双手姿态。** 四目空间标定与四目 MANO 姿态优化需要区分。之前的 WiLoR 训练对照使用 cam0/cam1，不能当作四目训练对照。

将同一份三维结果投影到多个相机，不等于这些相机共同参与优化。四目 MANO 姿态联合优化使用各原图中与模型点对应的独立二维观测、身份与遮挡判断，并用各相机自己的 K、KB D 和外参构建误差；不需要先去畸变。当前由 NOKOV 三维点直接生成的二维参考仅用于诊断和筛选，不能重复当作独立图像真值。

## 显式选择标签生成模式

标签入口支持 `--label-config /home/p3/data_sda1/EGO_glove/ego_hand_system/configs/native_label_generation.json`。不传该参数，或使用下面的默认配置，均保持现有 NOKOV 三维拟合 → 各相机独立原图投影、筛选和导出流程：

```json
{
  "schema_version": 1,
  "multiview_mano_optimization": {
    "enabled": false
  }
}
```

`--cameras camera0 camera1 camera2 camera3` 只选择四路输出，不会自动开启该开关。已经实现的四目共享空间校准也不等同于开启 MANO 图像优化。

**`enabled=true` 已接入实际联合损失。** 保留真实 K、KB D、相机间外参和已经接受的空间对齐，优化共享 MANO 姿态及位移，然后分别投影到四路原图。旧 `w_2d=0` 仍禁用 NOKOV 自投影二维项；新的 RGB 项由 `rgb_weight` 独立控制。

算法使用 NOKOV 三维球心约束、四路真实 RGB 检测球心的稳健投影误差和原有时序/姿态正则。各相机先按有效观测归一化，再平均四路误差；像素残差以 `sigma_px=4` 归一化，乘 `0.006 m × rgb_weight` 后加入损失。没有匹配的点不参与 RGB 项；自动匹配不保证完全识别遮挡和误关联，仍需抽查。固定附着位置由基线轨迹估计为网格三角形及重心权重，球心沿表面法向偏移 **2 mm**（用户确认球直径 4 mm、直接贴手套）。当前使用映射后的 **21 个 marker**，没有使用全部 24 点。联合阶段冻结基线手形，细化逐帧姿态和位移。

没有专门附着标定录制，因此附着点是候选估计，并非独立验证的解剖学真值；手套厚度尚未测量。RGB 自动匹配受初始投影门限影响，留出帧只能验证当前录制上的投影表现；不能当作跨录制泛化或关节精度证明。留出 RGB 不参与拟合，留出帧仍使用 NOKOV 三维监督和时序约束。当前观测包是本段录制专用，不可直接复用于下一段；新录制应重做对齐、基线和观测包。

### 开启联合优化的具体命令

先按第 1～3 节完成默认基线，再执行下列步骤。以下给出 20261111_192916 的实际路径；新录制替换对应变量。输出使用新目录，保留基线供对照。

```bash
TASK_ROOT=/home/p3/data_sda1/EGO_glove
TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
TASK_RUN="$TASK_ROOT/wilor_training/output/native_kb_training_20261111"
TASK_PROCESSED="$TASK_RUN/real_dual/pair_20261111_192916"
TASK_BASE="$TASK_RUN/fourview_192916/native_labels"
TASK_ALIGN="$TASK_RUN/fourview_192916/alignment_180"
TASK_JOINT="$TASK_RUN/joint_mano_192916"
mkdir -p "$TASK_JOINT"

# 1. 生成独立 RGB 观测与表面附着包；已有包不可覆盖。
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/prepare_multiview_mano_bundle.py" \
  --input "$TASK_BASE/mano_input_cam01.npz" \
  --baseline-fit "$TASK_BASE/mano_fit_full_cam01" \
  --rgb-correspondences "$TASK_ALIGN/rgb_correspondences.json" \
  --rgb-alignment "$TASK_ALIGN/fourview_alignment.json" \
  --marker-diameter-mm 4 --output "$TASK_JOINT/constraints.npz"

# 2. 写入显式启用配置；路径及基线哈希会检查。
cat > "$TASK_JOINT/enabled_config.json" <<EOF
{
  "schema_version": 1,
  "multiview_mano_optimization": {
    "enabled": true,
    "bundle": "$TASK_JOINT/constraints.npz",
    "initial_fit": "$TASK_BASE/mano_fit_full_cam01",
    "rgb_weight": 0.25,
    "sigma_px": 4.0,
    "pose_iterations": 200
  }
}
EOF

# 3. 联合拟合、四路原图导出、严格检查、合并及静态预览。
TASK_LABELS="$TASK_JOINT/pipeline_enabled"
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/run_nokov_native_labels.py" \
  --processed-session "$TASK_PROCESSED" \
  --aligned-nokov "$TASK_ALIGN/aligned_nokov21.npz" \
  --hand-side both --cameras camera0 camera1 camera2 camera3 \
  --sample-stride 8 --max-samples 0 --device cuda \
  --label-config "$TASK_JOINT/enabled_config.json" --output "$TASK_LABELS"

# 4. 视频另行生成；30 FPS 原视频按每 8 帧采样，回放为 3.75 FPS。
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/render_wilor_training_videos.py" \
  --dataset "$TASK_LABELS/export/merged" --output "$TASK_LABELS/export/videos" \
  --fps 3.75 --workers 2 --video-codec auto
```

使用相同附着点、相同有效帧比较基线和联合结果（输出 JSON 不覆盖已有文件）：

```bash
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/compare_multiview_mano_fits.py" \
  --bundle "$TASK_JOINT/constraints.npz" \
  --baseline-fit "$TASK_BASE/mano_fit_full_cam01" \
  --refined-fit "$TASK_LABELS/mano_fit_full_cam01" \
  --output "$TASK_JOINT/comparison_full.json"
```

`constraints.json` 记录附着估计误差及来源，拟合 `summary.json` 记录训练/留出 RGB 分相机误差。导出质量门限在新模式下针对表面 marker 球心误差；`fit_nokov_marker_*` 表示该误差，`fit_nokov_joint_*` 仍表示关节到 marker 的诊断距离，二者不能混用。标签始终保存标准 MANO 关节/网格，而不是把表面 marker 改名为关节。不同模式的过筛数量不能直接代表关节精度提升。

默认 `configs/native_label_generation.json` 继续保持 `enabled=false`。开启模式的缺少观测、标定/帧序列不匹配和基线哈希不一致都会报错，不静默降级。输出根目录保存解析后的 `label_generation_config.json`。


### 20261111_192916 联合模式整段验收

使用上面的 `enabled=true` 配置，左右手各拟合 1,612 帧。以每 8 帧抽样、双手完整帧策略导出 **475 张原图、950 份标签**：camera0 180 张、camera1 180 张、camera2 102 张、camera3 13 张。950 份标签全部通过 KB 投影和 MANO 参数重建检查，最大投影自洽误差约 0.000061 px，最大网格重建误差约 1.2e-7 m。这些是格式/几何自洽检查，不是实际标注精度。

同一候选附着位置、相同有效帧、未参与 RGB 损失的观测比较如下，单位 px；观测数指匹配的 marker 点数，不是帧数。结果来自 `joint_mano_192916/comparison_full.json`。

| 手 | 相机 | 留出观测数 | 中位数：基线 → 联合 | 中位数下降 | P95：基线 → 联合 |
| --- | --- | ---: | ---: | ---: | ---: |
| 左手 | camera0 | 131 | 12.18 → 5.76 | 52.7% | 24.60 → 14.27 |
| 左手 | camera1 | 70 | 11.76 → 6.55 | 44.3% | 25.94 → 17.53 |
| 左手 | camera2 | 78 | 13.18 → 6.65 | 49.6% | 28.43 → 16.66 |
| 左手 | camera3 | 135 | 12.42 → 5.93 | 52.3% | 25.52 → 15.66 |
| 右手 | camera0 | 48 | 9.50 → 7.01 | 26.1% | 21.82 → 18.01 |
| 右手 | camera1 | 36 | 8.01 → 6.32 | 21.1% | 25.41 → 20.18 |
| 右手 | camera2 | 40 | 8.82 → 6.43 | 27.1% | 27.54 → 20.55 |
| 右手 | camera3 | 14 | 7.30 → 5.36 | 26.6% | 14.63 → 10.27 |

此处提升针对本录制、自动匹配到的 **marker 球心投影误差**。联合模式同时改变了三维拟合点语义并增加 RGB 项，尚未做仅去掉 RGB 项的消融，不能把全部提升归因于四目 RGB，也不能等同于 MANO 解剖关节精度提升。右手 camera3 只有 14 个留出观测，证据有限。基线导出 488 张与联合导出 475 张使用了不同的误差点语义，数量不直接比较精度。当前只验证标签算法，未将这批新标签重新用于 WiLoR 训练对照。

本次输出根目录：
`/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/joint_mano_192916/pipeline_enabled/`

- 标签：`export/merged/`；静态四路预览：`export/preview/`。
- 视频：`export/videos/`；每路独立文件，沿用 3.75 FPS 回放。
- 拟合及分相机误差：`mano_fit_full_cam01/summary.json`。
- 检查记录：`export/logs/merged_check.log`；模式记录：`label_generation_config.json`。

### 关闭 2 mm 球心偏移的对照实验

用户要求检查偏移对视觉贴合的影响，因此增加 `joint_mano_192916_zero_offset` 实验。原算法的该项是沿网格法向将表面点移至球心；本次将 `constraints.npz` 内两只手全部 `offset_m` 设为 **0**，不更改 NOKOV 原始数据。球的物理直径仍记录为 4 mm。

仅关闭偏移，不重新估计三角形或重心权重；附着点仍来自原先 2 mm 假设下的估计，以保持此次对照只改变一项。RGB 观测、训练/留出划分、相机标定、基线初始化、200 次姿态迭代及 RGB 权重 0.25 全部保持原样。新旧观测包除 `offset_m` 外已逐字段验证一致。此次试验用于判断偏移项影响，不能预先断言 0 mm 更准确。

实际完成左右手各 1,612 帧拟合，导出 **499 张原图、998 份双手标签**：camera0/1 各 189 张、camera2 107 张、camera3 14 张；998 份标签全部通过投影和 MANO 重建检查。`zero_offset_verification.json` 确认两手保存的观测点与无偏移表面点一致（最大差小于 8e-8 m）。样本数量变化不代表视觉质量必然提高，需对照视频判断。

本次路径：

```bash
TASK_ROOT=/home/p3/data_sda1/EGO_glove
TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
TASK_RUN="$TASK_ROOT/wilor_training/output/native_kb_training_20261111"
TASK_ZERO="$TASK_RUN/joint_mano_192916_zero_offset"

# 已生成专用零偏移观测包，配置引用该包；基线和原 2 mm 结果均保留。
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/run_nokov_native_labels.py" \
  --processed-session "$TASK_RUN/real_dual/pair_20261111_192916" \
  --aligned-nokov "$TASK_RUN/fourview_192916/alignment_180/aligned_nokov21.npz" \
  --hand-side both --cameras camera0 camera1 camera2 camera3 \
  --sample-stride 8 --max-samples 0 --device cuda \
  --label-config "$TASK_ZERO/enabled_config.json" --output "$TASK_ZERO/pipeline_enabled"

"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/render_wilor_training_videos.py" \
  --dataset "$TASK_ZERO/pipeline_enabled/export/merged" \
  --output "$TASK_ZERO/pipeline_enabled/export/videos" \
  --fps 3.75 --workers 2 --video-codec auto
```

## 联合结果视觉变差的定位（20261111）

原先的“各相机生成各自标签”实际是共享一套 NOKOV 双手 MANO 拟合，再使用各相机 K/D/外参投影，并非每台相机单独优化不同的 MANO。此次比较固定原录制、空间对齐、物理左右身份和相机标定，比较默认、联合 2 mm、联合 0 mm 三组。

### 已定位的主要变化：三维拟合点被整体替换

`fit_mano_sequence.py` 在默认分支令 `fit_points=joints`；联合分支令 `fit_points=multiview.centres(vertices)`，同一个 `loss3d` 随之改为候选表面附着点到 NOKOV marker 的误差。新增模式不仅增加 RGB，还替换了三维主监督。该附着点来自基线网格的全局最近表面搜索，固定三角形及重心权重；没有专门的手套附着标定。被冻结的基线手形、实际手套/marker 和 MANO 表面之间存在误差，姿态和根部位移就会被用于补偿这些误差。此次蒙皮分组检查未发现明显跨手指附着，但并不能证明局部附着位置准确。

### 控制实验

固定初始基线、零偏移、同一段 128 帧（pair 540～667）、200 次姿态迭代、手形冻结及全部时序参数，仅改变拟合点语义/RGB 权重。诊断副本仅允许 RGB 权重取 0，生产拟合器未修改。下表为相对初始基线的三维位移中位数，单位 mm；这衡量参数变化，不等于真实关节误差。

| 实验 | 左手关节变化 | 右手关节变化 | 左手手腕变化 | 右手手腕变化 |
| --- | ---: | ---: | ---: | ---: |
| 原关节目标，继续优化 | 0.37 | 0.40 | 0.33 | 0.25 |
| 候选表面目标，关闭 RGB | 5.54 | 4.47 | 12.13 | 12.28 |
| 候选表面目标，加入 RGB | 5.64 | 4.49 | 12.19 | 12.27 |

关闭/加入 RGB 的整体位移方向余弦相似度分别为左手 0.987、右手 0.997。该片段的 RGB 增量对关节位移 P95 约左手 0.97 mm、右手 0.015 mm，证明这个片段主要由表面三维目标驱动。右手 RGB 在该片段覆盖少，不能据此排除其他片段存在 RGB 误匹配。

整段 0/2 mm 结果的关节差异中位数约左手 1.72 mm、右手 1.61 mm；偏移项有影响，但不能解释全部主变化。两版的形状 betas 均与基线完全相同。

### 另外两个放大问题

- RGB 观测只来自空间标定时抽样匹配的稀疏帧：1,652 个参考时刻中，左手仅 53 个时刻有训练 RGB、其中 7 个有四路；右手仅 22 个时刻有训练 RGB、没有四路同时有观测的时刻。因此不是整段每帧都有四路有效图像约束。自动匹配的置信度当前统一置 1，遮挡和误关联需要进一步验证。
- 新模式导出按 marker 球心误差筛选，旧模式按关节到 marker 的距离筛选。两种门槛语义不同，marker 误差下降和检查通过不能证明骨架/手指视觉贴合提升。此前报告的约 21%～53% 降幅是候选附着点投影指标，不能用它认定标签整体更好。

### 每台相机的同帧对比

已生成三列对比视频：左列默认，中列联合 2 mm，右列联合 0 mm。仅使用三组均导出的同一相机同一帧，使用相同裁剪；全部共用原图经 SHA256 核对相同。camera0/1 各 164 个共同帧、camera2 89 个、camera3 13 个。共同帧覆盖不代表所有被拒绝帧。

视频与抽样图片：
`/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/joint_diagnosis_192916/comparison/`

- `camera0_baseline_2mm_0mm.mp4` ～ `camera3_baseline_2mm_0mm.mp4`。
- 对照图片 `camera*_sync*.jpg`，例如 camera2 sync832 可观察拇指位置的差别。
- 可复核数据：上级目录 `ablation_metrics.json`、`diagnostics_full.json`、`comparison_metrics.json`，以及控制实验命令和日志。

### 后续修改方向

先保留默认路线作为当前基线。联合模式应保留原三维约束，另外加入经过验证的 marker 表面观测项，而不是直接整体替换；增加相对基线姿态/位移的限制及超限回退，先对有可靠 RGB 观测的帧细化。附着位置需要按手指/骨段限制并独立验证；随后增加时序 RGB 跟踪、身份与遮挡筛选，以提高有效覆盖。验收需要同帧骨架/网格贴合、困难帧、marker 误差及轨迹稳定性共同判断。以上是待实现方案，本次定位未替换生产算法。

## 借鉴旧多目链路的代码审查（20261112）

本次检查了同项目 GEN 六目链路 `run_multiview_wilor_experiment.sh`、`fuse_multiview_wilor_guided.py`、`fuse_multiview_wilor.py`、`glove_marker_assist.py`、`fusion_anatomy_refinement.py`，以及 `omniegocap_orbbec/ego_hand_system` 中的旧实现副本。旧链路主要是各视角二维候选的关联、鲁棒三维融合和时序修复，不等同于直接用四路图像优化 MANO 参数。

| 可借鉴部分 | 现有实现 | 如何用于当前设备 |
| --- | --- | --- |
| 原像素反投影射线 | `fuse_multiview_wilor.py:_ray_in_base`、`camera_models/kb.py:unproject` | 已支持真实 KB 标定；只把坐标反投影为射线，不生成去畸变图像 |
| 逐点 RANSAC | `fuse_multiview_wilor.py:triangulate_ransac` | 对同一物理 marker 的多目观测找几何一致内点，拒绝错误视角，而非平均拉动所有相机 |
| 深度可靠性 | `_ray_geometry_weights`、`_ray_condition_number` | 近乎平行射线即使像素残差小也需降权/拒绝；四目不能照搬六目的视角数量门槛 |
| 视角共识 | `fuse_multiview_wilor_guided.py:_robust_view_consensus` | 按手、按帧动态去除异常视角，不固定认为某台相机永远可靠 |
| 同手/同指对应 | `glove_marker_assist.py:associate_marker_blobs`、`_remove_same_finger_reversals` | 借用一对一、手指归属和顺序约束，利用 NOKOV 物理身份引导；不得直接将球心视为解剖关节 |
| 时序修复与复投影保护 | `fusion_anatomy_refinement.py` | 借鉴可靠邻帧支持、骨长/掌部约束与修改后逐视角复投影检查，显式区分观测和插值 |

### 不宜照搬的部分

- 旧亮斑检测器针对低饱和高亮反光球；当前白手套上的球常呈暗斑，需沿用/改造当前暗斑检测，不能直接复制 HSV 亮斑阈值。
- 旧 WiLoR 单目二维输出使用虚拟裁剪相机投影，不是独立人工真值，且可能带镜头/模型偏差；可作关联初值或弱先验，不能作为 NOKOV 标签的唯一真值。
- 旧六目 DS 镜头参数及固定 camera2/3 锚点不能复制到 EGO 四目。复用的是统一的射线算法，传入本设备真实 KB 模型。
- `prepare_multiview_mano_input.py` 仍包含历史矫正坐标准备；当前保持 `prepare_native_nokov_mano_input.py`，不恢复图片矫正流程。
- 旧 marker 辅助默认只关联手指 1..20，将手腕视为没有直接 marker；本次 NOKOV 映射中索引 0 是实际 `WristM`。手腕需单独处理，不能直接套旧索引语义。
- 旧入口 `FINAL_RENDERER` 默认 `hybrid`，三维不可靠时可显示单目二维回退。旧视频的二维贴合好，不直接证明底层共享三维标签更准确；应核对具体视频的回退记录。

### 已做的真实数据复用验证

直接调用旧 `triangulate_ransac`，使用本次四目真实 KB 标定和现有稀疏 RGB marker 对应点，4 px RANSAC 门限及射线角度加权；没有改生产标签或标定。以原已对齐 NOKOV 作为距离诊断，不使用 NOKOV 距离挑选结果。

| 手 | 至少两视角的 marker 实例/时刻 | 至少三视角的 marker 实例/时刻 |
| --- | ---: | ---: |
| 左 | 427 / 46 | 171 / 20 |
| 右 | 133 / 15 | 17 / 2 |

这里“实例”是一个时刻一个 marker，不是一只完整手。旧几何模块可以用于 KB 四目，但现有观测只来自空间标定抽样，覆盖仍不足。它们还经过初始投影附近的自动关联，因此验证不是完全独立的标注精度测试。相关相机模型、marker 关联与多目融合的 25 项已有测试通过。

审查脚本和结果：
`/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/legacy_multiview_audit/probe_legacy_ransac.py`
及同目录 `legacy_ransac_probe.json`。

### 建议的接入顺序

先从整段四路原视频逐帧提取 marker 候选，以实际 NOKOV 投影作对应初值，借用同手分配、手指归属、一对一匹配和顺序校验；保存逐点置信度和明确的遮挡/缺失状态。再用原生 KB 射线 RANSAC、夹角/条件数和逐视角共识筛选观测，同时检查跨视角同步。此阶段先验证可靠率、覆盖率和 RGB/NOKOV 对应是否正确，不更新 MANO。

可靠观测验收通过后，再将其作为附加约束用于共享 MANO 的逐级细化。保留 NOKOV 原始三维观测，分别处理物理 marker 与解剖关节的语义；表面附着点应按手指/骨段限制并单独验证，不能重新整体替换三维主监督。可以先验证刚体修正，再开放局部手指姿态，均保留基线与逐阶段验收。

本次已完成旧代码审查及几何模块复用验证，尚未将逐帧观测与新鲁棒融合接入生产导出；当前 `--joint` 仍是前述受限修正 v2，不能宣称已采用旧六目融合链路。

## 逐帧观测层已接入：20261112 第一阶段

新增 `scripts/audit_native_multiview_observations.py`，统一入口增加 `--observations-only`。该模式完成原始数据准备及对齐后，逐帧检测 RGB marker、进行身份/手指拓扑关联与多目几何检查，再输出报告和四路视频，随后停止，不拟合 MANO。默认标签入口行为保持原样。具体简洁命令见命令文档。

实际实现：使用 NOKOV 原图投影作为搜索初值，复用暗斑检测器；复用旧 `associate_marker_blobs` 的手指归属、一对一和同指顺序约束，实际 WristM 另作受限匹配。跨手争用同一像素且代价接近时，双侧均拒绝。每相机时间差需在 1.5 ms 内。逐点多目 RANSAC 使用真实 KB 射线、4 px 内点门限、射线角度加权；只有至少三个内点视角、条件数不超过 10000、最大射线夹角至少 2 度且正深度的点，才标为几何一致。未通过者仍保留原匹配用于诊断，不补点。NOKOV 三维距离只用于报告，不用于挑选几何结果。

本次实际处理 192916 的全部 1,652 个同步时刻：

| 指标 | 左手 | 右手 |
| --- | ---: | ---: |
| 含 RGB 匹配的时刻 | 1,364 | 1,168 |
| 含至少一个三目几何一致 marker 的时刻 | 1,040 | 397 |
| 至少六个几何通过的相机点观测、覆盖至少两相机的时刻 | 977 | 210 |
| 原始 RGB 点观测数（跨相机计数） | 28,536 | 15,077 |
| 几何通过的 RGB 点观测数（跨相机计数） | 16,558 | 2,728 |
| 三角化 marker 到 NOKOV 距离中位数 / P95（mm） | 7.45 / 14.86 | 7.36 / 14.16 |

几何通过的 marker 实例共 5,512 个；一个实例是一只手、一个时刻、一个 marker。由于原流程只抽样、且原匹配最少点数门槛与新流程不同，这些覆盖统计不能直接当作精度提升比例。本阶段未使用 MANO 附着点、未改标定、未优化手姿态。覆盖提升仅表明新增了更多待验收证据。

输出目录：
`/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/dense_rgb_audit_192916/`

- `observations.npz`：实际 RGB 坐标、原始有效掩码、几何通过掩码、启发式置信度、三角化三维点、NOKOV 投影初值、时间戳和相机标定。
- `geometry_records.json`：每个几何通过 marker 的内点视角数、条件数、重投影残差及 NOKOV 距离。
- `summary.json`：覆盖、拒绝原因及限制；状态保留 `needs_visual_review`、`ready_for_mano=false`。
- `camera0_observations.mp4` ～ `camera3_observations.mp4`：每 8 帧展示一次，四路各 207 帧；`preview/` 为每路六张原图截图。

蓝圈是 NOKOV 投影初值，黄点是实际 RGB 匹配，绿点是通过三目几何检查的匹配。绿色不等于经人工确认的真实 marker 身份。已抽查正常、侧目及运动模糊帧；模糊帧会保留缺失。尚无独立人工像素真值，且匹配依赖 NOKOV 初值，不能据此宣称达到 7 mm 的独立标注精度。当前未划分用于 MANO 拟合的训练/留出集，不能直接把全部观测同时用于优化和评测；该 NPZ 使用独立审查 schema，不冒充已验收的拟合约束包。

本阶段的坏视角 RANSAC 排除、退化几何、跨手歧义及现有模块共 28 项测试通过；统一入口的五项检查通过，包括观测模式在 MANO 前停止。下一步需要继续审查偏差较大的 marker、右手低覆盖的相机/动作，以及固定方向误差是否与同步、标定或关联有关；这些证据通过验收后再接 MANO 修正。

## 利用完整 NOKOV 三维轨迹标定 marker 与手的关系（20261112）

原始每手 24 点为 WristM、WristIn、WristOut、HandOffset 和五指各 4 点。当前 `align_native_dual_nokov.py` 选择 `[0]+[4..23]`，只保留 WristM 和 20 个手指球心；其余三个腕掌点尚未进入 MANO 拟合。完整轨迹可以约束腕掌局部坐标、手指运动和固定附着关系，不能简单把球心当作关节中心。

本次只读检查原始 `nokov_markers.csv`，每手读到 6,077 个至少有有效 marker 的记录时刻，跨度 67.504 s。保留时刻中的各点有效率：左手最低约 98.50%，右手最低约 99.98%；这不是光学动捕精度指标，也不是缺失整帧的总覆盖率。腕掌点的相对距离并非恒定，例如 WristM—HandOffset 的 P05/P50/P95：左手 39.34/44.70/52.83 mm，右手 43.09/49.26/56.28 mm。因此不能直接把四点假设为一个刚体，须结合实际贴点位置区分腕部与掌部运动、软组织/手套形变和可能的跟踪异常。

检查脚本和 JSON：`wilor_training/output/native_kb_training_20261111/marker24_calibration_audit/audit.py`、`report.json`（相对 EGO_glove 根目录）。

### 可借用的旧附着标定实现

`nokov_ego_hand_validation/integration/pipeline/fit_surface_attachments_guarded.py` 已有按手指、骨段、背侧方向和蒙皮权重限制候选面的 `candidate_regions`，以及跨多帧估计固定三角形/重心坐标的标定阶段。比当前全网格最近面选择更有结构约束。但该旧脚本固定使用 3 mm 偏移、左手数据路径/变换及双目轮廓，不能原样用于本次双手 4 mm 球和四目数据；代码存在也不代表其结果已通过本次独立验收。

### 推荐的标定分解

1. 保留全部 24 点的原始球心、身份、有效性与时间戳，先审查长时间不合理跳变和对应交换；不对原始坐标统一减常数。
2. 根据真实贴点位置确定每点属于掌部、哪个手指/骨段或关节附近，以及哪些点可以共同建立稳定局部坐标。关节附近的皮肤点不一定严格附着于单根骨。
3. 从现有录制选动作多样的标定帧，留出连续片段/不同动作验证。估计每手共享的手形参数和每点在模型局部坐标中的附着位置；姿态与根位移是逐帧量。附着位置不得逐帧自由变化，需有手指区域、骨长/关节活动和手形先验限制，防止误差被任意分配给形状和姿态。
4. 在标定阶段用模型预测的 marker 球心解释真实 NOKOV 球心，独立检查各点偏差和候选模型轮廓。已知球直径 4 mm 只提供半径物理信息，不能代表表面到关节的全部偏差，也不能直接套旧脚本的固定 3 mm。
5. 标定验收通过后固定附着参数，在其它帧求姿态，再输出 MANO 关节和网格。RGB 四目继续提供轮廓/匹配交叉检验，不把未经校准的视觉三角化直接覆盖 NOKOV。

该思路借鉴 MoSh 对模型形状及 marker 相对模型位置的估计：[MPI-IS 介绍](https://is.mpg.de/ps/news/motion-capture-reimagined)。MoSh 的一般方法不是本手套的开箱即用精度保证。仅有表面三维点不一定能唯一确定内部关节中心，仍受动作丰富度、贴点布局和模型先验影响。先利用现有录制估计候选；若无法稳定标定，再针对不足的自由度补采动作，而非预先要求重录全部数据。

当前仍缺少 Finger1～4 和四个腕掌点实际贴点位置的确认，已向用户询问。此次完成的是数据核查与标定方案分析，尚未生成新的 24 点解剖补偿标签。

## 各步骤使用哪些参数、从哪里读取

真实相机标定来自本次 EGO 录制携带的 `*_calibration_camera.yaml`；标准化后分别写入
`ego/normalized_multiview/calibration/camera0.json`、`camera1.json` 等文件。每个相机读取自己的标定，不能将 cam0 的内参复制给其他相机。

| 使用步骤 | 参数 | 来源 | 具体用途 |
| --- | --- | --- | --- |
| 原始检查、标准化 | `image_size`、相机编号与流映射 | 相机标定 JSON、`manifest.json` | 确认分辨率，并对应 `camera0 ↔ cam_0/left`、`camera1 ↔ cam_1/right`；left/right 流名表示相机位置，不表示人的手身份 |
| 时间同步 | 各相机 PTS、IMU 时间戳、NOKOV 时间戳、时间偏移 | 原始 CSV 与同步结果 | 将同一时刻的视频和双手三维观测配对；旧原始入口将 NOKOV 毫秒时间转为秒，当前固定时钟比例 `a=1`，不应沿用上一段录制的偏移 |
| 空间对齐 | NOKOV 头部刚体逐帧位姿、`T_cam0_head` | NOKOV 刚体 CSV、本段空间标定结果（旧入口为 `alignment/cam0_marker_calibrated/alignment.json`） | 将动捕坐标中的双手三维点关联到相机；此变换需标定，不能只靠相机内参完成 |
| 各相机三维坐标转换 | `T_base_camera`，4×4 刚体变换 | 对应相机的标定 JSON | 表示“该相机 → 标准化基坐标”的变换；基坐标中的点转到该相机时使用其逆变换，平移单位米 |
| 拟合输入准备 | 真实 `K`、KB `distortion`、物理相机相对旋转和平移 | 各相机标定 JSON，直接由 `T_base_camera` 计算 | 将 NOKOV 参考写入原相机坐标，生成原图参考投影；不计算双目矫正 |
| 拟合诊断 | 原相机 `K/D`、参考相机到各相机的 `R/t` | 原始标定，保存于 `mano_input_cam01.npz` 的 `native_left/right_*` 字段 | 拟合器使用可微 KB 投影计算原图像素误差，`projection_space=native_raw` |
| MANO 拟合 | `MANO_RIGHT.pkl`、`betas`、全局旋转、手指姿态、`translation`、左右手身份 | 授权 MANO 模型、双手三维参考与优化结果 | 使用右手规范参数表达两只物理手；以实际 `side` 做正确的镜像转换，得到各手的关节与网格 |
| 默认 MANO 优化设置 | `shape_frames=256`、形状迭代 800、姿态迭代 500、窗口 96/重叠 16、PCA 45、学习率 0.008 | `run_nokov_native_labels.py` | 控制手形估计与逐帧姿态优化；主监督权重 `w_3d=1`、`w_2d=0`，并使用姿态、手形及时间平滑约束 |
| 原图标签投影 | **原相机 `K`、4 个 KB 畸变系数 `D`**、原相机三维点 | 标定 JSON；`native_raw` 标签内保存 `K`、`distortion`、`camera_model=KB` | 将拟合后的 21 关节与 778 顶点投到原始鱼眼图，实际调用 `cv2.fisheye.projectPoints` |
| 包围框生成 | `bbox_margin=0.10`、`min_visible_vertex_fraction=0.50` | 导出器默认参数 | 根据原图上的网格投影生成姿态裁剪框；边距为投影框宽、高各自的 10%，至少 50% 顶点投影需在画内 |
| 质量筛选 | 三维误差 10/20 mm、原图重投影误差 25/60 px、`complete21`、`require_both_hands` | 导出器与统一入口 | 筛选拟合质量、画内完整性及双手标注完整性；详细解释见“导出范围、质量筛选与抽帧” |
| 保存与抽帧 | `sample_stride=1`、`max_samples=0`、JPEG quality 95 | 统一入口与导出器默认参数 | 默认逐帧筛选、不限量；原图重新编码为 JPEG，但不做几何矫正；一张相机帧共享给两只手 |
| 预览与视频 | 每相机随机预览 4 帧；视频命令示例 `fps=3.75`、`workers=2` | 统一入口参数、单独的视频命令 | 控制展示数量和播放速度，不改变三维标签；统一入口默认仅输出静态预览 |

### 真实相机参数、物理坐标变换与训练参数

1. **原相机真实参数：**`K_raw`、`D_KB`、`T_base_camera`，均取自本次录制对应相机的标定。
2. **物理三维坐标变换：**从参考相机到其他相机的 `R/t`，直接根据真实外参计算。它只旋转和平移三维点，不做图像矫正，不生成 `R1/R2/P1/P2` 或重采样映射。
3. **训练模型参数：**旧 WiLoR 的 256×256 输入、裁剪倍率 2.0 与虚拟焦距属于模型配置，不能当成相机真实标定。当前训练器已支持显式 `physical_kb`：原图投影、物理腕根估计、裁剪和特征采样均使用对应几何；旧透视模式仍拒绝直接读取原图标签。原图像素转射线的几何计算不会重采样图片。

标定外参的方向和最终投影关系可写为：

```text
X_camera = R_base_camera.T @ (X_base - t_base_camera)
X_camera_mesh = vertices + trans
pixels_raw = KB_fisheye_project(X_camera_mesh, K_raw, D_KB)
```

`vertices`、`joints_3d` 与 `trans` 必须属于同一物理相机坐标约定。MANO 的 `global_orient`、根关节旋转中心和平移也同步转换；不能只旋转二维点或只替换 `K`。

### 本机实际标定数值示例

下面读取自 2026-11-09 `172920` 录制标准化标定，单位为像素，分辨率均为 **1920×1200**。它们是核查示例，新录制仍自动读取自己的标定文件，不应照抄数值。

| 相机 | fx | fy | cx | cy |
| --- | ---: | ---: | ---: | ---: |
| camera0 | 505.278256 | 505.358731 | 938.814510 | 592.150652 |
| camera1 | 505.201976 | 505.147119 | 948.194092 | 591.453740 |

```text
camera0 D_KB = [0.07646121994, -0.01419877573,  0.00194070025, 0.00007099414]
camera1 D_KB = [0.07562121214, -0.01046009014, -0.00190034978, 0.00115754732]
```

该示例相机间距约 **0.100158 m**，来自真实外参。当前链路使用原相机约 **505 px** 的焦距与 KB 畸变参数；旧矫正路线约 302 px 的虚拟焦距不参与当前标签流程。

数值来源：

```text
/home/p3/data_sda1/EGO_glove/work/20261109/ego_172920/normalized_multiview/calibration/camera0.json
/home/p3/data_sda1/EGO_glove/work/20261109/ego_172920/normalized_multiview/calibration/camera1.json
```

合并后的 `summary.json` 在 `camera_calibrations_by_source` 中按来源录制保存相机标定，`index.jsonl` 的 `calibration_group` 选择对应组；同名 `camera0` 跨录制时不会共用一份参数。标定文件哈希和 MANO 资产哈希用于追溯版本。

## 当前实现与验证范围

| 阶段 | 状态 |
| --- | --- |
| 新双手原始录制的同步与空间对齐 | 新 `align_native_dual_nokov.py` 已处理真实双手资产；需强同步、已标定头部安装外参作为初值及原图 marker 质量检查；旧入口仍为单左手 |
| 已对齐双手参考 → MANO 输入与两条轨迹拟合 | 真实双手 192916 全序列通过，左右手各拟合 1612 帧 |
| MANO 轨迹 → 原图多实例标签 | 支持按帧共享图片、分别关联左右手标签、要求双手通过质量筛选 |
| 合并、严格检查与原图预览 | 保留来源录制和每段相机标定，支持共享图片的多个手实例 |
| WiLoR 原图训练 | `physical_kb` 已完成 80 步短程训练及独立真实双手姿态评测；当前精度未证明整体优于预训练模型 |

已对真实左手、右手录制分别做短序列验证：禁止调用双目矫正、矫正映射和 `remap` 时，原图准备、拟合、导出、MANO 检查及预览通过。验证使用既有轨迹初始化及少量优化迭代，用于检查坐标和梯度，不代表新模型性能或完整重训结果。2026-11-11 新增真实双手 192916 验证：导出 730 张原图、1460 份手标签，并生成 cam0/cam1 同帧双手视频；完整结果与训练对照见 [实际验收记录](../../wilor_training/RESULT_NATIVE_DUAL_20261111_zh.md)。193034 同步较弱，保留 `needs_review`，未进入正式标签。不得将旧单手对齐结果复制成双手参考。

## 1. 保存两设备同一次录制的原始数据

EGO 目录保留四路原始 MP4、各相机 PTS CSV、IMU CSV、相机/IMU 标定 YAML 和 `meta_*.json`。NOKOV 目录保留 SDK 帧、刚体、marker、骨架 CSV、资产描述以及采集元数据，左右手须使用可区分的资产身份。只有 XINGYING CAP 文件时，需先转换成受支持的输入结构。

推荐目录：

```text
data/<take>/
├── ego/          # 完整 EGOViewer 导出
└── nokov/        # 同期 NOKOV SDK 采集结果，包含双手与头部刚体
```

## 2. 同步、标定并建立双手参考

先按时间戳及 IMU/头部刚体运动完成同步，再通过头部刚体与相机外参把双手 NOKOV 三维点转换到标准化相机基坐标系。左右手必须来自同一时间轴，不能把不同录制的单手序列拼成双手同帧参考。

当前双手对齐入口是 `scripts/align_native_dual_nokov.py`，默认资产为 `left_hand_4mm`、`right_hand_4mm`，严格检查 24 个 marker 名称和顺序，提取 WristM 与各指 1..4，共 21 点。其他资产名通过 `--left-markerset/--right-markerset` 指定，但点序仍须匹配。需要之前可靠标定的 `T_cam0_head` 作为初始化，再用本段真实 cam0 原图 marker 修正共享 SE3，不能直接沿用上一段时间偏移。

该入口不会用 WiLoR 预测作为真值，也不运行图像矫正。旧 `run_ego_nokov_pipeline.py` 仍硬编码单左手，不能用于新双手原始录制。下面命令适用于一个 `part0001`、头部刚体 `orbbec_ego4`、NOKOV `device_timestamp_raw` 单位毫秒的当前设备数据。

### 两设备原始数据的具体命令

先设置本次路径；`TASK_PROCESSED` 和后续标签输出均使用新目录。以下种子外参来自已验证的 11 月 9 日安装标定，只用于优化初值；设备安装改变较大时需重新标定。

```bash
TASK_ROOT=/home/p3/data_sda1/EGO_glove
TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
TASK_EGO=/path/to/EGO-4B_本次完整录制目录
TASK_NOKOV=/path/to/同次session/nokov
TASK_PROCESSED="$TASK_ROOT/work/本次录制_processing"
TASK_SEED="$TASK_ROOT/work/20261109/alignment_172920/cam0_marker_calibrated_right/alignment.json"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

先为原始分段文件建立无分段名的软链接，供标准化工具读取。源文件不修改：

```bash
"$TASK_PYTHON" - "$TASK_EGO" "$TASK_PROCESSED/ego_input" <<'PY'
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
if target.exists():
    raise FileExistsError(target)
if not (source/'part0001').is_dir() or (source/'part0002').exists():
    raise ValueError('当前示例只处理单个 part0001')
links = {}
for stream in ['left', 'right', 'right_side', 'left_side']:
    for suffix in ['.mp4', '_pts.csv']:
        found = list((source/'part0001').glob(f'*_camera_{stream}_part0001{suffix}'))
        if len(found) != 1:
            raise ValueError((stream, suffix, found))
        links[f'{source.name}_camera_{stream}{suffix}'] = found[0]
for pattern in ['*_calibration_camera.yaml', 'meta_*.json']:
    found = list(source.glob(pattern))
    if len(found) != 1:
        raise ValueError((pattern, found))
    links[found[0].name] = found[0]
target.mkdir(parents=True)
for name, path in links.items():
    (target/name).symlink_to(path.resolve())
PY

"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/normalize_orbbec_ego4_session.py" \
  --session "$TASK_PROCESSED/ego_input" \
  --output "$TASK_PROCESSED/ego/normalized_multiview"
```

标准化复制四路原视频，读取真实标定并建立相机帧对应表，不改变视频像素。再自动定位本段 IMU CSV，同步两设备：

```bash
TASK_IMU=$("$TASK_PYTHON" -c 'from pathlib import Path; import sys; files=list((Path(sys.argv[1])/"part0001").glob("*_imu_part0001.csv")); assert len(files)==1, files; print(files[0])' "$TASK_EGO")
"$TASK_PYTHON" "$TASK_ROOT/nokov_ego_hand_validation/tools/synchronize_ego_imu_nokov.py" \
  --ego-imu-csv "$TASK_IMU" \
  --nokov-csv "$TASK_NOKOV/nokov_rigid_bodies.csv" \
  --rigid-body orbbec_ego4 --nokov-time-field device_timestamp_raw \
  --nokov-time-scale 0.001 \
  --output-dir "$TASK_PROCESSED/alignment/synchronization" --no-plot

"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/align_native_dual_nokov.py" \
  --dataset "$TASK_PROCESSED/ego/normalized_multiview" \
  --nokov "$TASK_NOKOV" \
  --sync "$TASK_PROCESSED/alignment/synchronization/imu_nokov_sync.json" \
  --seed-alignment "$TASK_SEED" \
  --output "$TASK_PROCESSED/alignment"
```

检查 `dual_alignment.json` 的 `status=accepted` 和 `dual_marker_preview.jpg`。当前自动门槛为强同步、每只手至少 50 个留出对应点、各手误差中位数 ≤ 12 px；还需查看 P95 和困难帧，不能只看中位数。失败时只生成 `candidate_aligned_nokov21.npz` 并返回非零状态；不能将候选改名跳过检查。通过后继续第 2.1 节四目空间优化，无需分别运行左右手。

该阶段应产出：

```text
work/<take>_processing/
├── ego/normalized_multiview/
│   ├── manifest.json
│   ├── multiview_frames.csv
│   ├── cameras/
│   └── calibration/
└── alignment/aligned_nokov21.npz
```

`aligned_nokov21.npz` 的必需字段：

| 字段 | 形状与含义 |
| --- | --- |
| `joints_base_m` | `[2,F,21,3]`，左右手在标准化基坐标系中的物理 marker 三维点，单位米 |
| `valid` | `[2,F,21]`，真实观测有效性；索引 0 为物理左手、1 为物理右手 |
| `reference_timestamp_ns` | `[F]`，必须与 `multiview_frames.csv` 严格一致 |

NOKOV marker 中心与 MANO 解剖模型关节是不同监督语义，拟合和验收时分别记录误差。画面内有效投影也不等同于 RGB 中没有自遮挡。

## 2.1 四路原图共同校准空间变换

输入必须是第 2 节同次录制的 `accepted` 对齐。优化的是共享空间变换，MANO 姿态仍由下一节的 NOKOV 三维约束拟合。

设置本次路径，输出必须是新目录：

```bash
TASK_ROOT=/home/p3/data_sda1/EGO_glove
TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
TASK_PROCESSED="$TASK_ROOT/work/本次录制_processing"
TASK_FOUR="$TASK_ROOT/work/本次录制_fourview"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

四目共同校准空间变换：

```bash
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/native_fourview_alignment.py" \
  --dataset "$TASK_PROCESSED/ego/normalized_multiview" \
  --aligned-nokov "$TASK_PROCESSED/alignment/aligned_nokov21.npz" \
  --seed-alignment "$TASK_PROCESSED/alignment/dual_alignment.json" \
  --max-calibration-frames 180 \
  --output "$TASK_FOUR/alignment"
```

读取四台物理相机自己的原始 K/D/外参及时间戳，在原图中提取暗色 marker 候选，自动关联到物理左右手的 NOKOV marker。相机时间差超过 1.5 ms 的行不参与此次图像约束。相同时刻的四个视角一起划分训练与留出集合，避免该时刻跨相机进入训练。

使用鲁棒损失优化一个共享六自由度变换；不优化镜头畸变，不移动图像，不用 WiLoR 预测充当真值。留出对应关系在拟合前固定，不按拟合后的误差重新筛选。

检查 `fourview_alignment.json` 的 `status`、`heldout_by_camera`、`heldout_by_side`，以及四份 `camera*_preview.jpg`。门槛包括每目至少 50 个训练点、20 个留出点，各目留出中位数 ≤ 12 px、P95 ≤ 30 px，两只手各至少 50 个留出点且中位数 ≤ 12 px。失败时返回非零状态，只保存 `candidate_aligned_nokov21.npz`。

## 3. 一次处理双手：拟合、四路原图导出、合并、检查、预览

输入是第 2.1 节通过四目检查的真实双手对齐结果。先替换本次路径：

```bash
TASK_ROOT=/home/p3/data_sda1/EGO_glove
TASK_PYTHON=/home/p3/miniforge3/envs/ego-hand/bin/python
TASK_PROCESSED="$TASK_ROOT/work/本次录制_processing"
TASK_FOUR="$TASK_ROOT/work/本次录制_fourview"
TASK_ALIGNED="$TASK_FOUR/alignment/aligned_nokov21.npz"
TASK_LABELS="$TASK_FOUR/native_labels"
unset PYTHONPATH
```

先预检命令，不创建输出：

```bash
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/run_nokov_native_labels.py" \
  --processed-session "$TASK_PROCESSED" \
  --aligned-nokov "$TASK_ALIGNED" \
  --hand-side both \
  --cameras camera0 camera1 camera2 camera3 \
  --sample-stride 8 --max-samples 0 \
  --output "$TASK_LABELS" \
  --dry-run
```

输入和对齐质量验收通过后，正式运行：

```bash
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/run_nokov_native_labels.py" \
  --processed-session "$TASK_PROCESSED" \
  --aligned-nokov "$TASK_ALIGNED" \
  --hand-side both \
  --cameras camera0 camera1 camera2 camera3 \
  --sample-stride 8 --max-samples 0 \
  --output "$TASK_LABELS"
```

`--output` 使用新目录。该入口默认 `--hand-side both`，同时拟合 `track_0` 和 `track_1`；准备输入后使用 `--w-2d 0`，避免将同一批三维点的投影再次当作独立二维真值。原图导出自动启用 `--require-both-hands`。

默认 CUDA 拟合与检查，`--device cpu` 可切换 CPU。`--sample-stride 80 --max-samples 32` 限制最终导出，不减少 MANO 拟合长度；数量上限按手实例计数，并保留同一同步时刻的完整实例组。

### 导出范围、质量筛选与抽帧

程序默认参数为 `--sample-stride 1 --max-samples 0`；本文第 3 节示例显式使用步长 8。若改为默认参数，则逐个同步帧检查，不主动降采样，也不限制导出实例数。导出的是**所有通过质量检查的样本**，不是原始录制的每一帧都必然保留。

实际顺序：有效同步与双手参考 → 检查拟合和相机视图质量 → 要求该相机帧双手均合格 → 按同步帧抽帧 → 应用数量上限 → 写入原图与双手标签。

主要质量条件：

- 存在有效的 MANO 拟合帧；每手 21 个拟合误差均有限，三维误差中位数 ≤ 10 mm、P95 ≤ 20 mm。
- `complete21` 要求该相机的 21 个参考点有效且在画内，导出的 21 个 MANO 关节也均需在原图内。
- 原图导出器将模型关节投影直接与原图上的 NOKOV 参考投影比较，重投影误差中位数 ≤ 25 px、P95 ≤ 60 px；不读取历史矫正坐标误差作为筛选依据。
- 网格投影有效、深度为正，至少 50% 的网格顶点投影在画面内，并能形成有效包围框。画面内投影比例不代表实际遮挡可见率。
- 严格双手模式下，任一只手不合格，该相机的该帧都不保留；其他相机仍按自己的结果判断。

| 参数 | 行为 |
| --- | --- |
| `--sample-stride 1` | 保留每个合格同步时刻，当前统一入口默认值 |
| `--sample-stride 2` | 按同步帧编号每隔 2 帧选取；筛选前为 30 FPS 时约相当于 15 FPS |
| `--sample-stride 5` | 按编号每隔 5 帧选取 |
| `--sample-stride 80 --max-samples 32` | 小样本检查用法，按编号间隔 80 帧选取，每段录制限制手实例数量 |
| `--max-samples 0` | 不限制数量 |

固定间隔模式会额外保留最后一个合格同步时刻。同一个被选中的时刻保留其所有合格相机与左右手实例。数量上限按**每段录制的手实例数**计算；为了保持同步组完整，不会从组内拆掉某只手或某个相机，因此结果可能少于上限；若第一组已超过上限，仍保留该完整组。

例如 cam0/cam1 都有合格双手时，一个同步时刻对应 **2 张原图、4 份手标签**；此时 `--max-samples 32` 通常对应 8 个同步时刻、16 张图片、32 份标签。

底层 `export_multiview_wilor_training_dataset.py` 还支持 `--sample-stride 0` 的运动自适应抽帧，但当前两个统一入口要求正整数步长，不启用自适应模式。质量排除原因记录在 `rejected.jsonl`；抽帧统计查看 `summary.json` 的 `sampling`，数量上限查看 `max_samples`。

## 4. 输出与同帧双手组织

```text
work/<take>_fourview/native_labels/
├── mano_input_cam01.npz
├── nokov_observations_cam01/
├── mano_fit_full_cam01/
│   ├── track_0.npz                 # 物理左手
│   └── track_1.npz                 # 物理右手
├── native_export_config.json
├── commands.json
├── logs/
└── export/
    ├── merged/
    │   ├── images/                # 每相机、每同步帧一张原图
    │   ├── labels/                # 每只手独立姿态参数
    │   ├── frames.jsonl           # 一帧关联两只手的标签
    │   ├── index.jsonl            # 每手一行，双手可引用同一 image
    │   ├── rejected.jsonl
    │   └── summary.json
    ├── recordings/
    ├── preview/                   # 同帧叠加左右手
    ├── logs/
    └── commands.json
```

例如一张 `images/00000000.jpg` 可同时关联两份标签：

```json
{"image":"images/00000000.jpg","hand_count":2,"hands":[{"side":0,"label":"labels/00000000.npy","index":0},{"side":1,"label":"labels/00000001.npy","index":1}]}
```

原始图像保持物理侧，标签携带各相机原始内参和 KB 畸变参数。MANO 在参考相机的物理三维坐标中拟合；导出其他相机视图时，直接通过真实外参转换三维几何及 MANO 参数，再用该相机的原始 KB 模型投影。整个标签链不产生 `rectification_cam01.npz`。

`--require-both-hands` 按“相机 + 同步帧”筛选：任一只手未通过质量检查时，丢弃该相机帧并记录原因；不会将残留的单手标签冒充完整双手标注。当前严格双手集合也会排除只有一只手在画面内的帧。训练单手可见场景时应另行配置可见手监督及完整性策略。

## 5. 已有双手拟合结果的批量导出

每段录制有自己的双手拟合和标定时，编辑配置中的 `recordings`，每项填写唯一 `name`、`fusion`、`mano_fit`、`dataset`、`reference_camera` 和 `cameras`。一段录制中已经包含双手时，只需写一项，不要拆成两个单手录制。

```bash
bash /home/p3/data_sda1/EGO_glove/ego_hand_system/scripts/run_native_wilor_export.sh \
  --config /path/to/双手录制列表.json \
  --output /path/to/new_dual_hand_output \
  --require-both-hands
```

默认配置 [`configs/native_ego_glove_export.json`](../configs/native_ego_glove_export.json) 是历史单左手与单右手的两段录制，用于兼容回归，不是双手同帧参考。对这个历史配置要求双手完整帧会因没有合格双手样本而停止。

合并器保留来源录制、每段相机标定和帧级双手关联，不能按相邻帧随机拆分训练验证。当前原图格式为 schema 6，`joints_2d[778,2]` 是旧字段名，实际存放网格顶点投影；21 个关节由 `joints_3d` 与 `trans` 配合相机模型投影。

## 6. 验收与后续训练

检查 `frames.jsonl` 中双手样本的 `hand_count=2`、`summary.json` 中图片数与手实例数、左右手数量及来源录制；查看预览时同时核对身份、网格、手指和画面边缘。

严格检查覆盖原图鱼眼投影、每只手 MANO 参数重建、标签与相机标定一致性。同帧共享图片、实例去重和双手缺失策略另有回归测试。数值一致性不能替代真实 marker、标定和解剖精度评估。

后续 WiLoR 训练按手生成裁剪，但同一相机帧的左右手仍属于同一个帧组及录制组；两份手实例不能跨训练/验证集合。训练配置必须显式选择 `recipe.camera_model: physical_kb`，对应 `native_raw` 标签；不要用旧透视模式直接加载。真实训练、评测命令及当前指标见 [实际验收记录](../../wilor_training/RESULT_NATIVE_DUAL_20261111_zh.md)。当前评测使用标签框及已知手身份，待用户原图 YOLO 就绪后再评估检测器串联效果。

### 单独生成原图标签可视化视频

当前统一命令自动生成 `export/preview/` 中的静态叠加图片；视频需要在导出完成后单独执行。以下变量沿用第 3 节，`TASK_LABELS` 指向本次标签输出根目录：

```bash
"$TASK_PYTHON" "$TASK_ROOT/ego_hand_system/scripts/render_wilor_training_videos.py" \
  --dataset "$TASK_LABELS/export/merged" \
  --output "$TASK_LABELS/export/videos" \
  --fps 3.75 \
  --workers 2 \
  --video-codec auto
```

输出在 `"$TASK_LABELS/export/videos/"`，每段来源录制、每台相机分别生成 `*_labels.mp4`，`summary.json` 记录帧数、帧率、时长及实际编码器。同一帧同时叠加左右手标签。

视频按导出样本的同步帧顺序、以指定帧率播放；被质量筛选或抽帧排除的时刻会跳过，不补帧，也不恢复原始录制时长。本文 `--fps 3.75` 对应 30 FPS 输入按步长 8 抽样的参考播放速度。若导出步长改为 1，可使用 `--fps 30`；播放帧率不能恢复筛选造成的时间间隔。输出目录若已有同名视频会报错，应换新目录。

## 7. 192916 四目真实数据验证

本次结果根目录：

```text
/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/fourview_192916/
```

`alignment/` 为首次 90 帧采样尝试：右手留出点仅 45 个，未通过 50 门槛；保留候选以追溯。扩大到约 180 帧后，`alignment_180/` 通过原门槛；没有降低质量阈值。

| 相机 | 训练对应点 | 留出对应点 | 留出中位数，px | 留出 P95，px |
| --- | ---: | ---: | ---: | ---: |
| cam0 | 417 | 179 | 4.80 | 11.09 |
| cam1 | 256 | 106 | 5.37 | 11.12 |
| cam2 | 268 | 118 | 4.98 | 12.60 |
| cam3 | 455 | 149 | 4.50 | 12.26 |

总体留出中位数从 4.93 px 变为 4.86 px，P95 从 11.79 px 变为 11.83 px。四目都提供了真实图像像素约束，但当前数据上的总体改进很小，不能宣称标签精度显著提升。

这些对应关系是在初值附近自动筛选的匹配候选，并非人工核验真值，指标不覆盖匹配失败或重遮挡的时刻；种子 cam0 标定也来自同一段录制。留出结果衡量本次四目修正，不能视为整条标定链路对未见录制的独立验证。画内投影不等于没有自遮挡。

侧目完整双手覆盖少于正前方相机，因此四路最终导出量可以不同。最终数量、质量排除原因与视频以 `native_labels/export/merged/summary.json`、`rejected.jsonl` 和 `export/videos/summary.json` 为准。

本次双手全序列拟合、四目原图导出、严格检查、合并与预览已通过。左右手各拟合 1612 帧；相对 NOKOV marker 中心的拟合误差中位数/P95 分别为左手 5.04/18.32 mm、右手 5.36/17.48 mm。

| 本次步长 8 的最终输出 | 图片数 | 手实例标签数 |
| --- | ---: | ---: |
| cam0 | 184 | 368 |
| cam1 | 184 | 368 |
| cam2 | 104 | 208 |
| cam3 | 16 | 32 |
| 合计 | 488 | 976 |

每张图片均关联两份物理左右手标签，左右手各 488 份。侧目数量少主要受画内完整性及质量筛选影响；不同采样步长的输出数量不能直接用于比较路线精度。结果在 `native_labels/export/merged/`，原图叠加图在 `native_labels/export/preview/`，每路视频在 `native_labels/export/videos/`。

新增四目外参恢复、质量拒绝、CLI 参数传递和候选对齐拦截，与既有原图标签几何和合并测试共 12 项通过。此次未重新训练 WiLoR；之前两目训练对照的结果不能外推为四目训练提升。


## 历史兼容入口

原有 `prepare_nokov21_mano_input.py` 和 `export_multiview_wilor_training_dataset.py` 保留历史矫正路线兼容。旧导出命令若传 `--image-space native_raw`，会转发到新原图导出器；旧的 `--rectification` 参数不会被读取。当前原图统一入口改用 `prepare_native_nokov_mano_input.py` 与 `export_native_wilor_training_dataset.py`，不再调用旧矫正准备器或读取矫正文件。

旧单手原始采集说明见 [采集与处理说明](../../nokov_ego_hand_validation/docs/capture_and_process_zh.md)；旧入口仍为单左手；新的双手原始录制使用本文第 2 节。

## 24 点共享附着位置试拟合（20261112，实验结果）

已接入用户确认的 marker 贴点区域，使用完整 NOKOV 24 点进行小段多帧附着标定。每手 80 帧标定，另检查左手 20 帧、右手 24 帧；固定基线手型，法线偏移为 0。手指与 HandOffset 使用共享三角形重心坐标，三个腕部点使用低权重的腕部局部坐标。

当前仅生成实验模型与原图／基线／新方案的四路对照，尚未证明标签贴合改善，不替换基线，也未接入默认全量导出。指尖检查仅在本次细化中留出；基线初始化本身曾使用这些点，不能作为严格独立精度评估。

实现、完整限制、实验指标与重跑命令见 [24 点试拟合说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/marker24_pilot/README.md)。当前查看器：http://127.0.0.1:8766/marker24_pilot/ 。

### 24 点实验的手背方向修正

旧版把当前 canonical MANO 的正叉积半空间误当作手背，实际选到了相反表面，导致网格被向外推。已将手背区域修正为负半空间，并增加基于确认手背照片的法线朝向检查。左右手旧候选均被检查拒绝，修正版通过。旧版及其关闭腕部约束实验不能用于排除该方向错误。

新页面：http://127.0.0.1:8766/marker24_fixed/ ，标题 `DORSAL-FIX-1`。已重新生成双手及四路视频，仍未替换基线。完整证据与限制见试拟合 README。

### 指尖约束与基线手指姿态锁定对照

恢复全部 24 点后，指尖拟合残差降低，但不能据此认定手指姿态正确。针对用户反馈的手指变形，新增显式 `--fit-all-markers --freeze-finger-pose` 试验选项：手型及手指角度保持基线，只允许整体刚体调整与固定附着标定。结果页 http://127.0.0.1:8766/marker24_pose_locked/ ，标题 `POSE-LOCK-1`，可与全24点自由姿态版切换。已验证手指参数和手型完全保持基线，尚未替换正式标签。重跑命令及限制见试拟合 README。


## GenRobot DEX 2.0 参考视频分析与后续路线（20261112）

用户参考视频 `/home/p3/视频/gen.qt`，长 157.5 秒，1280×720、24 fps。已每 8 秒抽帧，并检查 0、32、56 秒的完整分辨率画面；未进行完整音频转录。抽帧证据和来源记录位于 `/home/p3/data_sda1/EGO_glove/wilor_training/output/reference_video_analysis/`。

### 可以确认与不能确认的内容

视频标题为 GenRobot.AI DEX 2.0 Test Cases。24～80 秒附近的抽样帧展示 Ground Truth 与 Dex Tracking 的关键点／骨架对照，画面写有 `<2 mm`。另有手部模型渲染、腕部相机和触觉展示。视频没有给出足够信息把该数值解释为所有 mesh 顶点到实际手套表面的距离，也未说明统计口径、被测点定义、对齐方式和测试分布。

[Dex 2.0 官方产品页](https://www.genrobot.ai/products/dex2)明确介绍侧置外骨骼与 21 个关节编码器，直接记录关节运动。具体角度标定、骨架映射、手型适配及根部定位算法并未在已查资料中完整公开。因此可以确认硬件输入与我们不同，不能声称知道其完整私有求解器。

官方另有 [Gen-HumanEgo / HandTrack 说明](https://www.genrobot.com/blog/gen-human-ego)，明确介绍多视角视觉特征预测 MANO 和 21 个关键点。这是另一条公开的视觉路线，不能据此断言本段 Dex 手套视频的模型也是 MANO。

### 对当前项目的意义

当前录制文件清单包含 EGO 四路视频、相机标定、设备 IMU 以及 NOKOV 点／刚体数据，未发现手指关节编码器流。NOKOV 提供精确三维球心，但球心不等于内部关节中心。精确测量球心不能自动解决骨长、关节轴、贴点偏移和手套形变的识别问题。

当前基线把 21 个球心当作关节点拟合，因此基线手型也可能吸收这种偏差；后续固定 betas 并不能证明手型已经准确。附着点与逐帧整体位姿共同调整时，存在相互补偿的可能。锁定手指只保证局部姿态不再改变，不能保证刚体位姿改善。sync 632 右手旋转约 6.93 度、腕部移动约 21.5 mm 的退化正说明现有 marker 目标缺乏足够的独立图像验收约束。方向错误已修正，但模型／数据对应关系仍需要标定。

[MANO](https://mano.is.tue.mpg.de/)可表达手型与姿态相关形变，但不能预先假定固定的基线手型与实际戴手套表面完全一致。[MoSh](https://mosh.is.tue.mpg.de/)提供了从表面 marker 估计形状、姿态和附着关系的方法论参照，不代表本项目的 24 个点已经足够恢复唯一解，更不能直接保证达到视频的精度。

### 接下来按证据推进的顺序

1. **先分离投影误差与模型误差。** 建立固定审核集，覆盖张手、侧视、弯指、拇指对指、双手交叠、快速运动，并包含 sync 632、576、584、616。直接投影原始 NOKOV 球心到原图，对照实际图像中的 marker 中心；不经过 MANO。若这一层仍有稳定偏移，应先排查时间、外参、畸变与点对应，不能让手部模型移动去补偿相机误差。
2. **建立独立图像证据。** 原有 48 点确认大多未改动投影预填坐标，不能作为独立像素真值。用清晰帧中的实际球心、可见指尖、手掌与手指边界作核对。遮挡、运动模糊及左右手无法分清的边界要标记无效，不能当作拟合目标。
3. **标定个人手型与附着关系。** 先用现有清晰、手势多样的帧评估能否稳定估计掌宽、指长与贴点；如果不同动作组给出互相冲突的参数，再补专门标定录制。不要在每个业务帧上重新自由估计手型，也不要把所有 marker 强制放到零厚度的裸手表面。当前按用户要求保留法线偏移 0，任何新偏移模型应作为独立、可解释的实验，不悄悄恢复 2 mm 扣除。
4. **校准后再逐帧追踪。** 固定经过验证的手型和附着位置，综合 NOKOV 三维观测、可靠原图证据、关节范围及时间连续性。四路真实 K/KB D/外参用于投影，原图不去畸变。可见性与双手遮挡需单独处理，不能对每路所有边界一视同仁。
5. **独立验收再导出。** 分别报告图像球心重投影、可见轮廓距离、三维 marker 拟合残差、姿态与根部改动及时间稳定性；不能把其中一种指标称作整体三维真值精度。阈值依据固定审核集的测量误差和基线分布设定，而不是反复试常数。对比不同方法必须使用同一批帧，保留退化帧统计，不能通过只显示成功帧得出提升结论。

### 当前决策

保留基线作为正式输出，POSE-LOCK 等仍是诊断实验。优先开展固定审核集与投影／手型／附着误差分解，暂不继续增加无独立证据支撑的优化参数。工程上可以争取更稳定的轮廓贴合，但在没有独立验证和完整硬件条件说明时，不承诺复现视频 `<2 mm`。


## 原始 NOKOV 球心投影审核（20261112，RAW-AUDIT-1）

已完成四相机 × 12 时刻的原图审核集和交互核对工具。此步骤绕过 MANO，不改标定或基线。自动候选仍需人工核对，不能以候选距离证明标定合格。

详细结果与操作：[原始球心审核说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/native_marker_audit/README_zh.md)。

入口：http://127.0.0.1:8766/raw_marker_audit/ 。下一步：核对清晰真实 marker 中心并导出审核 JSON，再分析投影误差是否具有系统性，之后决定模型优化方向。


## 个人手型＋固定附着标定第一阶段（20261112）

已实现显式启用的分阶段标定，输出在 `wilor_training/output/native_kb_training_20261111/personal_calibration_v2`。完成两组时间块稳定性检查和固定手型对照。**当前只使用 marker，尚无独立图像轮廓／指尖约束，不批准为正式真值标签，也没有修改默认导出。**

完整实验结果、限制与命令：[个人手型标定说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/personal_hand_calibration/README_zh.md)。查看入口：http://127.0.0.1:8766/personal_calibration/ 。


## 原图轮廓约束实验（20261112，PERSONAL-3）

已接入四路 KB 原图轮廓候选，完成仅调整共享手型的受控实验。326 份候选中 233 份标定、93 份验证；留出图像轮廓距离改善较小，尚未通过个人手型或正式标签验收。可见指尖身份约束尚未接入。

详细方法、数值和复现命令：[原图轮廓实验](/home/p3/data_sda1/EGO_glove/wilor_training/tools/personal_hand_calibration/CONTOUR_STAGE_zh.md)。入口：http://127.0.0.1:8766/personal_contour_calibration/ 。


## 固定手型后的参数分项对照（20261112）

完成整手位姿、手指姿态、固定面片内附着参数和“附着重标定后再拟合手指”的四组对照；统一外轮廓定义，并重新渲染留出相机视角评分。改善有限，仍未通过手套贴合验收，不接入生产标签。

[完整结果、限制与命令](/home/p3/data_sda1/EGO_glove/wilor_training/tools/personal_hand_calibration/ABLATION_STAGE_zh.md)。页面：http://127.0.0.1:8766/personal_ablation/ 。


## 人工调整个人手型与目标 mesh（MANUAL-SHAPE-1）

已提供 PERSONAL-3 起点的参数化调整工具：四路原图同步投影、共享手型尺寸／betas、当前帧整体与手指姿态、可旋转三维预览，以及多视角确认和保存。保存同时计算局部 marker 剩余偏差，作为后续附着标定证据，不自动视为正式真值。

入口：http://127.0.0.1:8767/ 。[操作与参数说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/manual_shape_tuner/README_zh.md)。保存结果位于 `wilor_training/output/native_kb_training_20261111/manual_shape_tuning/`，`latest.json` 指向最新记录。

## 显式接入完整执行链路（24 marker＋共享手型＋四目轮廓）

原始录制入口新增 `--optimization-config`，读取 `personal_marker24_optimization.enabled`。关闭时不运行新增阶段；开启时先生成基线，再调用 `run_personal_marker24_pipeline.py`，以统一 context 传递本次 dataset、accepted alignment、baseline fit、placement 和手背参考帧。各阶段从上一阶段的实际输出读取，不再从固定的 PERSONAL-2／3 实验目录取输入。

左右手各自共享一套 betas 和 marker 附着关系。marker 阶段按帧拟合动作；四目轮廓阶段当前仅修正共享手型。`sampled_sequence` 对所有合格采样帧估计动作，`holdout` 保留小片段实验语义。导出转换只包含实际拟合帧，并重新展开含均值的 MANO 完整姿态，经原生 KB 导出器和 MANO 重建检查后生成实验标签。merge 保留实验状态，不自动当作已验收真值。

`diagnostics: true` 另外运行四组参数对照、统一外轮廓和重新渲染评分；结果用于诊断，不自动按分数选模型。旧 `--joint` 保持独立，不能与新路线混用。

[干净执行命令和配置说明](EGO_NOKOV_PERSONAL_MARKER24_COMMANDS_zh.md)。配置示例的手背参考帧和时间留出窗口对应当前 20261111 录制；新录制需显式指定适当参考和窗口。

执行链路验证：开关／参数校验测试 10 项、原生标签／合并回归测试 10 项均通过；人工调整工具回归 13 项通过。现有 20261111 数据完成两轮低迭代端到端检查：包含可选诊断的片段模式，以及全体合格采样帧模式。后者左右手各拟合 194 帧，输出 483 张原图、966 份标签和四路标签视频；逐份标签验证每只手 betas 一致。结果位于 `wilor_training/output/native_kb_training_20261111/explicit_pipeline_sampled_smoke_v2`。这些是执行链路验证，不作为贴合精度验收。


## 同时修正逐帧动作与共享附着（JOINT-PA-1）

新增显式 `joint_pose_attachments` 配置，原配置默认关闭。使用 `configs/native_personal_marker24_joint.json` 可在分阶段结果后，固定共享 betas，同时优化逐帧旋转、平移、手指姿态和共享表面／腕点附着。四目轮廓、NOKOV24 和时序项共同约束；额外输出相机留出实验用于检查。本次有 127 个左手、134 个右手采样帧参与，未参与帧保留原动作并标记。

[结果与完整命令](/home/p3/data_sda1/EGO_glove/wilor_training/tools/joint_pose_attachment/README_zh.md)。修正前后：http://127.0.0.1:8766/joint_pose_attachment/ 。实际导出标签：http://127.0.0.1:8771/ 。


## 实际轮廓输入核查（CONTOUR-AUDIT-1）

入口：http://127.0.0.1:8766/contour_audit/ 。紫色为实际优化点，绿色为从原始视频精确重放的分割。829 份接受观测逐点核验一致。**已发现 camera0 / sync632 右手连到背景横梁，错误边缘实际参与了求解；此前轮廓指标下降不足以证明真实贴合改善，应先修复观测再评估。**

[操作、来源和重放命令](/home/p3/data_sda1/EGO_glove/wilor_training/tools/joint_pose_attachment/CONTOUR_AUDIT_zh.md)。

## 优质关键帧标定后固定手型与附着（KEYFRAME-CAL-3）

完成严格分割质量筛选，左右手各选 16 个至少两目通过的关键时刻，仅用这 80 份视图估计共享手型／附着。其余 178 个采样时刻／手固定共享参数，用 NOKOV24 驱动动作。在 190 份未参与标定图像上，重新渲染的轮廓距离相对原始基线下降左手 19.5%、右手 26.9%；两组独立标定的中性手型顶点差异中位数为 0.33 mm、0.75 mm。指标依赖自动分割，困难接触动作未通过独立真值验收，不替换默认路线或正式训练标签。

[干净复现命令、结果位置、原生 WiLoR 对比限制](/home/p3/data_sda1/EGO_glove/wilor_training/tools/keyframe_calibration/README_zh.md)。样本：http://127.0.0.1:8766/quality_keyframes/ ；mesh：http://127.0.0.1:8766/keyframe_calibration/ 。

## 完整掩膜驱动共享手型（FULL-MASK-1）

已完成仅共享 betas 和标定帧受限动作两组对照，使用真实 KB 原图投影、完整区域 IoU 和像素距离项。纯手型组在未拟合图像上，IoU 相对上一轮提高左手 2.39、右手 1.28 个百分点，效果有限；固定附着下的 marker 残差有所上升，尚未批准为最终真值。不改变默认导出路线。

[执行命令、验证范围与结果位置](/home/p3/data_sda1/EGO_glove/wilor_training/tools/full_silhouette/README_zh.md)。交互对照：http://127.0.0.1:8766/full_silhouette/review/ ；四路近景：http://127.0.0.1:8766/full_silhouette/closeups/ 。

## NOKOV 手部骨架源数据核查

确认本次数据每手除24个marker外，还保存26段SDK骨架位姿。完成6077帧骨长与旋转链检查和四目8时刻原图核对；SDK骨架具有动作先验价值，但节点原点不应直接当真实内部关节，五个Finger0原点重合、Hand根与WristM相差约21～26mm。尚未改动MANO拟合或标签。

[检查入口、结果与复现命令](/home/p3/data_sda1/EGO_glove/wilor_training/tools/nokov_skeleton_audit/README_zh.md)。查看：http://127.0.0.1:8766/nokov_skeleton/review/ 。

## SDK 骨架按已补偿内部偏移处理的标签对照

按用户指定假设完成两种腕根映射的固定手型 MANO 动作拟合，未额外减2mm。Hand根输出458张图/916份双手标签，Finger0公共根输出519张图/1038份标签，全部通过格式、MANO重建和KB投影检查。但同组可靠分割上的IoU相对当前标签下降约5个百分点，保留实验状态，不替换正式标签。

[完整方法、假设、结果和命令](/home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_skeleton_fit/README_zh.md)。查看：http://127.0.0.1:8766/sdk_skeleton_labels/review/ 。


## SDK 方向辅助及 Finger0 参考腕根（SDK-DIRECTION-1）

用户选择 Finger0 公共根为优先腕根候选，原来两根映射页面已默认选 Finger0。本轮先从6077帧记录恢复SDK骨段局部偏移及候选marker相对关系，再用每手16个优质时刻校准15条指骨方向。映射来自录制数据及当前MANO初值，未取得原始NOKOV模板，不等于确认了内部关节定义。Finger0仅作参考原点，Hand提供掌部方向。

固定betas、marker附着和逐帧根平移，以相同的小幅动作自由度比较marker-only与marker+SDK方向辅助。后者相对marker-only在未参与标定的190份图像上IoU增加左0.39／右0.07个百分点；相对当前标签为左−0.53／右+0.43个百分点，尚无整体替换依据。没有SDK绝对关节位置／骨长硬目标，也未额外减2mm。每组导出505张原图／1010份双手实验标签，执行原生格式和MANO重建检查；保持未批准训练状态。

[方法、命令、指标和完整产物](/home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_direction_fit/README_zh.md)。原图对照：http://127.0.0.1:8766/sdk_direction/review/ ；视频：http://127.0.0.1:8766/sdk_direction/closeups_direction/ 。
