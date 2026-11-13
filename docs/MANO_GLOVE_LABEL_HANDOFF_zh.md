# 手套 MANO 真值标签优化：AI 交接（2026-11-12）

## 目标与当前结论

目标是利用设备四目原图与 NOKOV，让 MANO mesh 稳定贴合真实左右手手套，输出可用于 WiLoR 训练的参数和标签；尤其要解决手指姿态、对指、遮挡及腕部偏差。YOLO 与手工左右手框由用户负责。用户不希望逐点点击或困难的手动调参，希望自动标定后直接看视频复核。

**已有可运行的改进和实验，但尚未得到可确认替换原路线的最终真值。** NOKOV 手势模型能支持 MANO 参数拟合，SDK 节点的解剖定义及 marker→内部关节偏移仍未核实。用户最新明确偏好 **SDK Finger0 公共根**，应保留为优先腕根候选；不能据此认定它就是真实内部腕关节。

## 已完成及效果

| 阶段 | 已做内容 | 结论／限制 |
|---|---|---|
| 原图路线与显式开关 | 原图双手标签、真实 KB 投影、WiLoR 标签检查与查看器；优化由配置开启 | 后续独立实验没有自动替换默认导出 |
| 24 marker 附着与个人手型 | 左右手各共享一套 betas；表面点用面片、重心坐标、有效法向偏移，腕点用局部位置 | 已估计，不等于最终个人手型／附着关系已确认 |
| 优质关键帧标定 V3 | 筛出270份可靠视图，每手选16个多目时刻、共80份图像标定；固定共享参数后驱动动作 | 190份未参与标定的图像上，轮廓距离比原始 NOKOV21 基线下降左19.5%、右26.9%；腕点和困难动作仍有偏差 |
| 完整轮廓共享手型 | 在 V3 上只优化共享 betas，保留动作／附着 | 当前后续实验的比较起点为 `full_silhouette_v1/shape_only`，仍非最终真值 |
| SDK21 位置拟合 | 假定 SDK 已补偿内部偏移，比较 Hand 与 Finger0 腕根 | SDK残差下降，但轮廓IoU整体下降约5个百分点；用户更喜欢其中 Finger0 的视觉结果，该偏好已保留 |
| SDK方向辅助 V1 | 同自由度比较仅marker重拟合、marker＋弱SDK方向辅助；固定手型／附着／根平移 | 增益小，未全面超过当前标签；不是上一组 Finger0 位置拟合的继续迭代 |

最后一组采用同一190份可靠图像、同一腕部评分区域：

| 手 | 当前共享手型标签 IoU | 仅marker重拟合 | marker＋SDK方向 |
|---|---:|---:|---:|
| 左 | 79.74% | 78.82% | 79.21% |
| 右 | 81.77% | 82.13% | 82.20% |

SDK方向相对marker对照改善左0.39／右0.07个百分点；相对当前标签为左−0.53／右+0.43个百分点。指标依赖自动分割，不是独立三维真值；对应 marker／SDK 数据仍参与动作拟合，不能称为全数据留出。

## 接手必须知道的实现事实

- marker直接贴手套，跨录制位置保持固定；球直径4mm。**没有统一减2mm**；已有表面法向有效偏移在0～4mm内估计，不能当作实测球半径或手套厚度再次扣除。
- 当前录制每手有24个marker、26个SDK骨段，6077个SDK时刻。五个Finger0原点重合，不是五个MCP。SDK Hand根距离WristM约21～26mm；SDK指节点整体靠近表面marker，不能仅凭命名认定为内部关节。
- 最新 `profile/mapping.json` 从录制位姿恢复骨段局部偏移、估计候选marker局部关系；**未取得原始NOKOV模板，不代表已读出或确认原配置的内部偏移**。
- 最新方向实验从 `shape_only` 初始化；每手16个优质时刻校准15条指骨方向，仅弱约束方向及掌部旋转，不约束SDK绝对关节位置或骨长。Finger0只作参考根；刚性掌部方向来自SDK Hand，不能直接用会运动的Thumb0方向替代。
- 最新两组每手拟合194个实际采样时刻；姿态增量每指范数≤0.10rad，整体旋转增量≤0.035rad，根平移及共享参数冻结。**这不是每帧开放所有参数的四目联合优化**。
- 几何单位米；内部两手均使用MANO_RIGHT规范，物理左手投影前恢复X镜像。SDK四元数保留世界坐标，必须经头部刚体和相机外参转换；不能直接复制给MANO。
- 本环境45维 `hand_pose_pca` 实际是加MANO均值前的轴角分量，模型 `use_pca=False`；导出完整姿态必须正确加均值，并用MANO重建验证。
- 图像始终保留原始畸变，投影使用真实K、KB畸变D及外参。原生WiLoR的虚拟相机平移不能直接当真实相机三维位置。
- 所有新候选保持 `approved_for_training=false`。格式／参数重建通过仅证明内部一致，不证明服帖；本轮没有训练新WiLoR权重。原生WiLoR旧合并视频与192916录制也不是同一批帧，尚无严格同帧排名。

## 代码、输入与结果定位

以下路径缩写只用于本文，均指同一台设备：

```text
BASE=/home/p3/data_sda1/EGO_glove
PROJECT=$BASE/ego_hand_system
TRAIN=$BASE/wilor_training
EXP=$TRAIN/output/native_kb_training_20261111
Python=/home/p3/miniforge3/envs/ego-hand/bin/python
```

| 路径（按上述缩写展开） | 用途 |
|---|---|
| `$BASE/data/20261111/20261111_nokov_hand/pair_20261111_192916` | 本轮原始EGO＋NOKOV配对录制 |
| `$BASE/glove_marker_placement.json` | 用户确认的贴点；WristIn／WristOut必须保留对应关系 |
| `$EXP/explicit_pipeline_sampled_smoke_v2/context.json` | 真实相机、同步、外参、基线fit及贴点等输入的完整路径；优先读它 |
| `$EXP/keyframe_calibration_v3/selection.json` | 样本质量、左右手标定时刻、分割路径及相机参数 |
| `$EXP/full_silhouette_v1/shape_only` | 当前比较起点，左右手 `*_pilot.npz` 含betas、动作、mesh、附着和marker观测 |
| `$EXP/nokov_skeleton_audit_v1/aligned_skeleton.npz` | 已对齐SDK位置、世界四元数、骨段名称与父子关系 |
| `$EXP/sdk_skeleton_assumed_internal_v1` | 原来的Hand／Finger0位置拟合、标签、四路视频及比较报告 |
| `$EXP/sdk_direction_v1` | 最新方向实验；`profile/`、`marker_only/`、`direction/`、`review/`、`closeups_*/`、`comparison_metrics.json`、`verification.json` |
| `$TRAIN/tools/{keyframe_calibration,full_silhouette,sdk_skeleton_fit,sdk_direction_fit}` | 各阶段代码、run.sh与README；最新重点读 `sdk_direction_fit/build_mapping.py`、`fit.py` |
| `$PROJECT/docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md` | 原始录制导出和实验入口的干净命令 |
| `$PROJECT/docs/EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md` | 历史方法详细说明；必要时再查，避免把旧结论当最新结论 |

最新两组各导出505张原图／1010份双手标签，位于 `$EXP/sdk_direction_v1/{direction,marker_only}/export/merged`。全量通过原生KB标签、MANO重建与双手配对检查；每组四路视频各158采样时刻、3.75帧/秒、约42秒。

复现最新实验（复用已有输入，默认新建输出目录）：

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_direction_fit/run.sh
```

查看：[最新原图对照](http://127.0.0.1:8766/sdk_direction/review/) · [四路视频](http://127.0.0.1:8766/sdk_direction/closeups_direction/) · [原来的Finger0位置结果](http://127.0.0.1:8766/sdk_skeleton_labels/review/)。服务启动方法见 `$TRAIN/tools/sdk_direction_fit/README_zh.md`。

## 未完成与建议接续顺序

1. **先弄清SDK模板语义及腕根偏移。** 若能取得NOKOV原工程／手势模板／关节定义，请向用户索取；现有推断不能替代配置事实。以用户偏好的Finger0为候选，验证到MANO腕关节的固定局部变换，分清原点、静止轴和骨长差异。
2. **固定当前比较起点做困难动作诊断。** 重点检查sync576、632、664、688及前后帧的四目：手指变形来自SDK映射、marker身份／遮挡、附着、手型还是图像同步；不要仅根据一帧或SDK残差选方案。
3. **继续联合标定共享手型与固定附着。** 只用优质多姿态／多目样本，分阶段估计；验证腕点模型、手指长度／宽度和marker残差尾部。MANO betas表达能力有限，不能保证表达所有手套褶皱／厚度／袖口。用户目前不要求接SAM3。
4. **只有证据支持再扩大逐帧动作优化。** SDK方向或可信视觉姿态可作辅助，保留marker-only对照、独立图像评分、时序与接触检查；不要盲目增权、放开全部参数或为分数自动逐帧挑结果。
5. **最终仍待端到端验收与训练对照。** 在同录制同帧对比原生WiLoR及生成标签，验证困难动作、真实相机一致性和时序后，再决定正式导出与训练。当前独立实验尚未整体接入原始录制的一键优化配置；验证有效后再显式接入。

仓库已有大量用户及前序工作改动；接手先看git状态，保留历史产物，新增实验用独立目录。不要将格式通过写成精度验收通过。
