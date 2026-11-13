# 联合重拟合（joint_refit）

**现在的入口**：`bash scripts/export_ego_nokov_labels.sh --recording <录制> --optimization-config configs/native_joint_refit.json`（在 `ego_hand_system` 下运行）。它先跑基线，再调用 `scripts/run_joint_refit_recording.py`，对每段录制、每只录到的手单独标定，然后拟合全部时刻、导出并验收。命令、输出位置、续跑和汇总训练集见 [docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md](../../docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)。

下面的 `run_joint_refit_labels.sh` 是旧入口，只能用于 192916，而且依赖那段录制已生成的中间结果。保留它只为复现 v1～v6 的结果。

## 历史记录：联合重拟合（joint_refit_v4：Finger0 腕根＋贴点修正＋逐帧对齐修正）

入口（在 `ego_hand_system` 下）：`bash scripts/run_joint_refit_labels.sh [输出目录]`。本轮结果：`wilor_training/output/native_kb_training_20261111/joint_refit_v4`（`v3` 无对齐修正，`v2` 只有腕根锚点，`v1` 三者都没有），标签在其 `final/export/merged`。

代码位置：本目录 `ego_hand_system/scripts/joint_refit/`（`fit.py` 拟合、`export.py` 导出与检查、`render_video.py` 对照视频、`viz.py` 单帧诊断）。与 `run_personal_marker24_pipeline.py` 相同，仍复用 `wilor_training/tools/` 下的 `personal_hand_calibration`、`full_silhouette`、`marker_placement` 及 nvdiffrast 运行时；旧入口 `wilor_training/tools/joint_refit/run.sh` 只做转发。

## 整段录制与自动验收（v5_dense）
- 入口脚本在采样时刻标定之后，继续执行：`build_dense_source.py`（全部1612个时刻的输入）→ `image_alignment.py`（逐帧对齐修正）→ `dense_drive.py`（冻结共享参数，全时刻拟合＋验收）→ `export.py` → `verify_export.py` → `render_video.py`。输出在 `<结果>/dense/`，标签在 `<结果>/dense/fit/final/export/merged`。
- 验收门槛（每只手每个时刻）：对齐已由图像修正；手指 marker 不少于19个；marker 残差中位≤2.5mm 且最大≤8mm。不合格的不写入，原因在 `dense/fit/{left,right}_rejected.json`。
- 本轮结果（`joint_refit_v5_dense`）：左手 1349/1612、右手 1348/1612 通过；剔除中 213 个时刻是对齐未修正，51／52 个是 marker 残差。导出 3423 张图、6846 份标签（要求双手同时合格且该视图21个关节完整），格式与 MANO 重建检查通过。
- `verify_export.py`：随机抽1491份导出标签，把朝向相机的 marker 用标签自带相机参数投到标签自带的图上，与图中亮点比最近距离：拟合结果中位2.65px，未修正的 NOKOV 为4.91px。左手（3～4px）差于右手 camera0（1.3px）；camera2 右手 p90 为18px。结果在 `export/image_evidence_check.json`。
- 对指：通过验收的帧里拇指与其余四指指尖靠近时 mesh 无穿透（最大1.3mm），间隙中位2～4mm，未加额外约束。
- 仍为 `approved_for_training=false`：未经人工看视频验收，未做训练验证。

## 做法
- **阶段A 标定**（`calib/`）：训练帧上同时优化每手共享 betas、21个表面 marker 的面片局部三维偏移、腕部 marker 局部位置，以及逐帧姿态／朝向／平移；约束为 NOKOV24 marker 加优质（accepted）多目轮廓。
- **阶段B 仅marker驱动**（`drive/`）：冻结共享参数，全部194帧只用 marker 拟合姿态。
- **阶段C 最终**（`final/`）：冻结共享参数，marker 加全部优质轮廓拟合；这是导出的标签。
- 姿态先验为 MANO 扫描姿态分布的马氏距离，不再锚定旧姿态。旧姿态的 DIP 关节主要变化方向是绕骨轴扭转，不符合解剖。

## 流程（`run_joint_refit_labels.sh`）
1. `symmetrize_attachments.py` → `attachments/`：贴点修正。
2. `image_alignment.py` → `aligned/`：逐帧对齐修正。
3. `fit.py` → `calib/`、`drive/`、`final/`：标定、仅 marker 驱动、最终（加轮廓）。
4. `export.py` → `final/export/{merged,viewer,logs}`：WiLoR 原图标签、查看器、检查。
5. `render_video.py` → `closeups_final/`：四路对照视频。

## 逐帧对齐修正（v4，`image_alignment.py`）
- 诊断：NOKOV marker 投影与原图中的 marker 亮点相差中位约5像素，逐帧不同，但同一时刻左右手一致（相关0.84／0.94）、各相机一致，是动捕到相机换算的逐帧误差（约6～7mm、1.4°），不是手上 marker 的问题；时间偏移只能解释很小一部分（约9ms）。
- 处理：每个时刻在四路原始视频帧上检测亮点，按视图投票匹配两只手的 marker，解一个六自由度刚性修正，修正 `target24` 与 Finger0 目标；原值保留在 `target24_uncorrected`。
- 结果：200个时刻中172个得到修正，投影误差 rms 中位 5.8→1.5 像素；留出一路相机检验 4.5→1.4 像素。其余28个时刻未修正（`image_alignment_corrected=false`），仍按原对齐拟合并导出。
- 效果：右手仅 marker 帧 IoU 85.4%→86.9%，最终 90.9%→91.7%；左手基本不变。左手 sync664 拇指已贴上。

## 贴点修正（v3，`symmetrize_attachments.py`）
- 诊断：两只手套贴法相同，但自动估计的贴点有5个左右手相差11～15mm，其中一侧落在手指侧面（Thumb3、Ring2 左手；Index3、Middle2 右手；Pinky2 两侧）。同一手指上的 marker 近似共线，指节可绕连线滚动，marker 残差看不出这个错误。
- 处理：左右贴点相差超过8mm 时统一采用更靠手背的一侧；两侧都不在手背则落到同一指节同一轴向位置最靠手背的面片。HandOffset 不处理。改动记录在结果目录 `attachments/attachment_changes.json`。
- 效果：左手仅 marker 留出帧 IoU 84.2%→87.2%，右手无变化（85.8%→85.4%）。

## 腕根（v2）
- 腕根锚点改为 SDK Finger0 公共根：每手标定一个固定的掌内局部偏移（左约7mm、右约11mm），逐帧让 MANO 腕关节加该偏移贴合 Finger0。
- 袖口上的三个腕部 marker（WristM／WristIn／WristOut）权重设为0：它们随袖口相对手掌滑动（掌系内标准差约4～9mm），而 Finger0 只有约1～2mm。
- 四组对比（偏移自由／为零 × 保留／去掉腕部 marker）在 `joint_refit_v2_root_ablation`；偏移强制为零时 IoU 略降，故采用自由偏移。
- `fit.py` 不传 `--skeleton` 即为 v1 行为。

## 验证设置
- 每手每4个有优质轮廓的时刻留出1个，阶段A完全不使用；另从多目训练帧里每3帧留出1个视图。
- `summary.json` 记录 before／calib／drive／final 的 IoU 与 marker 残差。IoU 评分区域与 full_silhouette 相同（固定腕平面远端）。

## 已知限制
- 只有 marker、没有优质轮廓的帧，改善有限（留出帧 IoU 约 +1～2 个百分点）。表面 marker 模型约1mm 的误差对应几个像素的轮廓差。
- betas 有分量到达 ±3 上限；多数法向偏移贴近下限 −2mm：MANO 手型空间不足以表达手套的宽厚。
- 腕／袖口处 mesh 仍会超出手套边缘（MANO 手腕截面比手套袖口短而窄）。
- 拇指在部分帧明显偏离（如左手 sync664）；对指与遮挡未处理。
- 所有输出 `approved_for_training=false`。
