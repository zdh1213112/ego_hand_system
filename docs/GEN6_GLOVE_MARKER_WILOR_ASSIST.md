# GEN 六目手套 Marker 视觉辅助 WiLoR

该功能使用戴手套 EGO 原始图像中可见的反光 marker 球，逐帧辅助 WiLoR 的二维关节点，
再进入现有 Double-Sphere 多相机 RANSAC 融合。它不读取 NOKOV marker/skeleton CSV，
不依赖外部动捕时间同步，也不修改 `calibration/*.json`。

## 算法

每路相机、每一帧按以下顺序处理：

1. 将 BGR 原图转成 HSV，使用 `S < 100`、`V > 160` 得到低饱和高亮掩码；
2. 对掩码做 8 邻域连通域分析，按面积、宽高、长宽比、填充率和圆度保留小圆亮斑；
3. 按 detector 框把亮斑分配给具体的物理手，避免两只近邻手互相使用 marker；
4. 对每个 WiLoR 假设只保留其 20 个手指投影点周围 `45 px` 范围内的候选；
5. 在 `35 px` 内生成粗略二维平移候选，选择能解释最多 marker 的稳健偏移；
6. 平移后以 `13 px` 为门限，用带手指归属约束的匈牙利算法完成一对一匹配，并拒绝
   同一手指上顺序反转的匹配；
7. 3 点且覆盖 2 指时只记录为 `evidence_only`，以很弱的权重帮助多视角选择假设，不改变
   任何关节；至少 8 点、覆盖 4 指且残差达标时才允许修正；
8. 估计偏移最多 `20 px`，真正应用到整手的位移最多 `10 px`；已匹配关节按 `0.15`
   权重靠近 marker 且局部移动最多 `3 px`。骨段长度变化超限时自动取消局部拉动。

WiLoR 索引 `1..20` 是五根手指各 4 点，和手套每根手指的 4 个反光球对应；索引 `0` 是
手腕，没有直接 marker，只有强支持通过全部门控后才对它应用有上限的整体二维平移。
marker 数和残差只作为较弱的假设选择代价；多视角重投影与 RANSAC 仍是主约束。

## 融合后三维解剖与时序修复

逐帧融合先执行逐手的多视角一致性检查：对每只手的三维候选计算各相机重投影残差，
当至少两个其它视角有足够内点且某一路同时满足绝对、相对异常条件时，只剔除该路，
不影响其它手或其它帧。默认阈值为 `35 px`、相对其它视角 `2.5` 倍、最少 `8` 个内点。
因此 camera5 可以在异常帧被动态排除，camera3 也不会因为固定 anchor 身份而强行污染
最终三维；正常帧仍保留全部六目观测。

融合默认再执行一次离线的零相位三维修复，专门处理侧视、遮挡造成的深度尖峰、骨长突变
和短时错误手型，不再增加二维 marker 吸附强度：

1. 按每个物理手的完整时间序列估计稳定骨长，拒绝离手中心过远、相对邻帧跳变或骨长严重
   异常的关节；
2. 不超过 3 帧的内部空洞由前后可靠帧插值；普通插值无法覆盖时，必须有至少两个邻近姿态
   且当前帧至少 3 个可靠掌部锚点，才可用刚体对齐补点；
3. 用前后各至少 2 帧、半径 4 帧的 Theil–Sen 稳健直线预测检查掌部刚体轨迹；预测残差超过
   `70 mm` 且当前帧至少 6 个关节已被拒绝时，才修正整手瞬时位置/深度尖峰；
4. 在每帧自己的掌坐标系内检查手指形状。至少 2 个非掌部关节的归一化残差超过 `0.90` 时，
   只修这些关节并重新施加骨长约束；没有异常的手实例保持原始融合结果；
5. 可靠关节只允许最多 `0.35` 权重、`20 mm` 的弱调整；局部时序修正最多 `40 mm`，整手
   刚体修正最多 `80 mm`，并统一使用 `0.50` 的保守混合；
6. 每个修正关节重新投影到所有参与相机。通常任何相机中的位移不超过 `30 px`，多视角残差
   中位数最多恶化 `6 px`；否则自动缩小或撤销修正。强时序证据也必须通过该重投影保护，
   不再存在绕过门控的特殊覆盖路径；无法由前后两侧支持的物理异常点直接留空。
7. 对前后各至少两帧支持的 1–2 帧异常段使用刚体对齐后的线性插值，限制手指单帧步长
   `120 mm`、掌部步长 `200 mm`，以消除连续短时错误而不过度压制真实快速动作。

这一步可以用 `--anatomy-refinement 0` 关闭。主要可调项为：

```text
--robust-view-consensus 1
--consensus-min-views 3
--consensus-min-inlier-joints 8
--consensus-outlier-median-px 35
--consensus-outlier-ratio 2.5
--anatomy-outlier-window 4
--anatomy-outlier-distance-m 0.10
--anatomy-bone-relative 0.45
--anatomy-smoothing-radius 2
--anatomy-shape-strength 0.55
--anatomy-reliable-adjustment-blend 0.35
--anatomy-max-reliable-adjustment-m 0.020
--anatomy-max-reprojection-regression-px 6
--anatomy-max-reprojection-shift-px 30
--anatomy-temporal-radius 4
--anatomy-temporal-palm-residual-m 0.070
--anatomy-temporal-local-residual 0.90
--anatomy-temporal-blend 0.50
--anatomy-temporal-max-global-adjustment-m 0.08
--anatomy-temporal-max-local-adjustment-m 0.04
--anatomy-temporal-spike-joint-step-m 0.12
--anatomy-temporal-spike-palm-step-m 0.20
--anatomy-temporal-spike-max-spread-m 0.08
```

## 运行

在现有六目命令后增加：

```bash
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/glove_recording.mcap \
  --output /path/to/glove_marker_run \
  --device cuda \
  --max-frames 0 \
  --glove-marker-assist 1
```

旧命令行别名 `--nokov-wilor-assist 1` 仍可用，但实际执行的是本页描述的逐帧 RGB marker
检测，不再表示加载相机 refinement。

启用 marker 后默认使用 `--detector-handedness adaptive`。每帧先尝试严格的 detector
左右身份；仅当严格融合拒绝时，才用多视角几何和 marker 证据重试。这是因为黑色手套
数据中 detector 可能把两只手同时分成同一类别。需要完全保持旧行为时可显式指定
`--detector-handedness strict`。

可调整参数：

```text
--marker-saturation-max 100
--marker-value-min 160
--marker-min-matches 3
--marker-min-finger-groups 2
--marker-global-min-matches 8
--marker-global-min-finger-groups 4
--marker-search-padding-px 45
--marker-bbox-padding-px 12
--marker-seed-distance-px 35
--marker-match-distance-px 13
--marker-max-shift-px 20
--marker-max-applied-shift-px 10
--marker-blend 0.15
--marker-max-local-adjustment-px 3
```

`--marker-min-matches` 是“可作为证据”的门限，真正移动整手由更严格的
`--marker-global-min-matches` 控制。marker 不一定正好位于关节旋转中心，因此不建议把
`--marker-blend` 设得很大。

## 输出与检查

```text
glove_marker_run/
├── normalized_multiview/
├── wilor_multiview/                         # 原始 WiLoR，保持不变
├── wilor_multiview_glove_marker_assisted/
│   ├── summary.json
│   └── camera*/
│       ├── predictions.jsonl
│       ├── summary.json
│       ├── marker_assist_preview.jpg
│       └── marker_evidence_preview.jpg
└── fusion_multiview_glove_marker_assisted/
    ├── accepted.jsonl
    ├── rejected.jsonl
    ├── summary.json
    ├── diagnostic_6view.mp4
    └── final_only_6view.mp4
```

预览图中橙色是原始 WiLoR，青/绿色是通过严格门控后的保守修正，洋红圆圈是已经分配给
该 detector 的 marker。`marker_assist_preview.jpg` 优先保存风险最大的修正案例，
`marker_evidence_preview.jpg` 保存最接近强修正门限的只读证据案例。`summary.json` 还记录
稳定的拒绝原因码、估计/实际偏移和骨长变化。

融合诊断视频中彩色细骨架仍是相机内的 WiLoR/marker 观测，黄色 `ANATOMY L/R N` 是三维
修复后重新投影的骨架，`N` 为该手移动超过 2 mm 的关节数。`accepted.jsonl` 中
`unrefined_joints_base_m` 保留修复前坐标，`anatomy_refinement` 记录逐手修正、拒绝、插值和
位移统计；其中 `temporal_global_corrected`、`temporal_global_reprojection_override` 和
`temporal_local_corrected_joint_count` 可定位强时序修复。融合 `summary.json` 记录骨长误差、
掌坐标形状跳变、掌心步长、三维加速度，以及修复前后多视角重投影分位数。无法通过时序和
重投影双重支持的异常点会保留原结果，并计入 `unrepaired_outlier_count`。

`final_only_6view.mp4` 是交付检查用的简洁版本：每只物理手只画一个最终 detector 框和一次
最终 `joints_base_m`，左手为青色 `FINAL L`，右手为黄色 `FINAL R`；不画原始 WiLoR、
marker 匹配或任何中间骨架。拒绝帧保持无检测结果。主运行脚本默认只生成诊断版；
需要同时生成该最终版时，显式添加 `--final-video`（`--no-video` 会关闭所有视频输出）。
此开关仅控制视频输出，不改变融合或时序修正结果；已有最终版文件不会自动删除。
最终版也可以单独执行：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/normalized_multiview \
  --fusion /path/to/fusion_multiview_glove_marker_assisted \
  --output /path/to/final_only_6view.mp4 \
  --overlay-mode final-only
```

若要检查时序/解剖修正前保存的原始多视角融合骨架，可在上述命令中添加
`--joints-source unrefined`。画面会标记为 `RAW L/R` 和 `RAW FUSED`，建议使用
`final_only_no_temporal_6view.mp4` 作为输出文件名，便于与修正后的最终版对照。

若只看各相机内参与融合的 detector 框和 WiLoR 二维 `joints_2d`，使用
`--overlay-mode wilor-2d-only`。该模式不会绘制 `ANATOMY` 或任何三维融合骨架。

如果只需要更稳定的二维轨迹、不需要手势分类或三维坐标，可使用
`--overlay-mode wilor-2d-stabilized`。该模式只在相机平面内对手指局部形状做轻量
时序滤波，并抑制单帧二维异常点；腕部的整体运动仍保留。默认参数可用以下选项调整：

```text
--wilor-2d-smoothing-radius 1
--wilor-2d-smoothing-strength 0.35
--wilor-2d-outlier-threshold-px 25
--wilor-2d-wrist-outlier-threshold-px 100
```

例如单条序列输出稳定二维视频：

```bash
SEQUENCE_DIR=/path/to/gen6_batch_0905/DAS-Ego_xxx

conda run --no-capture-output -n ego-hand \
python scripts/render_multiview_wilor.py \
  --dataset "$SEQUENCE_DIR/normalized_multiview" \
  --fusion "$SEQUENCE_DIR/fusion_multiview" \
  --output "$SEQUENCE_DIR/fusion_multiview/wilor_2d_stabilized_6view.mp4" \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
  --overlay-mode wilor-2d-stabilized \
  --video-codec auto
```

对已经完成融合的单条序列，只生成相机内 WiLoR 二维骨架视频：

```bash
SEQUENCE_DIR=/path/to/gen6_batch_0905/DAS-Ego_xxx

conda run --no-capture-output -n ego-hand \
python scripts/render_multiview_wilor.py \
  --dataset "$SEQUENCE_DIR/normalized_multiview" \
  --fusion "$SEQUENCE_DIR/fusion_multiview" \
  --output "$SEQUENCE_DIR/fusion_multiview/wilor_2d_only_6view.mp4" \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
  --overlay-mode wilor-2d-only \
  --video-codec auto
```

为批次目录下的所有序列生成该视频：

```bash
BATCH_DIR=/path/to/gen6_batch_0905

for sequence_dir in "$BATCH_DIR"/*/; do
  conda run --no-capture-output -n ego-hand \
  python scripts/render_multiview_wilor.py \
    --dataset "${sequence_dir}normalized_multiview" \
    --fusion "${sequence_dir}fusion_multiview" \
    --output "${sequence_dir}fusion_multiview/wilor_2d_only_6view.mp4" \
    --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
    --overlay-mode wilor-2d-only \
    --video-codec auto
done
```

这两个命令只读取已有的规范化相机视频和融合结果进行可视化，不会重新运行 WiLoR 推理或
多视角融合。若输出路径中已经存在同名文件，渲染器会覆盖它。

## 用六目投票专门优化 cam3 的二维骨架

如果只想改善 `camera3` 这一目的画面，使用 `cam3-optimized-2d`。它不是把 cam3
强行替换成某一个相机，而是先把 cam3 排除在外，对其余可用相机的每个关节用 RANSAC
重投影误差形成独立共识，再把共识投影回 cam3：cam3 与共识一致时保留原始 WiLoR；
cam3 是离群点、且有足够其它相机支持时，才进行保守融合。三目或更稀疏帧使用其它相机
的多数票，只有一个其它相机或其它相机自身投票不一致时回退到原始 cam3。三维交会点只
作为内部几何投票，不会在视频中画三维骨架。
另外，投票修正还要通过时间门控：只有当 cam3 原始关节相对前后 raw 帧已经出现明显
不连续（默认至少 20 px）时才允许修正；raw 本身连续时，即使其它相机能形成几何共识，
也保留 cam3 原始二维结果。这一层是为了避免“其它相机的错配/遮挡共识”把稳定的 cam3
骨架强行拉走。
注意：已有的 `diagnostic_6view.mp4` 不会被这个可视化模式修改，它仍然保留原始诊断叠加；
请查看下面新生成的 `cam3_optimized_2d*.mp4`。

六目画面对比（只改 camera3，其余五目保持原始二维 WiLoR）：

```bash
SEQUENCE_DIR=/path/to/gen6_0907/DAS-Ego_xxx

conda run --no-capture-output -n ego-hand \
python scripts/render_multiview_wilor.py \
  --dataset "$SEQUENCE_DIR/normalized_multiview" \
  --fusion "$SEQUENCE_DIR/fusion_multiview" \
  --output "$SEQUENCE_DIR/fusion_multiview/cam3_optimized_2d_6view.mp4" \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
  --overlay-mode cam3-optimized-2d \
  --cam3-camera camera3 \
  --cam3-min-support 3 \
  --cam3-raw-fallback \
  --video-codec auto
```

只输出优化后的 cam3 画面：

```bash
conda run --no-capture-output -n ego-hand \
python scripts/render_multiview_wilor.py \
  --dataset "$SEQUENCE_DIR/normalized_multiview" \
  --fusion "$SEQUENCE_DIR/fusion_multiview" \
  --output "$SEQUENCE_DIR/fusion_multiview/cam3_optimized_2d.mp4" \
  --cameras camera3 \
  --columns 1 \
  --overlay-mode cam3-optimized-2d \
  --cam3-camera camera3 \
  --cam3-min-support 3 \
  --cam3-raw-fallback \
  --video-codec auto
```

主要门控参数如下；默认值适合当前六目数据：

```text
--cam3-ransac-threshold-px 15
--cam3-min-support 3              # 其它相机达到 4 目时至少 3 票；2/3 目使用多数票
--cam3-correction-threshold-px 18
--cam3-strong-blend 0.80
--cam3-replacement-threshold-px 80
--cam3-smoothing-radius 2
--cam3-smoothing-strength 0.45
--cam3-temporal-radius 2
--cam3-temporal-gate-px 35
--cam3-temporal-margin-px 15
--cam3-min-raw-temporal-residual-px 20
--cam3-raw-fallback              # 全局拒绝帧仍显示 cam3 可用的原始二维检测
```

如果需要更保守（优先保持 cam3 原始稳定性），可提高
`--cam3-min-raw-temporal-residual-px`；如果需要更多跨视角修正，可适当降低它。通常不
建议关闭时间门控，否则几何投票在个别遮挡帧里会重新引入整手跳变。

`--cam3-raw-fallback` 只恢复 cam3 自己仍检测到的手，不会伪造缺失的另一只手，也不会把
拒绝帧重新纳入三维融合；因此这些画面会标记为 `RAW FALLBACK`，适合保持二维视频连续性。

渲染开始时会打印 `hands_with_consensus`、`target_outlier_joints`、`corrected_joints` 等
统计，便于确认算法确实只修正了 cam3 的离群点。

建议先运行 `--max-frames 60`，检查：

- 洋红圈是否稳定落在反光球而不是桌面高光；
- 修正骨架是否比橙色原始骨架更贴合手套；
- marker 辅助后的残差是否稳定低于原始 WiLoR；
- 新旧 fusion 的接受率、重投影中位数/P95 和左右手稳定性。

如果桌面或环境高光过多，可提高 `--marker-value-min`、降低
`--marker-search-padding-px`，或提高 `--marker-min-matches`。如果 marker 暗或运动模糊，
可适当降低亮度阈值或增大匹配距离，但应同时检查预览图，避免错误亮斑拉动关节。
