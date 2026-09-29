# GEN 六目 SAM3 手套掩膜辅助 WiLoR

该功能将外部 `auto-obb-annotator` 中的 SAM3 防护手套分割器接入 GEN 六目流程。SAM3
不直接生成 21 个关节点，而是给每个 YOLO detector 框和 WiLoR 左右手假设附加：

- SAM3 掩膜置信度；
- SAM3 掩膜框与 YOLO 框的 IoU；
- WiLoR 21 点落在手套掩膜内的比例（带 8 px 容差）。

历史版本还启用过一个保守的 2D 修正：只有非腕点跑到掩膜容差之外、且在
`--sam3-max-2d-correction-px` 范围内能找到真实掩膜像素时，才将该点向掩膜边界小幅
拉回。原始点保存在 `wilor_joints_2d`，供 A/B 对照；下游融合使用修正后的
`joints_2d`。腕点不自动拉回，因为手套分割经常不包含袖口/皮肤连接处。

这些证据进入六目融合的假设选择和逐视角一致性门控。只有当至少两个其它视角仍然支持
同一只手时，才会剔除一个明显不符合手套掩膜的视角；掩膜缺失时保持原来的 WiLoR/几何
路径，不会因为遮挡而强行删除结果。因此这是一个保守的检测、二维骨架和三维融合辅助。

## 单条记录

使用新的输出目录运行完整流程（`--max-frames 60` 可先做冒烟测试）：

```bash
cd /home/p3/data_sda1/ego_hand_system
./scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/record.mcap \
  --output /path/to/sam3_run \
  --detector /home/p3/data_sda1/ego_hand_system/models/wilor/detector.pt \
  --conda-env ego-hand \
  --device cuda \
  --max-frames 60 \
  --sam3-glove-assist 1 \
  --sam3-conda-env annotator \
  --sam3-model /home/p3/data_sda1/tool/auto-obb-annotator/models/sam3.pt \
  --sam3-prompt "hand wearing a black protective work glove"
```

SAM3 默认使用 **YOLO 框 + 文本提示**，比整幅图的文本搜索更适合当前 egocentric 画面。
如需只测试文本提示，可添加 `--sam3-text-only 1`。不同手套外观应替换
`--sam3-prompt`，但提示词最好同时包含 `hand` 和 `glove`。

需要注意：在戴黑色手套的实测记录上，SAM3 分割很稳定，但把 2D 点拉向掩膜并没有改善
WiLoR 的关节位置，反而会增加三角化输入的变化。因此当前主脚本默认只使用 SAM3 做门控，
不修改 2D：`--sam3-refine-2d 0`。如需复现实验修正，可显式打开
`--sam3-refine-2d 1 --sam3-max-2d-correction-px 24
--sam3-2d-correction-blend 0.75 --sam3-max-2d-bone-change-ratio 0.55`。如只想验证掩膜而不改 2D 点，可加
`--sam3-refine-2d 0`；如需更保守，可降低 correction blend。

输出结构：

```text
sam3_run/
├── wilor_multiview/                         # 原始 WiLoR，不改动
├── wilor_multiview_sam3_assisted/           # 2D 修正 + sam3_assist 字段
│   ├── summary.json
│   └── camera*/predictions.jsonl
└── fusion_multiview_sam3_assisted/
    ├── accepted.jsonl
    ├── rejected.jsonl
    ├── summary.json
    ├── diagnostic_6view.mp4
    ├── sam3_2d_only_6view.mp4               # 只看修正后的相机内 2D
    ├── sam3_segmentation_6view.mp4          # SAM3 手套掩膜叠加（独立渲染）
    ├── final_only_6view.mp4                 # 添加 --final-video 才生成
    └── pose_failure_report*.json             # 异常帧/时序 A-B 检查报告
```

`fusion_multiview_sam3_assisted/summary.json` 中的
`sam3_matched_view_count`、`sam3_accepted_view_count` 和
`sam3_rejected_matched_view_count` 用于检查掩膜门控是否实际参与；
`wilor_multiview_sam3_assisted/summary.json` 中的 `corrected_hypotheses`、
`corrected_joints` 用于检查 2D 修正是否实际触发。每个
`accepted.jsonl -> hands -> views -> cameraX -> sam3_assist` 保存该视角的详细分数。
原始输出与 SAM3 输出应使用相同的 normalized 数据和帧范围进行 A/B 比较。

当前主脚本默认使用逐关节病态射线拒绝、身份跟踪和保守可见性约束（fusion algorithm
`anchor_guided_condition_rejected_joint_temporal_v27_gapaware_cam2_sam3_angle_weighted`）：

- 先删除同一物理框被 detector 重复输出的高度重叠框（不论左右类别），避免同一只手被分配成两只；默认 IoU/包含率阈值为 `0.95`；
- 用至少 5 目、掌部至少 3 个内点且重投影中位数不超过 12 px 的帧作为可信锚点；
- 对逐关节病态射线（当前默认硬拒绝条件数 `>=150`，并对 `>=100` 的射线逐步降权）不输出伪深度点，交给邻帧双向轨迹处理；
- 默认关闭 anatomy 形状投影（`--anatomy-refinement 0`）。在弱基线下强行把每帧手指投影到
  全局骨长，会把正确的 2D 形状变成“扭曲的 3D 手”；只有有独立 3D 标注时才建议打开；
- 跨拒绝间隔的身份跟踪窗口默认 60 帧，避免断帧后左右手记录互换；
- 少于 18 个可靠关节时整手暂不输出；相机内腕点出现明显单帧大跳/合并框时整手暂不输出；
- 未能由邻帧确认的连续 3D 分支超过 30 mm/帧时清空新端点，而不是硬裁剪成固定步长；
- 3D 不可信时仍保留 detector 框和相机内 2D 骨架，因此不会用错误深度破坏稳定的 2D 结果。

时序跟踪本身也可能在射线/重投影检查后删掉部分关节，因此导出前会再次检查最终候选：
少于 18 个最终有效关节的整手会被清空，避免残缺骨架在相邻帧之间换形。最终交付渲染还会
默认要求 21 个关节全部有效（`--final-min-valid-joints 21`）；缺一根手指就不画这只 3D 手，
避免把残缺骨架误看成“手指扭曲”。诊断视频默认把
拒绝帧留空；若只想消除 1--2 帧短空档造成的视觉闪烁，可以在渲染时增加
`--interpolate-short-gaps 1`（只生成显示用插值，不修改 `accepted.jsonl`，长空档仍留空）。
如果审查视频必须连续显示，也可以使用 `--interpolate-all-gaps 1`；它只在渲染层对
同一物理手的前后完整姿态做显示预测，画面会标记 `PREDICTED`，不会修改融合 JSONL，
因此不能把该视频中的长缺口姿态当作真实三维测量。默认策略是 `hold`：长缺口内保持最近完整
可信姿态，不制造不存在的手指动作。若需要在缺口中连续过渡，可明确使用
`--all-gap-prediction-mode geodesic`，它按掌部和固定骨段方向过渡。若明确需要旧的线性效果，
再加 `--all-gap-prediction-mode linear`；正式结果不要使用长缺口预测视频，交付建议使用不补缺口的
`final_only_6view.mp4`，或只补 1--2 帧的 `final_only_6view_shortgaps.mp4`。预测层要求
腕点和掌部关键点存在，并允许前后端点各自缺少不同的少量指尖；真实融合仍保持严格的
18 点和可见性门限；最终渲染默认使用更严格的 21 点门限。

`final-only` 的最终框由最终三维姿态投影后的关节范围生成，而不是直接复用某一个相机的
原始 detector 框；这样某一相机漏检或框合并时，不会出现框单独跳走而骨架没有跳的假象。
诊断模式仍保留原始相机框，便于定位 detector 问题。

若某只手在当前帧的射线深度分支失败，但相机内仍有完整二维点，不要把 `geodesic` 补出的
三维姿态当成真实手势。显式增加 `--final-predicted-2d-fallback 1` 时，会对这只手改画当前
相机的二维观测，并标记为 `2D L/R`；只有通过质量门控的完整三维手才画 `FINAL L/R`。但在
黑色手套记录中 native 2D 也可能发生错误关联，因此主脚本默认值为 0，交付优先留空或保持
上一可信姿态；需要 2D 回退时应同时检查 SAM3/检测框是否对应同一只手。

当前推荐链路会在 cam2/cam3 目标相机重投影保护之后再执行一次逐关节双向时序复滤。
这样可以消除保护步骤重新暴露的短深度分支，但不会把持续分支锁成上一帧；没有可信
前后端点时仍保留缺失。`summary.json` 的 `post_temporal_joint_repass` 记录该阶段
及其修正数量。

这些设置对应 `fuse_multiview_wilor_guided.py` 的
`--suppress-duplicate-detector-boxes 1 --duplicate-detector-iou 0.95
--duplicate-detector-containment 0.95`、
`--condition-hard-threshold 150`、`--temporal-3d-guard 0`、
`--joint-temporal-max-step-m 0.025`、`--joint-temporal-min-hand-joints 18`、
`--joint-temporal-excessive-jump-threshold-m 0.060`、
`--post-temporal-branch-threshold-m 0.030`、`--post-temporal-joint-repass 1`。
六目运行时建议同时设置 `--anatomy-protected-cameras camera2 camera3`。详细的修正/清空帧列表及
校正前后掌部步长在 `fusion_multiview*/summary.json` 的 `joint_temporal_guard` 中；旧的
`temporal_3d_guard` 保持关闭，仅用于兼容旧配置。

### v11 逐关节深度连续性与射线降权（默认开启）

仅限制掌心的 v5/v6 保护仍可能让某一根手指在骨长投影后突然改变深度。v11 对每个
WiLoR 关节单独处理：

- 用前向和后向轨迹作为深度先验，再用各相机的 Double-Sphere 射线重投影校正；
- 根据实际射线夹角/条件数自适应降低病态三角化的权重。六目“都检测到”不等于深度
  一定可观测，近平行射线会产生很大的深度不确定性；条件数极高时直接采用双向
  时序预测，不让病态三角化改写关节深度；
- 对整段序列估计每只手的骨长，做有限强度的骨架约束和零相位平滑；
- 最后只修复有前后邻帧支持的短异常分支，不跨越真实 `sync_index` 缺口，也不凭空补回
  被几何门控拒绝的 3D 点；导出前再执行零相位分支修复。25 mm/帧只是识别异常段的阈值，不再把
  每一步硬截断成固定 25 mm，因此真实的快速手势不会被制造成等步长跳变。这一步只
  作用于 3D 导出点，不改相机内 2D 观测。

条件门控后的 NaN 不会被解剖插值重新填回；若一个分支在缺口后没有可信后端且超过
80 mm/帧，系统只清除该分支尾部；所有后处理结束后若仍超过 30 mm/帧，则仅清除新端点，
不会把它投影到错误深度。`summary.json` 中的
`condition_gate_restored_joint_count`、`joint_temporal_guard.boundary_tail_cleared_joint_count`
和 `joint_temporal_guard.excessive_jump_cleared_joint_count` 可用于确认这些保护是否触发。

### v12 整手身份跟踪（默认开启）

左右标签来自 detector，并不保证跨帧持久；双手交叉或遮挡时可能整帧互换。v12 在
解剖和 3D 时序处理之前，用六个相机的框中心/尺度与掌心轨迹做二选一匹配，必要时交换
完整的手记录。这样不会把两只手的 3D 轨迹混合，也不会改动任何相机内的 2D 点或 SAM3
掩膜。`summary.json` 的 `temporal_hand_identity.swapped_frame_count` 可确认是否触发。

主脚本对应参数为 `--temporal-hand-identity 1`、`--temporal-hand-identity-max-gap 60`、
`--temporal-hand-identity-swap-margin 0.25` 和 `--temporal-hand-identity-swap-min-cost 0.90`。

### v13 手指几何一致的最终导出（推荐交付）

如果已有 WiLoR/SAM3 融合结果，不必重新推理。`postprocess_coherent_final.py` 会在原有
`joints_base_m` 上执行一次离线几何稳态处理：用 3D 掌部连续性保持左右物理手身份（不再
仅凭相机框在交叉处换手）、在逐帧掌面坐标系中做零相位平滑、固定整段序列的骨长，并把
单帧过大的手指骨方向变化限制在 35°/帧以内。相机内 `views -> joints_2d` 和 SAM3 证据
完全不改；缺失关节仍保持缺失，不会被“补成”假的观测。

```bash
cd /home/p3/data_sda1/ego_hand_system
python scripts/postprocess_coherent_final.py \
  --source /path/to/fusion_multiview_sam3_assisted \
  --output /path/to/fusion_multiview_sam3_assisted_coherent_final
```

正式测量视频使用严格版（拒绝帧留空）：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/fusion_multiview_sam3_assisted_coherent_final \
  --output /path/to/fusion_multiview_sam3_assisted_coherent_final/final_only_6view.mp4 \
  --overlay-mode final-only --video-codec auto --max-frames 0
```

连续审查版默认使用 `hold`：缺口期间保持最近完整姿态，不会把手指拉成扭曲的中间形态；预测帧会标注
`PREDICTED`。如果确实需要动作连续过渡，再显式使用 `geodesic`。

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/fusion_multiview_sam3_assisted_coherent_final \
  --output /path/to/fusion_multiview_sam3_assisted_coherent_final/final_only_6view_allgaps_geodesic.mp4 \
  --overlay-mode final-only --interpolate-all-gaps 1 \
  --final-min-valid-joints 21 --all-gap-prediction-mode geodesic \
  --video-codec auto --max-frames 0
```

`hold` 是正式交付和保守审查的默认策略；`linear` 仅用于历史 A/B 对照，不作为
交付结果。`coherent_final_report.json` 中的 `finger_direction_rate_limited_count`、
`changed_joint_count` 和 `identity` 字段可用于确认该后处理是否实际执行。

戴手套数据不要把 `fit_mano_sequence.py` 产生的 MANO 时序点直接当作最终三维点：如果
它的多目重投影误差明显高于原始多目射线结果，会把手指拉到错误的深度分支。此时应从未经过
MANO 替换的 fusion 目录导出，再运行上面的几何稳态后处理；六目 2D/SAM3 证据保持不变。

该阶段只改 `joints_base_m`，不会改相机内的 `joints_2d` 或 SAM3 掩膜证据。相机内诊断/SAM3
2D 渲染会额外使用同一套明显单帧跳变保持规则；原始 `wilor-2d-only` 仍保持原始观测。可在
`summary.json` 查看 `joint_temporal_guard`、`corrected_joint_count`、射线条件数自适应
开关、最终连续性上限和保留缺失点计数。若要做未经 v11 的 A/B 对比，显式传入
`--joint-temporal-guard 0`。

### 当前数据的交付原则（2026-09）

在黑色手套记录上，SAM3 掩膜本身可靠，但没有深度信息，不能修复近平行相机射线造成的
错误三维分支。实测中“SAM3 修改 2D”没有优于原始 WiLoR 2D，因此默认只做门控，不改点。
同一 detector 框的两个 handedness 假设也不能同时显示；渲染器会按框 IoU 去重。最终视频默认
要求本相机存在观测（`--final-require-local-view 1`），遮挡相机不再强行投影另一相机的 3D
姿态；最终 2D 回退默认关闭（`EGO_FINAL_PREDICTED_2D_FALLBACK=0`），避免错误的 native
2D 把画面拉到手外面。需要连续审查时，使用 `hold-previous` 只保持最近一次可信姿态，
不要使用 `linear` 或跨长缺口的自由插值。

六目默认锚点现在优先使用 camera0+camera5（约 20 cm 宽基线）；只有选择的相机不含这两个
相机时才退回 camera2+camera3。对当前黑手套数据，宽基线时序分支的锚点三维差异中位数约
7.5 mm，而 camera2+camera3 约 22 mm。最终交付建议使用“逐相机可信 3D + 原生 2D 回退”
渲染器；它不会把一只手的失败传播到另一只手：

```bash
python scripts/render_hybrid_multiview.py \
  --dataset /path/to/run/normalized_multiview \
  --fusion /path/to/run/fusion_multiview_anchor05_condition150_temporal_v1 \
  --predictions /path/to/run/wilor_multiview_sam3_assisted \
  --output /path/to/run/final_hybrid_anchor05_strict_native2d.mp4 \
  --physical-side-swap 0 --min-joint-inlier-views 4 \
  --residual-percentile 75 --residual-threshold-px 12 \
  --wrist-threshold-px 15 --max-condition-number 100 \
  --max-stereo-difference-mm 20 --max-hold-frames 2
```

其中 `physical-side-swap 0` 与 WiLoR 的 `is_right=0/1`（左/右）一致；只有确认整段标签
全局反向时才显式传入 `1`。随后再生成两个对照：

```bash
# 纯相机内 2D（最不容易出现深度扭曲）
python scripts/render_native_2d_multiview.py \
  --dataset /path/to/run/normalized_multiview \
  --predictions /path/to/run/wilor_multiview_sam3_assisted \
  --output /path/to/run/native_wilor_2d_multiview.mp4

# 六目 3D，但只在本相机有证据时绘制；缺口保持上一可信姿态
python scripts/render_multiview_wilor.py \
  --dataset /path/to/run/normalized_multiview \
  --fusion /path/to/run/fusion_multiview_raw2d_condition150_temporal_v1 \
  --output /path/to/run/final_only_6view_local_evidence_hold.mp4 \
  --overlay-mode final-only --final-require-local-view 1 \
  --interpolate-all-gaps 1 --all-gap-prediction-mode hold-previous \
  --final-predicted-2d-fallback 0 --final-min-joint-inlier-views 4
```

当前工程内验证的连续性指标（仅表示时序稳定性，不等同于真实 3D 绝对误差）为 raw WiLoR
2D + 条件数 150 + 逐关节时序：掌部步长 p95 约 9.3 mm、关节步长 p95 约 13.2 mm。
若仍要求遮挡时每一帧都输出真实 3D，下一步不是继续叠加滤波，而是增加相机基线/重新做外参
标定，并用带真实 3D 标注的数据评估；现有六目近共线布局无法从错误 2D 唯一恢复深度。

如果 WiLoR/SAM3 推理已经完成，不必重新推理即可对已有 `accepted.jsonl` 应用上述身份、
稀疏手和相机框跳变保护：

如果检查视频发现从开头开始两只物理手整体左右相反，说明二选一初始化出现了全局 side
歧义，不能用逐帧交换解决。先在复制目录中显式交换物理 side，再做最终显示质量门控：

```bash
PY=/home/p3/miniforge3/envs/ego-hand/bin/python
$PY scripts/swap_physical_hand_sides.py \
  --source /path/to/fusion_multiview_depth_branch_gate_v2 \
  --output /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed
$PY scripts/postprocess_final_quality_gate.py \
  --source /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed \
  --output /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed_quality3_v1 \
  --min-views 3 --min-joints 21
```

该交换只修改 `side` 身份，不修改任何 2D/3D 坐标；质量门控只清除少于三目支持、不完整或
触发大时序全局深度修正的显示用 3D 姿态，原始融合目录和相机内 2D 证据仍保留。开头弱观测
阶段宁可留空，也不把一只手的错误姿态贴到另一只手上。

如果某帧的 3D 手被时序深度门控做过较大的全局分支替换，也应清除该帧的显示 3D。质量门控
默认会清除 `temporal_global_corrected=true` 且残差不小于 35 mm 的手；这些帧仍保留原始
二维观测，后续连续视频可以直接使用相机内 2D 作为安全回退：

```bash
$PY scripts/postprocess_final_quality_gate.py \
  --source /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed \
  --output /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed_quality4_temporal_v1 \
  --min-views 3 --min-joints 21 \
  --clear-temporal-global-correction 1 \
  --max-temporal-global-residual-mm 35
```

```bash
PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/postprocess_fusion_stability.py \
  --source /path/to/fusion/accepted.jsonl \
  --output /path/to/fusion_stability_guarded \
  --identity-max-gap 60 --min-hand-joints 18 \
  --excessive-jump-threshold-m 0.030 --view-wrist-jump-px 180
```

只看修正后的相机内 2D 骨架（不叠加 3D）时：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/sam3_run/fusion_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/sam3_2d_only_6view.mp4 \
  --overlay-mode sam3-2d-only --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

生成最终交付视频时，如果希望 1--2 帧的短拒绝间隔保持连续，可加：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/sam3_run/fusion_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/final_only_6view_continuous.mp4 \
  --overlay-mode final-only --interpolate-short-gaps 1 \
  --short-gap-max-frames 2 --short-gap-max-step-m 0.060 \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

完整连续审查版（长缺口也显示，但明确标记为预测）示例：

```bash
python scripts/render_multiview_wilor.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --fusion /path/to/sam3_run/fusion_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/final_only_6view_predicted.mp4 \
  --overlay-mode final-only --interpolate-all-gaps 1 \
  --all-gap-max-frames 0 --all-gap-max-step-m 0 \
  --final-min-valid-joints 21 --all-gap-prediction-mode hold \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

`geodesic`/`hold` 只用于审查视频连续显示；它们不会把遮挡期间的姿态当作真实测量。
需要研究旧的逐关节线性插值时才使用 `--all-gap-prediction-mode linear`，该模式可能再次
出现手指扭曲。

如果 WiLoR/SAM3 已经完成、只需要重新优化三维手指形状，不必重跑检测。可以对已有
`accepted.jsonl` 执行二维稳态射线重建：它在六目图像中先做短窗口零相位滤波，再按每个
关节重新求解六条标定射线，最后限制 40 mm/帧的孤立深度分支；长拒绝间隔仍保持缺失：

```bash
python scripts/reproject_fusion_temporal.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --source /path/to/sam3_run/fusion_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted_projection_consistency
```

该命令不会改写源目录或相机内二维观测。输出目录可直接按上面的 `final-only` 和
`sam3-2d-only` 渲染命令生成 A/B 视频；`summary.json` 的
`ray_reprojection_temporal_refinement` 记录每个阶段的修正数量和参数。

`diagnostic` 和 `sam3-2d-only` 现在也会使用相机内单帧跳变保持规则；若要保留未经处理的
WiLoR 2D 观测，请使用 `--overlay-mode wilor-2d-only`。

如需直接查看 SAM3 的像素级手套掩膜、YOLO 框、SAM 框和修正后的 2D 骨架，使用独立
渲染器：

```bash
conda run --no-capture-output -n annotator \
  python scripts/render_sam3_glove_multiview.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --predictions /path/to/sam3_run/wilor_multiview \
  --assisted-predictions /path/to/sam3_run/wilor_multiview_sam3_assisted \
  --output /path/to/sam3_run/fusion_multiview_sam3_assisted/sam3_segmentation_6view.mp4 \
  --tool-root /home/p3/data_sda1/tool/auto-obb-annotator \
  --model /home/p3/data_sda1/tool/auto-obb-annotator/models/sam3.pt \
  --prompt "hand wearing a black protective work glove" \
  --device cuda --max-frames 0 \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5
```

主辅助阶段只保存掩膜统计和关节点结果，不保存全分辨率掩膜像素；因此该可视化命令会
按相同提示词重新运行 SAM3，但不会修改任何预测或融合文件。

## 仅用其它五目校正 camera3（离线 A/B）

当 camera3 偶发跳变、而其它视角一致时，可以不重新运行 WiLoR 或 SAM3，直接使用去掉
camera3 的五目融合结果，把可靠的五目 3D 关节投影回 camera3。默认模式只修改
`camera3/predictions.jsonl` 中的 `joints_2d`，原始 `wilor_joints_2d` 和其它五路预测
保持不变；默认要求至少 3 个非 camera3 视角支持该关节，并限制单点最大移动 80 px。

```bash
PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/correct_target_camera_from_leaveout.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --predictions /path/to/sam3_run/wilor_multiview_sam3_assisted \
  --leaveout-fusion /path/to/sam3_run/fusion_cam3_leaveout_test \
  --output /path/to/sam3_run/wilor_multiview_cam3_leaveout_corrected \
  --target-camera camera3 \
  --source-cameras camera0 camera1 camera2 camera4 camera5 \
  --min-source-views 3 --max-frames 0
```

若该记录有明显快速运动，建议加上时序保护，避免单帧五目投影把稳定的 camera3
轨迹拉出新的跳变：

```text
--temporal-guard 1 --temporal-radius 2 --temporal-gate-px 35 \
--temporal-margin-px 15 --min-raw-temporal-residual-px 20
```

`--allow-handedness-relabel 1` 是更激进的可选实验：只有同一 detector 框的另一种
WiLoR 左右手假设至少改善 20 px 且通过 70 px 关联门限时才启用。它可以减少 detector
左右标签错误，但会改变该假设进入严格融合的身份；正式数据建议先用默认关闭状态，或
与 `--temporal-guard 1` 一起做独立 A/B。

再用与原实验相同的六目锚点做严格 A/B：

```bash
PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/fuse_multiview_wilor_guided.py \
  --dataset /path/to/sam3_run/normalized_multiview \
  --predictions /path/to/sam3_run/wilor_multiview_cam3_leaveout_corrected \
  --output /path/to/sam3_run/fusion_multiview_cam3_leaveout_corrected \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
  --anchor-cameras camera2 camera3 --detector-handedness strict \
  --sam3-gate 1 --robust-view-consensus 1 --workers 8 \
  --condition-hard-threshold 150 --anatomy-refinement 0 --max-frames 0
```

随后可按前面的渲染命令生成 `diagnostic_6view.mp4`、`final_only_6view.mp4` 和
`sam3_2d_only_6view.mp4`。应比较两个融合目录的 `summary.json`：接受帧数不应靠放宽
门限增加，重点观察 `reprojection_median_px`、`consensus_rejected_camera_counts.camera3`
和人工抽帧结果。该校正是保守的几何补偿，不保证在五目本身不一致时改善结果。

## 用其它五目优化 camera2（最终推荐链路）

如果最终要交付以 `camera2` 为目标视角的二维骨架，同时希望六目继续提供最稳定的
三维结果，使用下面的三阶段链路。第一步明确不把 camera2 放进参考融合，避免它的
错误观测泄漏到自己的校正基准；第二步只修改 camera2 的二维点；第三步再把修正后的
camera2 和其它五目一起融合。其它五路预测始终原样复制，便于回退和 A/B：

```bash
DATA=/path/to/run
PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/fuse_multiview_wilor_guided.py \
  --dataset "$DATA/normalized_multiview" \
  --predictions "$DATA/wilor_multiview_sam3_assisted" \
  --output "$DATA/fusion_multiview_leaveout_cam2" \
  --cameras camera0 camera1 camera3 camera4 camera5 \
  --anchor-cameras camera3 camera4 --detector-handedness adaptive \
  --sam3-gate 1 --robust-view-consensus 1 --temporal-3d-guard 0 \
  --temporal-3d-guard-min-anchor-cameras 4 \
  --joint-temporal-guard 1 --joint-temporal-max-step-m 0.025 \
  --workers 8 --max-frames 0

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/correct_target_camera_from_leaveout.py \
  --dataset "$DATA/normalized_multiview" \
  --predictions "$DATA/wilor_multiview_sam3_assisted" \
  --leaveout-fusion "$DATA/fusion_multiview_leaveout_cam2" \
  --output "$DATA/wilor_multiview_cam2_optimized" \
  --target-camera camera2 \
  --source-cameras camera0 camera1 camera3 camera4 camera5 \
  --min-source-views 3 --association-threshold-px 70 \
  --correction-threshold-px 18 --wrist-correction-threshold-px 30 \
  --replacement-threshold-px 70 --blend 0.8 --max-shift-px 80 \
  --temporal-guard 1 --temporal-radius 2 --temporal-gate-px 35 \
  --temporal-margin-px 15 --min-raw-temporal-residual-px 20

PYTHONPATH=scripts conda run --no-capture-output -n ego-hand \
  python scripts/fuse_multiview_wilor_guided.py \
  --dataset "$DATA/normalized_multiview" \
  --predictions "$DATA/wilor_multiview_cam2_optimized" \
  --output "$DATA/fusion_multiview_cam2_optimized" \
  --cameras camera0 camera1 camera2 camera3 camera4 camera5 \
  --anchor-cameras camera2 camera3 --detector-handedness adaptive \
  --sam3-gate 1 --robust-view-consensus 1 --condition-hard-threshold 150 \
  --min-joint-inlier-views 3 --anatomy-refinement 0 --anatomy-protected-cameras camera2 \
  --anatomy-protected-camera-shift-px 35 \
  --temporal-3d-guard 0 \
  --temporal-3d-guard-min-anchor-cameras 5 \
  --joint-temporal-guard 1 --joint-temporal-max-step-m 0.025 \
  --workers 8 --max-frames 0
```

这里的 `--min-joint-inlier-views 3` 会禁止最终结果使用不可靠的双目深度分支；
双目只用于锚点/跨相机关联。`--anatomy-protected-cameras camera2` 会限制后处理对目标相机的
二维投影偏移。融合完成后还会在相邻有效帧及最多 3 帧短间隔内清除“二维位置基本不动、但三维深度突然换分支”的
远端关节，保留缺失而不绘制错误跳变。该清除只作用于最终 3D JSONL，不修改
WiLoR/SAM3 的原始二维结果。

如果上游使用的是原始 WiLoR，把两个命令中的
`wilor_multiview_sam3_assisted` 换成 `wilor_multiview`，并把
`--sam3-gate 1` 改为 `--sam3-gate 0`；已有的 `sam3_assist` 字段会自动参与二维区域
门控和射线降权。最终目录可渲染：

```bash
python scripts/render_multiview_wilor.py \
  --dataset "$DATA/normalized_multiview" \
  --fusion "$DATA/fusion_multiview_cam2_optimized" \
  --output "$DATA/fusion_multiview_cam2_optimized/diagnostic_6view.mp4" \
  --final-output "$DATA/fusion_multiview_cam2_optimized/final_only_6view.mp4" \
  --overlay-mode diagnostic --cameras camera0 camera1 camera2 camera3 camera4 camera5

python scripts/render_multiview_wilor.py \
  --dataset "$DATA/normalized_multiview" \
  --fusion "$DATA/fusion_multiview_cam2_optimized" \
  --output "$DATA/fusion_multiview_cam2_optimized/cam2_optimized_wilor_2d_only.mp4" \
  --overlay-mode wilor-2d-only --cameras camera2 --columns 1
```

`fusion_multiview_leaveout_cam2/summary.json` 中的 `ray_geometry` 是不含 camera2 的
五目基准；最终六目 summary 中可检查 `ray_conditioning_*`、`min_ray_angle_*`、
`consensus_rejected_camera_counts.camera2` 和 `joint_temporal_guard`。没有真实 3D 标注时，
接受率、重投影误差、逐关节条件数和时序步长只能作为代理指标，不能替代绝对 3D 误差。

## 最终视频仍出现“手指扭曲”时：二维共识深度分支门控

如果骨长基本正常、但某一根手指在相邻帧突然绕到手背另一侧，通常不是
WiLoR 二维点整体漂移，而是近似平行的多目射线给出了两个都能投影到二维的深度分支。
此时不要再次逐关节强行三角化；用已有融合后的 `joints_base_m` 做保守门控：当三维
骨骼方向变化超过 75°、而不少于 3 个相机的同一二维骨骼方向变化不超过 30° 时，清除
该相邻手型（两端一起清除，避免无法判断哪一端错误）。掌部宽度小于 25 mm 或大于
180 mm 的完整姿态也会清除。这样不会修改 WiLoR/SAM3 二维数据，只把不可信的三维手型
交给连续视频的保守策略显示；连续版推荐使用固定骨长的 `geodesic` 过渡，避免线性补帧
逐关节拉伸造成手指变形，同时仍能跟随前后可信姿态运动。

```bash
PY=/home/p3/miniforge3/envs/ego-hand/bin/python
$PY scripts/postprocess_pose_branch_gate.py \
  --source /path/to/fusion_multiview_stable_joint_v7 \
  --output /path/to/fusion_multiview_depth_branch_gate_v1

$PY scripts/render_multiview_wilor.py \
  --dataset /path/to/normalized_multiview \
  --fusion /path/to/fusion_multiview_depth_branch_gate_v1 \
  --output /path/to/fusion_multiview_depth_branch_gate_v1/final_only_6view_complete.mp4 \
  --overlay-mode final-only --joints-source refined \
  --final-min-valid-joints 21 --interpolate-short-gaps 0 \
  --interpolate-all-gaps 0 --video-codec mp4v

$PY scripts/render_multiview_wilor.py \
  --dataset /path/to/normalized_multiview \
  --fusion /path/to/fusion_multiview_depth_branch_gate_v1 \
  --output /path/to/fusion_multiview_depth_branch_gate_v1/final_only_6view_allgaps_hold.mp4 \
  --overlay-mode final-only --joints-source refined \
  --final-min-valid-joints 21 --interpolate-short-gaps 0 \
  --interpolate-all-gaps 1 --all-gap-prediction-mode hold-previous --video-codec mp4v
```

门控的实际清除数量和每个触发骨骼保存在
`depth_branch_gate_report.json`；其中 `temporal_global_branch_correction` 表示上游时序
门控曾替换过可疑的全局深度分支。交付检查优先看 `final_only_6view_complete.mp4`；需要
连续且跟手的播放时使用 `hold-previous`，并可显式加 `--final-require-local-view 1`；缺深度
或本相机遮挡的手不会硬画来自另一相机的预测三维。需要当前二维观测时再显式加
`--final-predicted-2d-fallback 1`。`hold-previous` 仅适合完全不允许缺口内形状变化的
保守检查，但会在缺口期间停住。
不要把 `--all-gap-prediction-mode linear` 用于最终视频，它会把两个不同姿态的每根手指
独立拉直，重新制造“扭曲”。

可保存同一套 A/B 数值报告（若有 GT，再加上真实绝对误差）：

```bash
python scripts/evaluate_target_camera_optimization.py \
  --dataset "$DATA/normalized_multiview" \
  --raw-predictions "$DATA/wilor_multiview" \
  --optimized-predictions "$DATA/wilor_multiview_cam2_optimized" \
  --leaveout-fusion "$DATA/fusion_multiview_leaveout_cam2" \
  --original-fusion "$DATA/fusion_multiview" \
  --optimized-fusion "$DATA/fusion_multiview_cam2_optimized" \
  --output "$DATA/fusion_multiview_cam2_optimized/cam2_optimization_comparison.json"
# 有真实标注时，再追加：--ground-truth /path/to/gt.jsonl
```

主要阈值：

```text
--sam3-conf 0.25
--sam3-min-score 0.25
--sam3-min-box-iou 0.20
--sam3-min-joint-support 0.55
--sam3-hard-joint-support 0.35
```

`min-joint-support` 是“可信假设”门限，`hard-joint-support` 是“在其它相机一致时才可
剔除该视角”的更低门限。建议先保持默认值；如果遮挡较多导致误删，可降低
`--sam3-min-joint-support`，不要直接关闭六目几何和时序门控。

## 原生/改进 A/B 与取长补短

怀疑后处理使手势变慢或不跟手时，必须在同一份 `normalized_multiview`、同一批帧上做
原生基线。原生基线使用提交中的 `scripts/fuse_multiview_wilor.py`，不启用 SAM3、时序、
解剖或 MANO 后处理；改进版再使用 `fuse_multiview_wilor_guided.py` 和质量门控。两者的
`accepted.jsonl` 不要混写，视频也要并排检查。原生基线可能单帧重投影更低，但会因左右手
关联歧义大量丢帧；改进版覆盖率高、3D 连续，但不可靠相机不应继续显示投影出来的 3D。

最终交付采用逐相机的混合策略：可信时保留改进后的六目 3D；当前相机的 3D 投影与原始
WiLoR 2D 观测不一致时，改画该相机的 2D 骨架。`--final-2d-residual-fallback-px 35`
配合 `--final-2d-residual-percentile 95` 会用每根关节残差的 95 分位数做门控，比只看
中位数更容易拦住“只有一根手指扭曲”的情况；这只改变审查视频，不修改融合 JSONL。
主运行脚本通过 `EGO_FINAL_2D_RESIDUAL_FALLBACK_PX` 和
`EGO_FINAL_2D_RESIDUAL_PERCENTILE` 可调，设前者为 `0` 可关闭该显示回退。另有
`EGO_FINAL_MIN_JOINT_INLIER_VIEWS=4`：如果一根最终关节只有三目或更少射线支持，整手
直接显示原生 2D，避免时序补点把“完整”手指画到错误深度。

对于戴手套数据，建议审查视频使用更严格的组合：
`--final-2d-residual-percentile 95`、`--final-2d-residual-fallback-px 35`、
`--final-wrist-residual-fallback-px 24` 和 `--final-min-joint-inlier-views 4`。
它会拦截单根指尖的长尾误差和整手不跟手的情况；不满足条件时只显示当前相机的
原生 WiLoR 2D，不伪造“平滑但错误”的 3D。该策略只改变视频，不修改融合 JSONL。

若需重新融合，建议先以原生 WiLoR 2D 为输入，设置 `--condition-hard-threshold 150`，
关闭 `--anatomy-refinement`，只开启 `--joint-temporal-guard`。在这条记录上该组合的
连续帧掌部 P95 约 9.3 mm、关节 P95 约 13.2 mm；强行做骨骼补全会重新引入错误深度分支，
因此只应作为可选实验。

示例输出（同一条 20260910 黑手套记录）如下：

```text
fusion_multiview_native_baseline_v1/final_only_6view.mp4
fusion_multiview_depth_branch_gate_v2_physical_lr_fixed_quality4_temporal_v1/
  final_only_6view_hybrid_support3_guard20_p75.mp4
fusion_multiview_raw2d_condition150_temporal_v1_physical_lr/
  final_only_6view_best_safe_v1.mp4
native_vs_guided_support3_ab_6view.mp4
```

如需重新生成数值报告：

```bash
PYTHONPATH=scripts $PY scripts/compare_native_guided_ab.py \
  --native /path/to/fusion_multiview_native_baseline_v1 \
  --guided /path/to/fusion_multiview_depth_branch_gate_v2_physical_lr_fixed_quality4_temporal_v1 \
  --ab-video /path/to/native_vs_guided_support3_ab_6view.mp4 \
  --output /path/to/native_vs_guided_ab_report.json
```

并排视频左侧是原生 HEAD，右侧是“改进 3D + 原生 2D 回退 + 每个关节至少三目支持 + 仅补短缺口”。没有真实 3D
标注时，这个 A/B 只能证明覆盖率、重投影和连续性改善，不能替代绝对位姿误差评测。

## 批量处理

```bash
./scripts/run_multiview_wilor_batch.sh \
  --input-dir /home/p3/data_sda1/ego_hand_system/recordings/20260907 \
  --output-root /home/p3/data_sda1/ego_hand_system/output/gen6_sam3_batch \
  --device cuda \
  --gpu-profile rtx5090d \
  --max-frames 0 \
  --sam3-glove-assist 1 \
  --sam3-conda-env annotator
```

SAM3 会为每个相机、每个同步帧运行一次，速度明显慢于纯 WiLoR。推荐先用 60 帧检查
`summary.json` 和视频，再处理完整记录。脚本会保留原始 WiLoR 预测，方便在没有实际
改善时安全回退。
由于 SAM3 本身使用 GPU，批量脚本在启用该开关时会自动关闭 `rtx5090d` 的并行流水线，
避免下一条记录的 WiLoR 推理和当前记录的 SAM3 同时争用显存。

## 当前作用边界

SAM3 可改善错误检测、左右手假设选择、明显越出手套区域的 2D 点、错误相机视角剔除和
由此产生的 3D 深度跳变；它不能单独恢复所有相机同时错误的姿态，也不能替代 WiLoR 的
关节点回归或相机标定。最终
是否提升应比较同一记录的检测保留率、六目重投影误差、3D 掌部步长/加速度和人工抽帧结果。
