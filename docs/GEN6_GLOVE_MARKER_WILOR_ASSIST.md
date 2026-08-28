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
    └── diagnostic_6view.mp4
```

预览图中橙色是原始 WiLoR，青/绿色是通过严格门控后的保守修正，洋红圆圈是已经分配给
该 detector 的 marker。`marker_assist_preview.jpg` 优先保存风险最大的修正案例，
`marker_evidence_preview.jpg` 保存最接近强修正门限的只读证据案例。`summary.json` 还记录
稳定的拒绝原因码、估计/实际偏移和骨长变化。

建议先运行 `--max-frames 60`，检查：

- 洋红圈是否稳定落在反光球而不是桌面高光；
- 修正骨架是否比橙色原始骨架更贴合手套；
- marker 辅助后的残差是否稳定低于原始 WiLoR；
- 新旧 fusion 的接受率、重投影中位数/P95 和左右手稳定性。

如果桌面或环境高光过多，可提高 `--marker-value-min`、降低
`--marker-search-padding-px`，或提高 `--marker-min-matches`。如果 marker 暗或运动模糊，
可适当降低亮度阈值或增大匹配距离，但应同时检查预览图，避免错误亮斑拉动关节。
