# GEN 六目手套 Marker 视觉辅助 WiLoR

该功能使用戴手套 EGO 原始图像中可见的反光 marker 球，逐帧辅助 WiLoR 的二维关节点，
再进入现有 Double-Sphere 多相机 RANSAC 融合。它不读取 NOKOV marker/skeleton CSV，
不依赖外部动捕时间同步，也不修改 `calibration/*.json`。

## 算法

每路相机、每一帧按以下顺序处理：

1. 将 BGR 原图转成 HSV，使用 `S < 100`、`V > 160` 得到低饱和高亮掩码；
2. 对掩码做 8 邻域连通域分析，按面积、宽高、长宽比、填充率和圆度保留小圆亮斑；
3. 对每个 WiLoR 假设只保留其 20 个手指投影点周围 `45 px` 范围内的候选；
4. 在 `35 px` 内寻找最近亮斑，用中位数估计粗略二维平移；
5. 平移后以 `13 px` 为门限，用匈牙利算法完成关节与亮斑的一对一匹配；
6. 匹配至少 5 点并覆盖至少 3 根手指时，将整体平移应用到 21 点，再把已匹配的手指
   关节按默认 `0.35` 权重拉向 marker 中心；否则原样保留 WiLoR 结果。

WiLoR 索引 `1..20` 是五根手指各 4 点，和手套每根手指的 4 个反光球对应；索引 `0` 是
手腕，没有直接 marker，对它只应用可靠的手部整体二维平移。匹配后的 marker 数和残差
也作为一个较弱的假设选择代价；多视角重投影与 RANSAC 仍是主约束。

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
--marker-min-matches 5
--marker-min-finger-groups 3
--marker-search-padding-px 45
--marker-seed-distance-px 35
--marker-match-distance-px 13
--marker-max-shift-px 30
--marker-blend 0.35
```

`--marker-blend 0` 只应用粗略整体平移；`1` 会把已匹配手指点完全放到亮斑中心。marker
不在关节旋转中心时，建议先使用默认的柔性权重，而不是直接设为 `1`。

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
│       └── marker_assist_preview.jpg
└── fusion_multiview_glove_marker_assisted/
    ├── accepted.jsonl
    ├── rejected.jsonl
    ├── summary.json
    └── diagnostic_6view.mp4
```

预览图中橙色是原始 WiLoR，青/绿色是 marker 辅助后的 21 点，洋红圆圈是 HSV 检出的
匹配亮斑。`summary.json` 记录辅助假设比例、匹配点中位数、粗偏移量，以及
`raw_marker_residual_median_px` 到 `assisted_marker_residual_median_px` 的变化。

建议先运行 `--max-frames 60`，检查：

- 洋红圈是否稳定落在反光球而不是桌面高光；
- 修正骨架是否比橙色原始骨架更贴合手套；
- marker 辅助后的残差是否稳定低于原始 WiLoR；
- 新旧 fusion 的接受率、重投影中位数/P95 和左右手稳定性。

如果桌面或环境高光过多，可提高 `--marker-value-min`、降低
`--marker-search-padding-px`，或提高 `--marker-min-matches`。如果 marker 暗或运动模糊，
可适当降低亮度阈值或增大匹配距离，但应同时检查预览图，避免错误亮斑拉动关节。
