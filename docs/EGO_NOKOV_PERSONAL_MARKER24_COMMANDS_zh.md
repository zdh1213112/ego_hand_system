# 基线＋显式启用 24 marker／个人手型／四目轮廓

进入项目：

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
```

## 关闭优化：原基线命令

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916
```

不传优化配置，或配置中 `enabled: false`，均执行原基线。

## 开启今天的优化路线

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916 \
  --optimization-config configs/native_personal_marker24.json
```

先生成基线，再依次运行：24 点附着和共享手型标定 → 四路原图轮廓提取 → 共享手型轮廓修正 → 对比视频 → 优化结果标签导出与检查 → 标签视频。

配置文件：`configs/native_personal_marker24.json`。左手、右手各一套固定 betas，分别跨帧、跨相机共享。姿态、旋转、平移按帧估计。

**模板中的手背参考帧 616／584、留出区间 [512,704) 对应上述录制。换录制时应更新这些配置，确保参考帧确实显示手背。相机标定、时间和空间对齐从本次录制读取。**

## 查看计划，不执行

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916 \
  --optimization-config configs/native_personal_marker24.json \
  --dry-run
```

## 结果位置

输出根目录自动生成在 `EGO_glove/work/录制名_native_时间/`，也可加 `--output /新的目录`。

| 路径（相对输出根目录） | 内容 |
|---|---|
| `baseline/export/merged/` | 基线标签 |
| `baseline/export/videos/` | 基线标签视频 |
| `optimized/marker24/` | 24 点标定与共享手型、动作拟合 |
| `optimized/contours/observations.json` | 本次四目原图轮廓候选 |
| `optimized/contour_fit/` | 轮廓约束后的参数及 mesh |
| `optimized/contour_fit/review/` | 四路对比查看器和采样对比视频 |
| `optimized/export/merged/` | 优化路线的实验标签 |
| `optimized/export/videos/` | 通过筛选的相机标签视频 |
| `optimized/context.json` | 本次录制和阶段输入 |
| `optimized/commands.json`、`optimized/logs/` | 实际命令、状态与日志 |
| `optimized/pipeline_summary.json` | 流程完成状态和结果路径 |

四路对比视频会展示局部视图；标签导出要求同画面左右手及质量筛选通过，某路没有有效标签时不伪造该路标签视频。

## 配置项

| 配置字段 | 作用 |
|---|---|
| `enabled` | 必须明确为 `true` 才启用 |
| `placement` | 左右手的 marker 区域配置，相对配置文件解析 |
| `dorsal_reference` | 本次录制中左右手各自的手背参考相机和帧 |
| `holdout` | 不参与共享手型／附着标定的时间窗口，右端不包含 |
| `calibration_frames` | 每只手的标定样本数 |
| `calibration_steps` | 分阶段 marker／手型优化迭代数 |
| `contour_steps` | 共享 betas 的原图轮廓修正迭代数 |
| `pose_scope` | 默认 `sampled_sequence`：对本次所有合格采样帧拟合动作；`holdout` 仅复现采样标定＋留出片段实验 |
| `diagnostics` | 默认关闭；打开后额外运行整体／手指／附着分项对照及重新渲染评分，不自动选择某个分支替换标签 |
| `diagnostic_steps` | 分项对照迭代数 |
| `export_optimized` | 默认打开，生成独立的实验标签和视频 |

抽帧沿用命令的 `--sample-stride`，默认 8。只有真正经过优化且通过筛选的帧才进入优化标签；不把未优化帧混入该结果。`holdout` 模式不是整段录制优化。

当前四目轮廓阶段只调整共享手型；24 点阶段同时拟合动作。分项诊断是可选实验，尚未合成所有参数同时自由变化的求解器。没有接入 SAM3。

实验标签带 `personal_marker24_optimization_enabled: true` 与 `approved_for_training: false`。流程和数学重建检查通过，不代表实际手套贴合精度已验收。新入口不与旧 `--joint` 或 `--observations-only` 混用。

## 查看实际导出的优化标签

```bash
bash scripts/view_wilor_labels.sh --dataset /你的输出根目录/optimized/export/merged
```

打开 http://127.0.0.1:8770/ 。[查看器操作说明](WILOR_LABEL_VIEWER_zh.md)。


## 进一步同时修正动作与共享附着关系

原配置不变。要开启这一步，使用新配置：

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916 \
  --optimization-config configs/native_personal_marker24_joint.json
```

其中 `joint_pose_attachments.enabled: true`。共享手型保持固定，每帧动作和跨帧共享附着在同一目标中修正；有足够多视角轮廓的帧才参与。最终标签来源为 `optimized/joint_fit`，对比在 `optimized/joint_fit/review/`。原先“当前四目阶段仅改 betas”的说明仍适用于旧配置；新配置在该阶段之后增加动作＋附着联合修正。

[实测结果、限制与详细说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/joint_pose_attachment/README_zh.md)。
