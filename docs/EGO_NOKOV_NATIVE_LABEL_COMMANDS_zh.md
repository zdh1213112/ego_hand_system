# 原图双手标签：直接执行

详细原理和参数见 [完整说明](EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)。

## 已配对的录制：一个命令

进入项目后，只替换录制目录即可。目录内应有一个 `EGO*` 和一个 `session*/nokov`。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system

bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916
```

自动执行：标准化 → 时间同步 → 双手对齐 → 四目空间校准 → 默认 MANO 拟合 → 四路原图标签 → 检查 → 四路视频。

默认不开启 MANO 联合优化。输出自动保存到 `EGO_glove/work/录制名_native_时间/`，终端会打印标签和视频的完整路径：

- 标签：`baseline/export/merged/`
- 视频：`baseline/export/videos/`
- 日志：`logs/`

## 优化标签（训练用）：一个命令

在上面的命令后加 `--optimization-config configs/native_joint_refit.json`。不加这个配置时，只跑上面的基线流程。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system

bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_191855 \
  --optimization-config configs/native_joint_refit.json
```

单手、双手录制都可以，程序会自动识别录到了哪只手。配置里的 `placement` 指向 `configs/glove_marker_placement.json`，它只写了每个 marker 贴在哪一节手指、哪一面（手背），不含尺寸。所以同一副手套、同样的贴法，换人不用改；手套或贴法变了，才需要新的贴点文件和配置。手型、marker 贴点和腕根偏移在这段录制自己的数据上标定，不沿用其他录制的结果。基线结果照常生成，优化结果放在 `optimized/` 下。一段约 2000 帧的录制大约要跑一个多小时，其中轮廓提取约 25 分钟。

跑完后终端会打印这些路径：

| 内容 | 位置（相对输出目录） |
| --- | --- |
| 训练用标签 | `optimized/dense/fit/final/export/merged/` |
| 标签查看器 | `optimized/dense/fit/final/export/viewer/index.html`，用浏览器直接打开 |
| 每帧验收结果 | `optimized/dense/fit/summary.json`，剔除清单在 `{left,right}_rejected.json` |
| 原图核对 | `optimized/dense/fit/final/export/image_evidence_check.json` |
| 运行设置 | `settings.json`，记录本次是基线还是优化，以及用的配置 |

自动剔除规则：每只手每个时刻，对齐没能用图像修正的、手指 marker 少于 19 个的、marker 残差中位数超过 2.5 mm 或最大值超过 8 mm 的，都不导出。

### 中断后继续

任何一步失败后，修好原因，用同一个输出目录加 `--resume` 再跑。已经完成的步骤会跳过，中断时做了一半的步骤会删掉重做：

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /同一个录制 \
  --optimization-config configs/native_joint_refit.json \
  --output /上次的输出目录 --resume
```

如果录制、初始标定或配置和上次不一样，程序会拒绝续跑，这时请换新目录。`--resume` 只对本版本开始的运行有效，更早的输出目录没有完成记录。

## 汇总成训练集并训练

每跑完一段录制，就重新汇总一次。脚本会自动收集 `EGO_glove/work/` 下所有带优化、并且通过数据集检查的录制：

```bash
cd /home/p3/data_sda1/EGO_glove
/home/p3/miniforge3/envs/ego-hand/bin/python -B wilor_training/assemble_joint_refit_trainset.py \
  --output wilor_training/output/native_kb_training_20261111/joint_refit_trainset_新名字
```

- 每段录制时间轴上最后 10% 的时刻做验证集，其余做训练集。加入新录制不会打乱已有录制的划分。
- 图片和标签用硬链接，不额外占磁盘。
- 输出目录要用新名字，或者先删掉旧目录；内容不同的已有目录会报错。
- 不在 `work/` 下的旧结果用 `--extra 目录/export/merged` 加进来。例如 192916 的旧优化标签：`--extra wilor_training/output/native_kb_training_20261111/joint_refit_v5_dense/fit/final/export/merged`。用上面的命令重跑 192916 之后，就不用再加 `--extra`。

训练前，先把 `wilor_training/output/native_kb_training_20261111/joint_refit_train.yaml` 里的 `paths.dataset_root` 改成上面的汇总目录，再启动：

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --mode joint_lora \
  --config output/native_kb_training_20261111/joint_refit_train.yaml \
  > output/native_kb_training_20261111/train_joint_refit.log 2>&1 &
```

标签里的 `approved_for_training` 仍然是 `false`。这只是人工确认标记，训练脚本不读它。

## 两设备数据放在不同目录

```bash
bash scripts/export_ego_nokov_labels.sh \
  --ego /你的/EGO原始录制目录 \
  --nokov /你的/NOKOV原始目录
```

## 常用选项

在上述命令末尾按需添加：

| 选项 | 用途 |
| --- | --- |
| `--output /你的/新输出目录` | 自己指定输出位置 |
| `--observations-only` | 仅完成对齐、逐帧 RGB 对应与多目几何检查，输出观测视频；不拟合 MANO |
| `--dry-run` | 检查输入、查看执行步骤；不训练、不创建输出 |
| `--sample-stride 8` | 每隔 8 帧导出，默认值；通过质量筛选才保留 |
| `--source-fps 30` | 原视频帧率，默认 30；用于设置可视化回放速度 |
| `--device cpu` | 使用 CPU；默认 CUDA |
| `--resume --output 上次的目录` | 中断后继续，跳过已完成的步骤 |
| `--seed-alignment /新的/alignment.json` | 设备重新安装后指定新的初始安装标定 |
| `--joint --marker-offset-mm 0` | 额外生成实验性四目 MANO 联合结果，零偏移 |
| `--joint --marker-offset-mm 2` | 额外生成实验性四目 MANO 联合结果，2 mm 偏移 |

目前建议先看默认模式。联合模式已改为保留关节约束的受限修正 v2；不合格帧回退到基线，当前数据是否能改善仍以验收结果为准。相关诊断见完整说明。
联合模式同时保留默认结果，联合标签/视频分别在 `joint/export/merged/`、`joint/export/videos/`。

当前入口适用单个 `part0001`；NOKOV 录到单手或双手都可以，会自动识别。输出目录必须是新目录。相机标定读取本次录制；初始安装标定默认沿用本机已验证的 11 月 9 日标定，设备安装变化后需要更新。

## 当前推进阶段：先验收四目观测

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916 \
  --observations-only
```

报告和四路匹配视频位于新输出目录的 `rgb_observations/`。蓝圈是 NOKOV 投影初值，黄点是 RGB 匹配，绿点是至少三目几何一致的观测；绿色不等于已经确认 marker 身份或标签精度。

## 24 点试拟合实验（不替换基线）

仅适用于当前 `pair_20261111_192916`，自动生成新的实验目录：

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/marker24_pilot/run.sh
```

[详细说明与当前结果](/home/p3/data_sda1/EGO_glove/wilor_training/tools/marker24_pilot/README.md)。


## 可选：手动调整个人手型与 mesh

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/manual_shape_tuner/start.sh
```

打开 http://127.0.0.1:8767/ ，调整后点击“保存全部调整到本机”。本工具目前针对已有 20261111 录制及 PERSONAL-3 产物。

## 显式启用今天的 24 marker＋共享个人手型＋四目轮廓路线

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_192916 \
  --optimization-config configs/native_personal_marker24.json
```

完整开关、阶段、输出位置和配置说明见 [优化路线执行命令](EGO_NOKOV_PERSONAL_MARKER24_COMMANDS_zh.md)。不传配置或 `enabled: false` 执行基线；优化结果在单独的 `optimized/`。模板的参考帧和留出区间针对上述录制，新录制需更新。

## 加载已导出标签进行交互可视化

```bash
bash scripts/view_wilor_labels.sh --dataset /你的输出目录/optimized/export/merged
```

打开 http://127.0.0.1:8770/ ，可切相机／帧、显示左右手 mesh、缩放、播放和全屏。[完整查看器说明](WILOR_LABEL_VIEWER_zh.md)。

## 优质关键帧筛选与固定手型标定（当前录制实验）

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/keyframe_calibration/run.sh
```

基于本机 `pair_20261111_192916` 已完成的基线与分割审核，自动选择左右手各 16 个多目关键时刻，标定并固定手型／附着，再生成验证报告和四路对照视频。[结果、适用范围和说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/keyframe_calibration/README_zh.md)。这是独立实验入口，原始数据导出配置没有自动切换；尚未批准为训练真值。

## 完整轮廓驱动共享手型（独立拟合对照）

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/full_silhouette/run.sh
```

使用当前录制已筛选的优质分割，比较“只改共享手型”和“标定帧动作小幅修正”两组，生成独立评分及四路近景视频；没有替换默认标签导出。[结果与说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/full_silhouette/README_zh.md)。

## 假定 SDK 已配置内部偏移：标签对照

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_skeleton_fit/run.sh \
  --assume-sdk-internal-offsets
```

固定当前个人手型，比较 SDK Hand 根与 Finger0 公共根两种腕根映射，输出实际双手标签与四路视频。仅启用假设，不修改 NOKOV 配置；当前图像贴合整体变差，不替换旧标签。[结果与说明](/home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_skeleton_fit/README_zh.md)。


## SDK 方向辅助与 Finger0 腕根候选（独立对照）

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_direction_fit/run.sh
```

保留用户选择的 Finger0 公共根参考；固定当前手型、marker 附着和根平移，比较仅 marker 重拟合与新增 SDK 方向辅助，不采用 SDK 关节位置／骨长硬目标。本轮收益有限，不替换默认标签。[结果、范围与查看入口](/home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_direction_fit/README_zh.md)。
