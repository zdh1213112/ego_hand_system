# 原图手套标签导出

## 优化标签（训练用）一条命令

在 `ego_hand_system` 仓库根目录运行。把 `--recording` 换成已配对录制目录，显式指定 `configs/native_nokov_v4.json` 即可。程序读取本段 NOKOV 的手套描述，判断录到了哪只手。本次新建输出目录，并把实际生效的参数、权重 SHA-256、代码摘要及与签收参数的差异写入 `settings.json`。[显式配置](../configs/native_nokov_v4.json)

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /你的/已配对录制目录 \
  --optimization-config configs/native_nokov_v4.json
```

| 录制内容 | 实际执行的流程 | 训练格式标签，相对本次输出目录 |
| --- | --- | --- |
| 双手，各有完整 24 点手套定义 | 自适应双腕、解剖和 MCP 拟合，逐曝光导出，再对训练划分执行 v4 右小指修正和异常样本剔除 | `optimized/labels/`，读取 `optimized/train.jsonl` 和 `optimized/val.jsonl` |
| 仅左手或仅右手 | 已有的 `native_nokov_temporal.json` 单手时序流程，执行 NOKOV 与 SDK 逐指拟合、关节约束和轮廓核对 | `optimized/dense/fit/final/export/merged/` |

例如本文的 `pair_20261111_191855` 只录到了右手，执行第二行。验证第一行时，将 `--recording` 指向你的双手录制目录，例如 `/你的/双手录制目录/183304`。终端会打印识别到的左右手、方法和输出位置，`settings.json` 会记录实际分支、配置内容和录制路径。默认在工作区 `work/<录制名>_labels_<时间>/` 新建输出。可用 `--output /你的/新目录` 指定位置，加 `--dry-run` 只检查输入并查看分支，中断后用相同配置和输出目录加 `--resume` 继续。

### 两次需要核对的位置

1. **第 06 步腕部参照**。双手腕带和原图证据充分时，程序分别自动签收参照并继续。扩大采样后仍缺证据时，程序停下，原因在 `fitting/legacy/<录制名>/review/adaptive_wrist/unusable.json`。请核对对应手的腕带观测及录制配对，必要时补采。足够的证据才会生成该腕参照。
2. **最终数据集签收**。双手导出完成后，打开 `optimized/review/DATASET_REVIEW.md` 和整段视频。用户确认该版标签后才能另行执行 `--publish`。单手分支交付原生时序标签供查看。

### 本版实测范围

从同一份第 12 步结果重建，183304 和 183959 的 599 份右小指修正标签逐字段相同。从原片新算时，183304 重新选取了双腕参照，保留了原签收曝光的 58.80%；191855 单手分支产出 4,414 份标签，通过 8/19 项解剖检查。新录制请按本段视频和验收记录判断能否用于训练。[两段整链路实测、差异与视频](../tools/label_pipeline/VERIFY.md#round24-replay)

### 覆盖签收参数

保留同一入口，按需在所用 JSON 的 `label_pipeline.overrides` 下填写差异。默认 `{}` 使用签收参数。例如 `{"parameters":{"v4":{"fit":{"max_tip_displacement_mm":2.5}}}}` 只修改双手分支右小指内部指尖预算。单手逐指参数放在 `single_hand`，计算批大小等放在 `runtime`。终端逐项打印旧值和新值，`settings.json` 保存默认表及其 SHA、最终取值、观测权重路径和 SHA、代码文件摘要。单手或外部观测的权重使用情况见 `observation_weights.*.used`。算法参数使用 JSON 覆盖，输入和输出路径仍由命令指定。

### 换机器后需要的输入

- 在仓库旁检出 `nokov_ego_hand_validation`，在本仓库 `output/dependencies/nvdiffrast_runtime/` 安装原生光栅器。标定工具位于 `tools/label_pipeline/calibration/`，训练读取器位于 `training/`。使用 `ego-hand` 环境，或设置 `EGO_NATIVE_PYTHON` 指向该环境的 Python。双手拟合使用 CUDA。拟合和训练读取所用的 MANO 模型必须匹配。[依赖和默认参数](../tools/label_pipeline/defaults.yaml)
- 一个录制目录含一个 `EGO*/part0001/` 和一个 `session*/nokov/`。支持四机 MCAP，也支持四路原始 MP4、对应 PTS 及 IMU CSV。保留该次录制的相机标定、NOKOV 刚体、手部骨架、marker 和手套描述。配置自带 11 月 9 日设备安装初值，重新安装设备后在 JSON 的 `label_pipeline.seed_alignment` 指定新的初值。
- 双手分支默认从当前录制生成同曝光 WiLoR 观测。检测器、WiLoR 基础权重、观测用 adapter 和模型配置的位置在 JSON 的 `observation_weights`，其中 `ego_hand_system/` 前缀定位到当前检出目录。检测器与基础权重放在 `models/wilor/`，观测专用的签收版 v2 adapter 放在 `models/wilor/observation_v2/best_trainable.pt`，模型配置位于 `training/pretrained_models/model_config.yaml`。部署推理使用的 v4 adapter 由推理命令单独指定。已有本段四机观测可加 `--wilor-observations /你的/native_wilor.npz`，拟合前会核对参考时戳和四路曝光帧号。模型文件与录制数据需要部署到目标机器。

### 双手分支执行哪些优化

| 阶段 | 处理 |
| --- | --- |
| 01–03 | 转存或直接读取四路原图，用 IMU 和 NOKOV 刚体同步，双手 marker 对齐后进行四机空间标定。 |
| 04–05 | 按本段标定手形和表面挂点，修正无名指背侧挂点。每指至少 3 个实测点，关节限位权重为 0.05。结合 SDK 骨链、可见轮廓和连续曝光约束指节方向、反折及快速动作。 |
| 06–07 | 左右手分别用三个同名腕带球、原图亮点和指尖位移自动选外旋参照。腕带与手掌分别拟合，五个指尖受位移约束，名义预算为 3.5 mm。指甲 marker 对应网格表面挂点。结合解剖先验修整拇指和四指的远端弯折。若腕部观测不足，原因写入 `review/adaptive_wrist/unusable.json`。 |
| 08 | 在三维手掌坐标系中调整食、中、无名、小指 MCP，默认左手向外 1.5 mm、右手 3.0 mm。对指尖与指节角度逐指检查，历史 `group_009` 有独立签收参数。 |
| 09–12 | 按手和曝光筛选表面残差、遮挡及孤立姿态，导出四机原图、MANO 参数、21 个三维关节与 778 个顶点。按录制时间留出末段约 10% 验证，核对训练读取器和鱼眼投影。中间签收页位于 `fitting/legacy/<录制名>/review/`。 |
| v4 训练标签 | 在训练时段用指甲与远端 marker 修正右小指 PIP/DIP，以多视角实拍亮点或两种 NOKOV 配准复核，约束内部指尖位移在 3 mm 内。逐曝光排除 SDK 和多视角 WiLoR 同时缺乏支持的异常训练手姿，沿用原验证姿态。重算 MANO 几何和 KB 投影，读取最终 NPY 生成预览。 |

实际求解参数与历史签收核验随代码提供。[逐阶段运行说明](../tools/label_pipeline/README.md) · [默认参数](../tools/label_pipeline/defaults.yaml) · [五段历史回归](../tools/label_pipeline/VERIFY.md)

### 查看、签收和取用双手结果

| 要看什么 | 相对输出目录 |
| --- | --- |
| 本次配置及自动判定分支 | `settings.json`、`recording.yaml` |
| 左右腕参照与测量依据 | `fitting/legacy/<录制名>/review/WRIST_REVIEW.md` |
| **最终 v4 整段视频及验收页** | `optimized/review/camera0_full.mp4`、`optimized/review/DATASET_REVIEW.md` |
| 修正、剔除、划分数量，最终文件摘要 | `optimized/review/dataset_targets.json`、`optimized/rejected.jsonl`、`optimized/export_sha256.csv` |
| MANO 回算、KB 重投影和训练读取 | `optimized/geometry_check.log`、`optimized/dataset_profile.json`、`optimized/loader_check.json` |

最终视频保留相机 0 的原始时间轴，用训练文件的 `joints_3d + trans` 和鱼眼参数投影 21 个关节。训练格式的 `joints_2d` 保存 778 个网格顶点的投影。审核通过后，在 `optimized/review/dataset_signoff.json` 将 `approved` 改为 `true`，保留生成的 SHA。用原命令指定同一 `--output` 并追加 `--publish`，可在 `published/` 取得已签收的清单。发布会复核视频、索引、标签和图片摘要。目录之间复用了符号链接，搬到其他机器时要连同链接目标复制，或在复制时展开链接。

历史九段 v4 数据集可直接从开发机 `/home/tione/notebook/wilor_training/datasets/glove_trainset_20261123_v4/` 取用，共 101,446 份训练样本、11,933 份验证样本。对新录制始终以本次运行的划分和样本数为准。

详细原理和参数见 [完整说明](EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)。

## 原生基线导出

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

## 单独运行逐指或时序拟合

在上面的命令后加 `--optimization-config configs/native_nokov_articulation.json` 可启用实测 marker 与 SDK 指骨共同约束的逐指拟合。历史实验使用 `configs/native_joint_refit.json`。两种配置均记录在输出目录的 `settings.json` 中。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system

bash scripts/export_ego_nokov_labels.sh \
  --recording /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand/pair_20261111_191855 \
  --optimization-config configs/native_nokov_articulation.json
```

连续动作可在同一条命令中使用 `--optimization-config configs/native_nokov_temporal.json`。该配置用连续三帧的实测手套点和 SDK 指骨约束指间关节，同时固定每帧已有的腕点和手型。时间戳中断、漏帧和缺失的观测会截断相应的时序约束。五段录制的额外快速弯指样例经过原图亮点、原生 WiLoR 和关节范围核对，收录在本次数据集。

单手、双手录制都可以，程序会自动识别录到了哪只手。配置里的 `placement` 指向 `configs/glove_marker_placement.json`，它只写了每个 marker 贴在哪一节手指、哪一面（手背），不含尺寸。所以同一副手套、同样的贴法，换人不用改；手套或贴法变了，才需要新的贴点文件和配置。手型、marker 贴点和腕根偏移在这段录制自己的数据上标定，不沿用其他录制的结果。基线结果照常生成，优化结果放在 `optimized/` 下。轮廓提取和图像对齐用多个 CPU 进程并行，默认最多 16 个，可在 `run_joint_refit_recording.py` 里用 `--workers` 调整。

跑完后终端会打印这些路径：

| 内容 | 位置（相对输出目录） |
| --- | --- |
| 训练用标签 | `optimized/dense/fit/final/export/merged/` |
| 标签查看器 | `optimized/dense/fit/final/export/viewer/index.html`，用浏览器直接打开 |
| 每帧验收结果 | `optimized/dense/fit/summary.json`，剔除清单在 `{left,right}_rejected.json` |
| 原图核对 | `optimized/dense/fit/final/export/image_evidence_check.json` |
| 运行设置 | `settings.json`，记录本次是基线还是优化，以及用的配置 |

自动剔除规则：每只手每个时刻，对齐没能用图像修正的、20 个手指 marker 加手背 HandOffset 合计少于 19 个的、marker 残差中位数超过 2.5 mm 或最大值超过 8 mm 的，都不导出。

`native_nokov_articulation.json` 还要求每指至少三个实测点。它限制各指关节的屈伸、扭转和侧摆，根据每手标定的偏角检查实测 marker 与 SDK 骨链共同支持的转向，并在可靠视角微调轮廓。双手是否同时通过、该视角是否完整，会进一步影响训练标签数量。[逐指拟合和输出说明](../scripts/joint_refit/README_zh.md)

### 中断后继续

任何一步失败后，修好原因，用同一个输出目录加 `--resume` 再跑。已经完成的步骤会跳过，中断时做了一半的步骤会删掉重做：

```bash
bash scripts/export_ego_nokov_labels.sh \
  --recording /同一个录制 \
  --optimization-config configs/native_nokov_articulation.json \
  --output /上次的输出目录 --resume
```

如果录制、初始标定或配置和上次不一样，程序会拒绝续跑，这时请换新目录。`--resume` 只对本版本开始的运行有效，更早的输出目录没有完成记录。

## 汇总旧版 joint_refit 输出并训练

在仓库根目录执行，汇总带有检查记录的旧版 joint_refit 导出。若额外指定的旧标签位于其他位置，可加 `--extra /path/to/export/merged`。

```bash
python training/assemble_joint_refit_trainset.py --work-root ../work \
  --output training/datasets/joint_refit_new
```

每段末尾约 10% 时间留给验证。双手 v4 导出已包含本段训练和验证清单。用新数据训练时复制 v4 配方，将 `paths.dataset_root` 指向目标数据集并设置独立的输出目录。[训练命令](../training/README.md)

标签里的 `approved_for_training` 仍然是 `false`。这只是人工确认标记，训练脚本不读它。

## 两设备数据放在不同目录

```bash
bash scripts/export_ego_nokov_labels.sh \
  --ego /你的/EGO原始录制目录 \
  --nokov /你的/NOKOV原始目录
```

## 原生基线与 joint_refit 的常用选项

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

## 单独验收四目观测

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
