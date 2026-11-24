# 2026-11-11 原图双手标签与 WiLoR 对照验收

本文记录 cam0/cam1 阶段的实验。后续新增的四目空间优化与四路原图导出见 [四目操作说明](../../../ego_hand_system/docs/EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)；本文训练指标仍属于两目数据对照。

原图链路已用一段真实双手录制完成同步、对齐、双手 MANO 全序列拟合、标签导出、检查、训练与独立姿态评测。当前使用 cam0/cam1，**没有完成四目联合优化**。短程训练可以改善原图位置误差，但三维姿态仍有退化，当前权重应作为实验结果，不能认定为最终更优模型。

## 当前哪些相机参与哪一步

| 环节 | 当前实现 |
| --- | --- |
| 视频标准化 | 保留 EGO 四路原始视频与各自标定，不做图像去畸变 |
| NOKOV → 相机空间标定 | 使用 cam0 原图中的实际 marker 候选，与双手 NOKOV 三维点共同估计头部刚体到 cam0 的共享 SE3 变换 |
| 左右手 MANO 拟合 | 分别拟合真实 NOKOV 三维轨迹；`w_3d=1, w_2d=0`，使用姿态、手形和时序约束 |
| 标签投影、筛选、导出 | cam0/cam1 各自真实 K、KB D 和物理外参；一张原图对应两份手标签 |
| cam2/cam3 | 已保留原始视频和标定，本次未参与空间优化、标签导出或训练对照 |

当前二维拟合参考是 NOKOV 三维点的相机投影，并非独立图像标注。直接开启这批参考的二维权重不能构成四目独立约束。后续四目优化需增加各原图上的独立二维观测、身份关联和遮挡掩码，用四套真实 KB 投影构建鲁棒误差；这仍可以全程使用原图。

## 实际数据与标签结果

原始数据根目录：`/home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand`。本次检查到其中 `pair_20261111_192916`、`pair_20261111_193034` 同时包含真实左右手 marker；其他五段只包含右手，不能作为双手同帧证据。

所有本次实验文件在：

```text
/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/
```

| 录制 | 同步与标定 | 处理结果 |
| --- | --- | --- |
| 192916 | IMU/头部旋转相关峰 0.969，强同步；cam0 自动关联留出点误差中位数 5.25 px | 导出双手标签并作为独立评测录制 |
| 193034 | 同步相关峰约 0.537；右手留出点仅 34 个，低于 50 门限 | `needs_review`；只保存候选对齐，不进入训练或正式验收标签 |

192916 的留出关联误差 P95 为 **71.15 px**，左手 P95 为 **77.19 px**，存在明显自动匹配离群点；中位数通过当前门槛不代表尾部误差已解决。这些留出点未参与 SE3 拟合，但仍来自自动关联，需要人工核对困难帧，不能当成人工标注的独立解剖真值。

192916 标签输出：

```text
real_dual/pair_20261111_192916_native_labels/
├── commands.json
├── mano_fit_full_cam01/summary.json
├── export/merged/                     # 730 张原图、1460 份手标签
├── export/preview/                    # 原图同帧双手预览
└── export/videos/                     # cam0、cam1 各一份标签视频
```

左右手各拟合 **1612 帧**；相对 NOKOV marker 中心的拟合误差，左手中位数/P95 为 **5.04/18.32 mm**，右手 **5.36/17.49 mm**。这包含 marker 中心与 MANO 解剖关节定义差异。

导出使用 `--sample-stride 4 --max-samples 0 --require-both-hands`，筛选后 730 张原图、1460 份手实例，左手/右手各 730。每相机 365 张图片，图片均关联两只手。投影一致性、MANO 重建、合并和静态预览检查通过。

标签视频在 `export/videos/source0000_pair_20261111_192916__both_camera0_labels.mp4` 与 `camera1_labels.mp4`，各 365 帧、7.5 FPS。它们按合格样本顺序紧凑播放，跳过被筛选时刻，不等于完整原始时间轴。

## 原图与矫正图训练对照

对照使用同一录制、同步帧、相机及物理手身份匹配的 **256 个训练实例、64 个验证实例**。训练来自历史左手和右手录制，验证来自另一段左手录制；本表的验证集只含左手。真实双手 192916 整段没有参与训练和选择权重。

两条路线均为 80 个优化步、有效批量 4、随机种子 42、LoRA rank 4/alpha 8，使用同一个 `coarse_start.ckpt`。该初始化复制原始权重并将 refiner 的 pose/shape/camera 六个输出权重和偏置清零；原始文件未修改，来源哈希和变更键记录在 `initialization_provenance.json`。

原图路线使用 `physical_kb`，并适配原图特征采样和物理根位置估计；矫正路线使用 `physical_perspective`。因此这是两套完整路线的短程对照，包含模型适配差异，不能解释为只改变去畸变这一变量的最终结论。各路线按各自验证 loss 选择最佳权重：原图 step 40，矫正图 step 80。

下表已将矫正路线预测通过射线和真实 KB 参数转换回同一个原图坐标；没有重采样图片。转换后两套 GT 平均二维差 **0.000123 px**，三维差低于 **0.00003 mm**。

| 64 个配对验证实例；误差越低越好 | 原图路线 | 矫正路线转回原图 |
| --- | ---: | ---: |
| 腕根相对 21 关节 MPJPE，mm | 40.87 | 44.65 |
| 原图二维平均误差，px | 45.79 | 32.65 |
| 物理相机中腕根位置误差，mm | 74.42 | 31.33 |

原图路线的相对三维误差低约 **8.5%**，但二维与绝对位置误差更大，不能声称原图路线整体胜出。完整结果见 `paired_native_space_comparison.json`，可用仓库入口 `compare_native_rectified_predictions.py` 重新计算。

## 真实双手独立评测

从 192916 合格双手同步时刻均匀抽取 32 个时刻 × 2 相机 × 2 手，共 **128 个手实例、64 张图片**。使用标签框和已知物理手身份，评价姿态网络；**尚未接入用户未来原图 YOLO 的检测框，不评价检测召回或左右手分类准确率**。

| 相同真实双手原图；误差越低越好 | 原始预训练权重＋当前 KB 几何适配 | 共同 coarse_start 初始权重 | 原图 v3 短程训练 |
| --- | ---: | ---: | ---: |
| 腕根相对 MPJPE，mm | 52.59 | 71.52 | 62.62 |
| 原图二维平均误差，px | 164.29 | 66.01 | 32.15 |
| 腕根位置误差，mm | 167.52 | 127.40 | 72.13 |

与共同训练初始化相比，v3 的 MPJPE 下降约 **12.4%**、二维误差下降 **51.3%**、腕根误差下降 **43.4%**。与原始预训练权重相比，二维误差下降 **80.4%**，但 MPJPE 增大约 **19.1%**，不能认定整体优于原始权重。

v3 左手/右手 MPJPE 为 **55.70/69.55 mm**，原图二维误差 **29.94/34.36 px**，腕根误差 **74.99/69.26 mm**。目前手指姿态、绝对深度与困难帧仍需改善；80 步小样本实验也不足以确认最终泛化能力。

查看结果：

```text
evaluations/real_dual_original/             # 指标、预测、同帧双手叠加图
evaluations/real_dual_coarse_start/
evaluations/real_dual_native_v3/
evaluations/real_dual_video/real_dual_original_vs_native_v3.mp4
```

对照视频上排为原始权重、下排为 v3，左列 cam0、右列 cam1；绿色为拟合标签、红色为预测，框来自标签。视频为 32 个抽样时刻、5 FPS、6.4 秒的概览，已验证解码帧数。完整标签视频用于检查更连续的动作。

## 复现入口

完整标签操作流程与每步参数见 [EGO + NOKOV 双手原图标签流程](../../../ego_hand_system/docs/EGO_NOKOV_NATIVE_LABEL_PIPELINE_zh.md)。本次标签子命令保存在 `real_dual/pair_20261111_192916_native_labels/commands.json`，各评测命令保存在 `evaluation_commands.json`。

训练配置为 `native_train_v3.yaml` 和 `rectified_train_v3.yaml`；重跑时先复制配置并将 `paths.output` 改为新目录：

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py \
  --config /path/to/copied_native_config.yaml
```

原图评测示例，`--output` 同样使用新目录：

```bash
TASK_RESULT=/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111
TASK_DUAL="$TASK_RESULT/real_dual/pair_20261111_192916_native_labels/export/merged"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/p3/miniforge3/envs/ego-hand/bin/python evaluate_glove.py \
  --root "$TASK_DUAL" --manifest "$TASK_DUAL/evaluation_128.jsonl" \
  --checkpoint "$TASK_RESULT/coarse_start.ckpt" \
  --adapter "$TASK_RESULT/train_native_v3/best_trainable.pt" \
  --camera-model physical_kb --batch-size 2 --device cuda:0 \
  --output "$TASK_RESULT/evaluations/new_native_evaluation"
```

原图投影、物理左右手转换、裁剪与特征采样、权重恢复、旧格式隔离和训练配置回归均已运行。该验收证明当前原图标签与姿态训练链路可以执行；四目独立观测优化、弱同步录制修复、原图检测器接入及最终模型精度验收仍需后续工作。
