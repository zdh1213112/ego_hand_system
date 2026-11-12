# WiLoR 共享三维骨架和网格

`scripts/run_multiview_wilor_experiment.sh` 支持 `--wilor-mode legacy|shared|both`，默认使用 `shared`。

当前 `final_only_4view.mp4` 已恢复为原流程骨架＋各相机原始 WiLoR 网格；`final_shared_4view.mp4` 保留共享三维骨架＋网格。两者仅网格配色和透明度一致，不再复用同一份网格。

| 模式 | 三维与显示行为 |
| --- | --- |
| `legacy` | 保留原有骨架融合、时序处理与二维回退；默认 hybrid 视频叠加各路原始 WiLoR 网格 |
| `shared`（默认） | 为每帧每只手拟合一份 MANO 参数，骨架和网格共同投影到各路，没有独立视角回退 |
| `both` | 复用同一次检测与 WiLoR 推理；`final_only_Nview.mp4` 为原流程骨架＋各路原始网格，`final_shared_Nview.mp4` 为共享三维骨架＋网格 |

共享模式直接处理原始图像，原图入口不做去畸变。三维拟合与显示使用标定中的相机外参和原始 KB、GEN-DS 或针孔模型。检测框直接读取各相机原始检测结果，与共享拟合是否采用该路观测无关；未参与融合的检测和没有有效共享三维模型时的检测也会显示。

框的坐标、左右手类别及置信度保持检测器输出；画面文字恢复为之前的 `FINAL L / FINAL R`，不显示置信度，以统一对比视频的标注风格。这里的 `FINAL` 仅沿用共享结果的显示标签，不表示该检测参与了融合；框始终来自检测器。原始置信度仍完整保存在投影审计文件中。同一检测框的 WiLoR 左右手两种姿态假设只画一次；不按共享手的数量筛框，不平滑或扩展框，也不从三维骨架重新生成框。共享渲染器使用 `--detector-predictions RAW_WILOR_PREDICTIONS`，主脚本自动传入原始预测目录，原始检测来源文件的 SHA256 写入渲染元数据。

## 本次输入的完整命令

```bash
bash scripts/run_multiview_wilor_experiment.sh \
  --session /home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO-4B_AZLCA610091_20261111_193031/part0001 \
  --detector models/wilor/2.0_glove_2900_epoch150_2906.pt \
  --external-boxes output/wilor_ego4_20261111_193031_part0001_glove_20261112/detector_cache \
  --output output/wilor_ego4_20261111_193031_part0001_glove_20261112 \
  --conda-env ego-hand --device cuda --max-frames 0 \
  --gpu-profile rtx5090d --batch-size 4 --frame-batch-size 4 \
  --preprocess-workers 2 --max-detections-per-class 0 \
  --compile-backbone 0 --fusion-workers 2 \
  --wilor-mode both --final-video --no-diagnostic-video
```

`rtx5090d` 是现有批处理和混合精度配置的名称，也可在当前 RTX 5060 上使用上述显存较小的批大小。两种 WiLoR 结果使用完全相同的推理文件；保存 MANO 参数不会额外执行另一套网络推理。默认 `compatible` 保留 FP32 推理。`--no-diagnostic-video` 只跳过诊断视频，仍输出最终视频。

`part0001` 内的四路视频和时间戳按选定分段读取，标定与元数据从录制父目录读取，不拼接其他分段。

本次执行期间，检测权重改名为 `models/wilor/2.0_glove_2900_epoch150_2906.pt`，用户已确认是同一份权重。GEM-X 已经使用它完成全部检测，因此本次 WiLoR 直接复用原模型的完整检测结果：四路分别为 6,040、6,008、5,962、5,900 个物理检测框。仅导出原始检测框、置信度和检测手侧，WiLoR 独立计算自己的关键点与 MANO 参数，不导入 GEM 姿态。新旧文件名及当前 SHA256 记录在输出目录的 `detector_identity.json`。导出命令如下，需在上面的 WiLoR 命令前执行一次：

```bash
conda run -n ego-hand python scripts/export_cached_multiview_detections.py \
  --dataset output/gem_ego4_20261111_193031_part0001_glove_20261112/normalized_multiview \
  --predictions output/gem_ego4_20261111_193031_part0001_glove_20261112/gem_multiview \
  --output output/wilor_ego4_20261111_193031_part0001_glove_20261112/detector_cache
```

缓存带有来源、同步帧、计数与 SHA256 校验；运行配置记录每个缓存文件的 SHA256。`--external-boxes` 是 WiLoR 的显式选项，使用时不重新加载 YOLO 权重；新录制可去掉此选项，改用 `--detector models/wilor/2.0_glove_2900_epoch150_2906.pt`，现场执行检测。检测缓存必须对应相同输入图像及坐标系。上面的缓存复跑命令保留原模型路径作为历史来源，与本次 `run_config.json` 一致。

输出完成后生成对比统计与左右并排视频：

```bash
conda run -n ego-hand python scripts/compare_shared_wilor_multiview.py \
  --run output/wilor_ego4_20261111_193031_part0001_glove_20261112 \
  --output output/wilor_ego4_20261111_193031_part0001_glove_20261112/comparison_native_vs_shared.json \
  --video output/wilor_ego4_20261111_193031_part0001_glove_20261112/comparison_native_vs_shared.mp4
```

对比视频左侧为原有 hybrid 骨架和各路独立 WiLoR 网格，右侧为共享 MANO 骨架与网格。误差配对只使用同一同步帧、手侧、相机、关节、检测框及相同二维目标的样本。

各路原始网格由保存的 WiLoR MANO 参数直接重建，保留原始相机平移与 WiLoR 的焦距投影，不进行多视角拟合、形状调整或额外对齐。加速推理的恢复保留原始半精度 MANO 和推理批次划分；全片 47,820 个假设的关节重建差为 0。检测/假设选择跟随原流程显示的骨架，短时保持的骨架使用原来源帧的网格。左右手仅沿用蓝／橙配色、55% 透明度。对应选项是 `render_hybrid_multiview.py --wilor-native-mesh`，主脚本默认 hybrid 最终视频已自动使用；旧缓存缺失 MANO 参数时会明确报错，不会替换成共享网格。

### 历史对照：使用相同网格投影比较骨架

此前生成的 `final_legacy_shared_mesh_4view.mp4` 使用原流程骨架和共享网格，保留为历史对照。主脚本已不再将它覆盖到 `final_only_4view.mp4`。最初的无网格骨架基线保存在 `final_legacy_skeleton_only_4view.mp4`。

该视频的骨架仍可能采用各路二维回退，不能称为骨架与网格共同三维一致。真正同时保持骨架与网格三维一致的文件仍为 `fusion_multiview_shared_wilor/final_shared_4view.mp4`，通过 `--wilor-mode shared` 或 `both` 生成。

仅重新渲染现有结果时，可以在原 `render_hybrid_multiview.py` 命令上增加 `--shared-mano-mesh FUSION_SHARED_DIR --tile-width 960` 并指定新输出文件，无需重新检测或拟合。各帧各路的网格来源与投影指纹保存在对应的 `.mesh_projections.jsonl`。两种渲染共用 `shared_mesh_overlay.py`，网格不会随原流程骨架的二维回退而改变。

生成相同网格的左右对比视频：

```bash
conda run -n ego-hand python scripts/compare_shared_wilor_multiview.py \
  --run output/wilor_ego4_20261111_193031_part0001_glove_20261112 \
  --legacy-video output/wilor_ego4_20261111_193031_part0001_glove_20261112/fusion_multiview/final_legacy_shared_mesh_4view.mp4 \
  --output output/wilor_ego4_20261111_193031_part0001_glove_20261112/comparison_same_mesh.json \
  --video output/wilor_ego4_20261111_193031_part0001_glove_20261112/comparison_same_mesh.mp4
```

## 输出

```text
OUTPUT/
├── run_config.json
├── normalized_multiview/
├── wilor_multiview/                    # 同一份检测、二维假设与原始 MANO 参数
├── fusion_multiview/
│   ├── accepted.jsonl                 # 原有骨架融合结果
│   ├── final_only_4view.mp4            # 原有骨架＋各路原始 WiLoR 网格
│   ├── final_legacy_skeleton_only_4view.mp4 # 归档的原有无网格 hybrid 视频
│   └── final_legacy_shared_mesh_4view.mp4 # 历史对照：原有骨架＋共享网格
└── fusion_multiview_shared_wilor/
    ├── shared_pose.pt                 # 每帧每手共享参数、变换、21 个关节和 778 个顶点
    ├── accepted.jsonl
    ├── rejected.jsonl
    ├── summary.json
    ├── run_config.json
    ├── final_shared_4view.mp4          # 改动后：共享 MANO 骨架与网格
    └── final_shared_4view.projections.jsonl
```

辅助分支使用原有融合目录名，再添加 `_shared_wilor`。共享拟合默认 `--shared-steps 100 --shared-batch-size 32`。推理完成后可切换 `shared` 与 `both` 复用结果；旧推理如果没有保存 MANO 参数，脚本会说明原因并要求新输出目录，不覆盖旧结果。

## 三维一致性的实现与边界

1. 每个检测框保留左右手两种 WiLoR 假设，并额外保存 MANO 旋转、形状参数。保留 WiLoR 的右手规范模型；物理左手只镜像一次，并翻转网格三角形绕序。
2. 用标定射线和重投影误差关联各路同一只手，每只手独立关联，不要求各路同时看见双手。
3. 逐帧优化共同的手指关节旋转、全局旋转、平移与有界尺度。每侧使用序列内稳健聚合的形状参数。同一次 MANO 前向生成全部骨架关节和网格顶点。
4. 各相机使用完全相同的三维数组，仅相机变换和投影不同。投影日志记录相同的 `sync_index:side` 模型编号。共享结果不可用时标记 `NO VALID SHARED 3D`，不独立修改任何相机的骨架或网格。

视频叠加检测框、21 点骨架与半透明网格，左蓝 `#1964ff`、右橙 `#ff7b24`，透明度 55%（不透明度 0.45）。

共享视频的骨架连接线会裁切到图像边界：投影有效但位于画面外的关节点仍用于连线，仅隐藏其关节点圆点；不会因为一个端点越界就丢掉整条线。相机后方或非有限的无效投影仍不参与绘制。此修正只影响边缘显示，不改变三维关节、网格、相机投影或拟合结果。渲染元数据中的 `skeleton_render_style=valid_projection_segment_clipping_v2` 会使旧版渲染缓存自动更新；投影日志记录 `drawn_bone_count` 和 `clipped_bone_count`。

三维一致性指显示内容来自同一模型，不等于真实三维精度。检测或二维姿态错误、原图畸变下的网络误差、遮挡、窄基线和错误关联仍可能影响拟合。当前最多关联一只左手和一只右手，不进行多人的身份跟踪；共享参数按帧拟合，不套用原有骨架独立平滑，以免骨架与网格脱离。对比时应同时看有效帧覆盖率、回退数量和重投影误差；原流程的二维回退与共享三维模型不是相同的误差统计对象。
