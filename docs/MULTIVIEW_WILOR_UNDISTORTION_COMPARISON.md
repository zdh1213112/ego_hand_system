# 多路鱼眼 WiLoR：原图与去畸变对比

新增入口 `scripts/run_multiview_wilor_undistorted_experiment.sh` 用于“去畸变后检测、估计、融合”。
原入口 `scripts/run_multiview_wilor_experiment.sh` 默认仍处理原图。两者共享检测阈值、双手性 WiLoR 推理、
锚点选择、融合、时序处理及渲染实现，避免两份实验脚本逐渐产生不同默认参数。

## 运行同一段数据

在项目根目录执行，`SESSION` 换为含四路 MP4、PTS CSV 和厂商 KB 标定的 EGO4 目录；
环境应与原脚本相同（下面使用本机 `wilor`，原脚本默认环境名为 `ego-hand`）。
每组参数使用独立的输出目录。第一次建议 60 帧，确认结果后再设 `--max-frames 0` 运行全段。

```bash
SESSION=/path/to/EGO4_SESSION
COMMON=(--session "$SESSION" --conda-env wilor --max-frames 60
        --detector models/wilor/detector_1009.pt --final-video)

bash scripts/run_multiview_wilor_experiment.sh \
  "${COMMON[@]}" --output output/wilor_raw_comparison

bash scripts/run_multiview_wilor_undistorted_experiment.sh \
  "${COMMON[@]}" --output output/wilor_undistorted_fov120 \
  --undistort-fov-deg 120 --original-video

conda run --no-capture-output -n wilor \
  python scripts/compare_multiview_wilor_undistortion.py \
  --raw-run output/wilor_raw_comparison \
  --undistorted-run output/wilor_undistorted_fov120 \
  --output output/wilor_comparison_fov120.json
```

GEN 多目 MCAP 同样支持：将共同参数中的 `--session "$SESSION"` 改成 `--mcap /path/to/recording.mcap`；
默认六目、`camera2` 为同步参考，也可沿用原脚本的 `--cameras` / `--reference-camera`。
EGO4 默认四目、`camera0` 为同步参考，锚点为 `camera0 + camera1`。
`--gpu-profile`、`--batch-size`、`--glove-marker-assist`、`--sam3-glove-assist`、
`--allow-single-hand-anchors`、`--no-video`、`--stop-after-inference` 等原入口参数均可透传。
比较融合指标时不要使用 `--stop-after-inference`。

新入口额外支持：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--undistort-fov-deg` | 120 | 针孔输出的水平视场角，范围 `(0, 175)` 度；各目相同。 |
| `--undistort-width` / `--undistort-height` | 保持原分辨率 | 必须同时给出正偶数。首次比较建议不改分辨率。 |
| `--original-video` | 关闭 | 额外生成“原始视频背景 + 去畸变处理结果”的混合 3D/2D 视频。 |

可另跑 FOV 100、140 等实验，每次只改变 FOV 并使用新的输出目录。
视场越窄，越可能裁掉边缘手部；视场越宽，边缘拉伸和无效区域越明显。
一个针孔视图无法保留完整的超广角鱼眼视场。无效映射像素以黑色填充，掩码保存在各相机目录。

## 几何与数据约定

流程为：原始归一化/同步 → 每目独立去畸变 → YOLO 手检测 → 双手性 WiLoR 关节估计 →
用新标定反投影/三角化 → 原有融合与时序处理 → 诊断/最终视频。

- 输入支持项目已有的 KB 和 **GEN 定义的 Double-Sphere** 模型，使用同一投影实现计算反向采样映射。
- 输出标定明确为 `PINHOLE`，畸变数组为空。不能把 KB 的四个系数清零来冒充针孔模型。
- 新焦距 `fx = fy = width / (2 tan(FOV/2))`，主点位于图像中心；保留原相机光轴与 `T_base_camera`，
  不做双目极线旋转，因此融合结果仍在原 rig/base 坐标系中，单位 m。
- 原同步表逐字节保留，帧索引、纳秒时间戳、同步参考及相机编号不变。
  每路解码/写入至最后一个被选中的原始帧，包含同步表跳过的帧，避免索引错位。
  视频只是顺序帧容器；真实时间以 CSV 为准，不从重编码的视频 PTS 推断。
- 去畸变视频使用无损 FFV1/Matroska，避免引入额外的有损编码变量；插值本身仍会改变像素。
  OpenCV 需支持 FFV1 写入，长序列磁盘占用可能明显增大。
- 默认下游阶段，包括手套辅助、SAM3、融合和渲染，均使用去畸变视频及对应针孔标定。
  WiLoR 保持原有裁剪/预测投影逻辑；`camera_translation` 仍是模型预测量，不能直接视为标定后的绝对手部位姿。
  本实验与原脚本一样，最终多视角 3D 来自标定射线融合，不额外输出 MANO 网格。

启用 `--original-video` 时增加一个独立的显示步骤：处理与可信度判定仍在去畸变坐标中完成，
最终使用原始视频帧和原始 KB/DS 标定绘制结果。这个显示步骤不修改预测、融合或比较统计。

## 输出与恢复运行

新实验目录包含：

| 路径 | 内容 |
|---|---|
| `run_config.json` | 原实验参数及 `image_preprocessing`，防止复用不同 FOV 的缓存。 |
| `normalized_multiview/` | 原始归一化数据。 |
| `undistorted_multiview/` | 去畸变视频、针孔标定、原同步表。 |
| `undistorted_multiview/source_calibration/` | 原 KB/DS 标定，用于将观测映射回原图。 |
| `undistorted_multiview/undistortion.json` | 参数、源文件签名、新内参、每目有效像素比例。 |
| `undistorted_multiview/cameras/*/valid_mask.png` | 白色表示有效采样区域，黑色表示无效区域。 |
| `wilor_multiview/` | 去畸变图像上的检测及 WiLoR 双手性假设。 |
| `fusion_multiview*/` | 融合 JSONL、统计及 `diagnostic_Nview.mp4`；启用 `--final-video` 时另有最终视频。 |
| `fusion_multiview*/final_on_original_Nview.mp4` | 启用 `--original-video` 时生成：原图背景上的去畸变处理结果。 |
| `fusion_multiview*/final_on_original_Nview.json` | 原图来源、帧数、坐标转换方式、可信度判定域和显示统计。 |

原样重复命令可复用完整阶段。改变 FOV、分辨率、检测器或融合参数须另设输出目录。
去畸变阶段使用临时目录，成功后才发布结果；缓存校验包括源清单、标定、时间表哈希及视频大小/修改时间。
`EGO_WILOR_UNDISTORT_*` 是新入口向共享脚本传递设置的内部环境变量，运行原图基线时不要手工设置它们。

## 在原始输入视频上显示去畸变结果

在原来运行去畸变实验的命令末尾追加 `--original-video` 即可。使用同一个输出目录和相同计算参数时，
已完成的检测、WiLoR 和融合阶段直接复用，只增加原图显示视频。`--final-video` 控制原有去畸变视图的最终视频，
`--original-video` 控制额外的原图版本，可以同时使用；`--no-video` 和 `--stop-after-inference` 会禁止视频输出。
原图版本使用混合 3D/2D 渲染器，即使另外通过 `--final-renderer fused` 选择了去畸变视图的纯融合视频也是如此。

- 融合 3D：用原始外参变换到各相机，再通过原始 KB/DS 投影到鱼眼像素。
- 2D 回退：先在去畸变坐标中完成跟踪、平滑和短时保持，再通过“针孔射线 → 原始相机投影”转换。
- 检测框：裁到有效输入画幅，采样四条边并映射，取映射边界的轴对齐包围框；非线性转换不能只用两个角点。
- 超出去畸变画幅或原始图像有效范围的关节隐藏，不会因为原图背景更宽就恢复被裁掉的观测。
- 3D 的像素残差门限和可信度判定保持在去畸变坐标中执行，不会因显示坐标变化而改变阈值含义。
- 渲染前核对数据来源、同步表、相机编号、原始标定及光轴/外参；不匹配即停止，避免错帧或错误回投影。

也可直接只调用渲染器，不加载检测器或 WiLoR 权重：

```bash
conda run --no-capture-output -n wilor python scripts/render_hybrid_multiview.py \
  --dataset output/wilor_undistorted_fov120/undistorted_multiview \
  --original-dataset output/wilor_undistorted_fov120/normalized_multiview \
  --predictions output/wilor_undistorted_fov120/wilor_multiview \
  --fusion output/wilor_undistorted_fov120/fusion_multiview \
  --output output/wilor_undistorted_fov120/fusion_multiview/final_on_original_4view.mp4 \
  --min-joint-inlier-views 4 --residual-percentile 75 --residual-threshold-px 12 \
  --wrist-threshold-px 15 --max-condition-number 100 --max-stereo-difference-mm 20
```

手套辅助或非默认手性模式下，将 `--predictions` / `--fusion` 换为该实验实际目录。
直接渲染时应沿用该实验的显示阈值；上面列出的是入口脚本默认值。

## 如何判断效果

比较工具拒绝来源、同步表、原标定或预处理之外的已记录配置不同的实验。生成 JSON 汇总和同名 CSV：

1. **检测覆盖**：每目有检测的帧数、比例、检测框数；同一个框的两个手性假设只计一次。
2. **融合覆盖**：含有限 3D 结果的帧、手、关节数量，以及完整 21 关节手数；接受帧数单列。
3. **统一坐标残差**：将去畸变 2D 观测沿射线投回原始鱼眼图像，再与最终融合 3D 的原图投影比较。
   输出两组全量统计，以及相同 `(sync_index, side, camera, joint)` 上的配对统计和逐项 CSV；
   `delta_px = 去畸变残差 − 原图残差`，负值代表该项内部重投影残差下降。
4. **可视检查**：对齐两组视频的帧号，特别看画面边缘、遮挡、快速运动及左右手身份。
   同时检查有效区域掩码。有效输出像素比例不代表保留了多少原始视场。

这些指标没有人工真值：检测变多不等于准确率提高，重投影误差变小也不等于绝对 3D 精度提高。
两组可能选择不同的框或手性，配对结果仍需检查身份。全量残差的样本组成不同，应结合配对统计和覆盖率判断，
避免只剩少量容易帧时误认为更准。若要下精度结论，应补人工标注或同一时刻的可靠 3D 真值。

共享管线中的像素阈值保持同一数值，但去畸变后角度尺度会变化；此实验评估的是保留既有参数时的整体流程效果，
不能把差异全部归因于模型本身对畸变的敏感度。当前实现并不预设去畸变一定有提升。

## 验证

```bash
conda run --no-capture-output -n wilor python -m unittest discover \
  -s tests -p 'test_multiview_undistortion.py' -v
```

测试包含 OpenCV KB 映射对照、GEN-DS 射线一致性、针孔三角化、真实 FFV1 视频往返、
不连续同步帧与纳秒时间戳保留、缓存配置隔离、比较口径和 shell 下游数据集路由。
原图回投影测试位于 `tests/test_multiview_original_projection.py`，覆盖 KB/DS 几何、视场屏蔽、
检测框边界、源数据校验和真实原图帧上的 3D/2D 显示；脚本路由测试还验证补生成视频时不重跑推理。
