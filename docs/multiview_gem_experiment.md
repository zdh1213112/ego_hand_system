# 多视角 GEM-X 手部实验

`scripts/run_multiview_gem_experiment.sh` 复用 `run_multiview_wilor_experiment.sh` 的输入、检测和原有后处理流程，以 GEM-X EgoHand 替代 WiLoR 的姿态估计和 MANO 网格重建，并新增共享 SOMA 多视角拟合。**本入口直接使用输入原始图像，不执行图像去畸变**。

共同流程依次为：输入归一化和同步、原图手部检测、每个检测框的左右手两种 GEM 假设、可选手套标记与 SAM3 辅助。相机选择、检测阈值及视频开关沿用 WiLoR。默认 `shared` 模式直接执行逐手多视角关联和共享 SOMA 拟合，按需渲染共享视频；跳过原流程融合、时序处理及诊断视频，也不检查或读取原流程融合缓存。`independent` 保留原有骨架融合、时序处理与独立网格视频；`both` 保留两个分支并复用同一次检测与 GEM 推理。共享模式完成后切换到 `both` 可复用预测，再补齐原流程结果。

## 使用

```bash
cd /home/lff/xense_project/ego_hand_system
bash scripts/run_multiview_gem_experiment.sh \
  --session /path/to/ego4_session \
  --output output/gem_raw_experiment \
  --detector models/wilor/2.0_epoch150_1600_1111_best.pt \
  --conda-env ego-hand --device cuda --max-frames 60 \
  --gem-mode shared --final-video
```

将 `--max-frames` 改为 `0` 可处理全部同步帧。也支持 WiLoR 入口的 `--mcap FILE` 和其他参数，运行 `bash scripts/run_multiview_gem_experiment.sh --help` 查看。输出目录可以重用，但已有 `run_config.json` 中的配置和模型路径必须一致；不完整阶段需使用新目录。`--stop-after-inference` 在推理后停止；`--no-video` 跳过视频；`--final-video` 额外生成最终视频。

`--gem-mode` 有三个选项：

| 模式 | 输出 |
| --- | --- |
| `shared`（默认） | 同一份 SOMA 骨架和网格通过各相机标定重投影，无独立视角回退 |
| `independent` | 原有流程，各相机自己的 GEM 网格；骨架采用原来的 hybrid/fused 规则 |
| `both` | 同时生成上述两种视频，便于比较 |

共享拟合默认 `--shared-steps 100 --shared-batch-size 32`，可按需要调整；这两个参数与 GEM 网络推理的 `--batch-size` 不同。共享输出会校验输入文件和拟合配置，配置改变时要求使用新输出目录。仅切换显示模式可以复用已有推理、融合和共享拟合结果。

GEM-X 默认从 `GEM_X_ROOT` 或 `/home/lff/xense_project/GEM-X` 查找，也可以指定 `--gem-root`、`--checkpoint`、`--soma-assets`、`--gem-python`。检测器默认为 WiLoR 实验的 `models/wilor/detector.pt`，可通过 `--detector` 指定其他同格式权重。安装环境可运行 `scripts/setup_gem_hand_environment.sh`。

`--gpu-profile compatible` 是默认 FP32 配置。`rtx5090d` 沿用基线的检测器置信度、批量检测和每类框数量设置，并对 GEM backbone/head 启用 FP16 autocast；相机求解和 SOMA 几何继续在 FP32 下计算，精度设置记录在运行配置中。`--compile-backbone` 可编译 GEM 的 backbone，GEM 裁剪和模型前向使用其自己的实现。`--batch-size` 控制 GEM 每次处理的手框假设数量；遇到 CUDA 显存不足会自动减小该批量。WiLoR 专用的裁剪并行参数 `--preprocess-workers` 对 GEM 模型裁剪无效。

## 相机、精度和结果

图像、检测框和 2D 关键点始终使用原始图像像素。GEM-X 的普通 Hand Pose 接口按针孔相机处理图像；对 KB/DS 鱼眼，本流程直接将原始标定矩阵作为针孔近似传入 GEM，**不进行镜头校正**。GEM 的投影结果是近似原图坐标，尤其在图像边缘可能存在较大误差。后续多视角融合仍使用输入的原始 KB/DS 标定进行投影与反投影，不把近似针孔模型当作实际标定。GEM 的相机空间深度及网格需要按这种近似解释。

每个真实检测框生成左右各一条 GEM 假设，检测框、置信度、检测手型和索引与 WiLoR 推理接口一致。原有骨架融合器仍按原规则选择；共享拟合采用下面的逐手关联。GEM SOMA 的 25 点映射为显示使用的 21 点：腕、拇指四点和其余四指的 MCP、PIP、DIP、指尖。两种手部模型的关节是近似对应。

```text
输出目录/
├── run_config.json              # 包含 GEM 权重和姿态相机近似信息
├── normalized_multiview/        # 原始视频、原始标定、同步表
├── gem_multiview/cameraN/
│   ├── predictions.jsonl        # 基线融合器所需的检测框及双假设 2D 点
│   └── gem_pose.pt              # 每帧双假设 SOMA 参数、25 点、网格和拓扑
├── fusion_multiview/            # 原基线融合结果和独立模式视频
└── fusion_multiview_shared_gem/
    ├── run_config.json          # 共享拟合输入指纹和参数
    ├── shared_pose.pt           # 每帧每手唯一的 SOMA 参数、变换、骨架、网格
    ├── accepted.jsonl          # 从共享模型导出的 21 点和质量记录
    ├── rejected.jsonl          # 关联/拟合失败的原因
    ├── summary.json
    ├── final_shared_4view.mp4
    ├── final_shared_4view.json
    └── final_shared_4view.projections.jsonl # 每相机投影及共同 model ID
```

GEM 的 `gem_pose.pt` 顶点和关节已包含相机平移，单位为米，不应再次叠加 `camera_translation`。原有 `fusion_multiview` 输出仍只有骨架；共享骨架和网格位于单独的 `fusion_multiview_shared_gem`。手套标记、SAM3 辅助及其他融合目录命名沿用基线规则，共享目录名在相应融合目录后添加 `_shared_gem`。

## 共享 SOMA 拟合与一致性

`fit_shared_gem_multiview.py` 将每只手独立关联，不要求两台相机都同时看到双手。它使用原始 KB/GEN-DS/针孔标定反投影像素射线，以几何一致性和检测手型软约束选择同一只手的多视角观测。同一个检测框不会被左右两只手重复使用。

固定的 SOMA 形状参数来自本段候选 GEM 参数的逐侧稳健中位数。每帧再联合优化手指旋转、全局旋转、平移和受限整体尺度，使**同一个模型**在所有选中视角下解释 2D 关键点，并使用三角化关节与姿态先验约束。模型通过后，骨架与网格从同一次 SOMA 前向计算生成，使用同一个全局变换。原有独立模式的骨架时序修正不会单独施加到共享骨架上。

默认要求至少两个支持视角、有效视差，以及各支持视角至少 16 个关节的误差不超过 30 像素、中位误差不超过 21 像素。无法满足条件的帧显示 `NO VALID SHARED 3D`，不插入单视角姿态或无限外推结果。

`render_shared_gem_multiview.py` 对每台相机重复使用同一份 `joints_soma_base_m` 和 `vertices_base_m`，通过其外参与原始镜头模型投影，不使用每视角平移、独立形变或 2D 回退。检测框仍为真实的各视角检测框。投影日志记录共同的 `sync_index:side` 模型 ID，方便逐点审计。

GEM 与 WiLoR shared 直接共用此渲染器中的 `draw_shared_skeleton` 和 `clip_projected_bone`，不维护两套骨架绘制代码。有效但越界的投影点保留用于连线，线段裁到画面边界；关键点圆点只画在真实的画内位置，无效投影不参与连线。渲染元数据中的 `skeleton_render_style` 参与缓存判断，旧视频会自动重新生成。已有共享三维结果时，可仅更新视频，无需重新检测、推理或拟合：

```bash
conda run -n ego-hand python scripts/render_shared_gem_multiview.py \
  --dataset OUTPUT/normalized_multiview \
  --shared OUTPUT/fusion_multiview_shared_gem \
  --output OUTPUT/fusion_multiview_shared_gem/final_shared_4view.mp4 \
  --cameras camera0 camera1 camera2 camera3 --max-frames 0 --video-codec auto
```

其中 `OUTPUT` 替换为已有实验输出目录。

原始 KB 鱼眼现在采用完整球面角投影和匹配的反投影，在径向多项式连接光轴的单调分支内允许 `z<=0` 的射线。有效域止于第一个径向驻点或 `theta=pi`；零向量、反向光轴及不可逆分支仍无效。图像越界另做裁切，针孔相机的正深度条件保持不变。共享 SOMA 拟合也使用 KB 角域及到相机的距离约束，避免把合法的侧后方鱼眼观测强行推回正 z。

共享视频配置记录 `camera_projection_style=kb_spherical_monotonic_v2`，运行上面的渲染命令会更新旧视频。重渲染复用原有三维参数、关联和质量记录；若要用新反投影及约束重新拟合，需为拟合指定新输出目录，原有配置校验会阻止误用旧拟合缓存。

共享模式保证几何来源一致，不代表真实姿态误差为零。检测错误、GEM 的原始鱼眼针孔近似、标定误差、遮挡和视差不足仍会影响拟合；视频投影也不包含场景级遮挡判断。当前按左右手维护最多两只手，未实现多人身份跟踪。

## 最终效果视频

使用 `--final-video`，默认共享视频位于 `fusion_multiview_shared_gem/final_shared_4view.mp4`，显示原始检测框、共享 21 点骨架、共享半透明网格。每格宽 960 像素，相机数量不为四时文件名相应变化。

`independent` 或 `both` 同时保留 `fusion_multiview/final_only_4view.mp4`。这个视频沿用 WiLoR hybrid 的骨架可信度、单视角回退、轨迹平滑与保持规则，网格仍来自该视角独立的 GEM 假设，与共享视频有明确区别。

网格按最终显示的手型着色：左手蓝色 `#1964ff`、右手橙色 `#ff7b24`。透明度为 55%，对应不透明度 `alpha=0.45`；检测框和骨架继续用原有左右手配色。脚本会检查网格样式版本，重新执行 `--final-video` 可更新旧配色的视频，无需重新进行推理和融合。

同名 `.json` 保存视频帧数、融合/回退次数、网格配色及实际网格绘制数量。`--final-renderer fused` 为原有的骨架渲染入口，以融合骨架为主，并按配置进行 2D 回退，不包含网格；需要上述三种叠加内容时使用默认的 `hybrid`。

当前支持范围：

| 显示内容 | 当前实现 |
| --- | --- |
| 各相机分别显示本视角的 GEM 网格 | `--gem-mode independent` |
| 同一组 SOMA 三维关节和同一份网格投影到各相机 | `--gem-mode shared`（默认），通过共同质量门限后显示 |
| 两种视频同时输出 | `--gem-mode both` |

`--final-renderer` 只控制独立模式中保留的旧骨架渲染路径；共享模型请选择 `--gem-mode shared`。
