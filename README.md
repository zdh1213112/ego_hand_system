# EGO Hand Inference

Orbbec EGO 四目离线手部推理专用项目。输入四路 MP4 录制目录或 Orbbec MCAP，执行检测、姿态推理、多视角关联和共享三维模型拟合，自动生成最终四目视频。

仅使用 WiLoR/MANO。每个同步时刻、每只手只有一份三维骨架和网格，使用原始鱼眼标定回投四路画面。视频为 2×2 拼接，叠加检测框、21 点骨架和半透明网格。没有有效模型时，视频保留原画面和状态标识，三维结果保留空帧。

## 输入

| 相机 | MP4 流名 | MCAP topic | YAML 相机 ID |
| --- | --- | --- | --- |
| camera0 | left | /camera/left | cam_0 |
| camera1 | right | /camera/right | cam_1 |
| camera2 | right_side | /camera/right_side | cam_2 |
| camera3 | left_side | /camera/left_side | cam_3 |

**MP4**：录制根目录或指定 `part0001`，包含四路 `*_camera_<流名>.mp4` 与对应 `*_pts.csv`；支持单个 `*_camera_<流名>_part*.mp4` 分段。选择分段时，标定和元数据从录制父目录读取，视频与时间戳只读取指定分段。一个输入不能包含多个分段，请分别运行并使用不同输出目录。

**MCAP**：包含上述四路 H264 topic 的 `.mcap` 文件。四路视频无损转封装；共享重建必须具有匹配的四目标定。

默认从录制元数据 `files.calibration_camera` 读取配套 YAML；也识别唯一的 `*_calibration_camera.yaml` 或 `*_params_cam_*.yaml`。可用 `--calibration` 显式指定 YAML，或包含 `camera0.json`～`camera3.json` 的目录。缺少标定、相机 ID 或分辨率不匹配会报错。

按 camera0 时间戳匹配其余三路，不复用同一路帧，默认容差 1500 微秒。只处理四路都有匹配帧的时刻。三维坐标使用标定参考系、单位米；时间戳沿用录制，不保证是 UTC。

## 环境与模型

需要 Conda、Git 和 FFmpeg：

```bash
git submodule update --init third_party/WiLoR
bash scripts/setup_python_environment.sh
```

准备以下本地资产：

```text
models/wilor/detector.pt
models/wilor/wilor_final.ckpt
models/wilor/model_config.yaml
models/mano/MANO_RIGHT.pkl
```

可用 `--detector`、`--checkpoint`、`--model-config`、`--mano-model-dir` 指定路径。默认环境为 `ego-hand`，可加 `--conda-env NAME`；设置 `EGO_HAND_PYTHON=/path/to/python` 可直接使用已有解释器。资产说明见 [models/README.md](models/README.md)。

## 运行

WiLoR + MP4：

```bash
bash scripts/run_multiview_wilor_experiment.sh \
  --session /path/to/recording/part0001 \
  --detector /path/to/hand_detector.pt \
  --output output/wilor_result
```

WiLoR + MCAP：

```bash
bash scripts/run_multiview_wilor_experiment.sh \
  --mcap /path/to/recording/part0001/recording_part0001.mcap \
  --calibration /path/to/recording/recording_params_cam_1920x1200.yaml \
  --detector /path/to/hand_detector.pt \
  --output output/wilor_mcap_result
```

支持 MP4 和 MCAP，也可用 `--input PATH` 自动判断 MP4/MCAP 录制根目录、分段目录或 MCAP 文件。MCAP 目录中有多个分段时须指定分段或文件。不同录制需使用不同输出目录。

默认 `--max-frames 0` 处理完整录制，始终生成最终视频。检查短片可加 `--max-frames 60`。默认 `compatible` 为 FP32；`--gpu-profile rtx5090d` 启用现有 FP16 加速路径，名称不限制显卡型号。默认批大小 4、帧批大小 1；显存较小时调小 `--batch-size`。共享拟合默认 `--shared-steps 100 --shared-batch-size 32`。

默认视频每格宽 960，保持原图纵横比。`--video-codec auto` 优先使用 NVENC，初始化失败时使用 mp4v。所有参数可通过 `--help` 查看。

只保留共享模型流程。模式选择、旧融合与时序回退、二维回退、去畸变实验、诊断/对比视频、Nokov、SLAM/VIO、实时采集及训练导出入口均已移除。

## 保存结果

只保存 A+B+C+D+E+G+H，字段见 [输出说明](docs/outputs.md)：

```text
OUTPUT/
├── final_shared_4view.mp4        # A：最终四目视频
├── joints3d.jsonl               # B：每帧三维骨架
├── hand_model.pt                # C+D：网格、拓扑、拟合后模型参数和变换
├── observations_2d/camera*.jsonl # E：各路检测和网络二维假设
├── projections_2d.jsonl         # E：共享模型回投各路的二维关节点
├── calibration/camera*.json     # G：内外参及畸变参数
├── timestamps/camera*.csv       # G：各路源帧时间戳
├── multiview_frames.csv         # G：四路同步映射
├── manifest.json                # G+H：来源、同步与标定信息
├── source_meta.json             # H：源录制元数据（有则保存）
├── run_config.json              # H：参数及输入/模型/代码身份
├── video.json                   # H：帧数、分辨率、编码和耗时
└── summary.json                 # H：完成标识、计数和产物 SHA256
```

中间文件位于 `.cache/`，导出与完整性校验成功后清理。失败时保留缓存，原参数重跑可复用已完成阶段；未完成的网络推理/拟合阶段会重跑。成功后同参数重跑只校验最终文件。输入、模型、代码或参数变化需要新输出目录。

不单独保存质量误差、拒绝原因或诊断报告；质量门限仍用于内部筛选。摘要保留基本有效帧和手数量。

## 验证

```bash
PYTHONPATH=scripts conda run -n ego-hand python -m pytest -q tests
```

测试覆盖 MP4/H264 MCAP 同步与标定、共享几何和鱼眼投影、MANO 参数重建、边界裁切、输出类别及完整性。MANO 测试需要已安装的资产及 WiLoR 依赖。

## 当前参考录制

以以下两个实际目录作为输入结构参考：

- MP4：`/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO-4B_AZLCA610091_20261111_191852`，根目录保存标定/元数据，`part0001/` 保存四路 MP4 和 PTS。
- MCAP：同一 recordings 目录下的 `EGO-4B_AZLCA610091_20261113_151426`，根目录保存 `params_cam_1920x1200.yaml`/元数据，`part0001/` 保存单个 MCAP。

这两个目录均可直接传给 `--input`，无需手动指定标定或移动文件。
