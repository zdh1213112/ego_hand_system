# Orbbec EGO4 共享 MANO 离线推理

本入口来自 `feature/ego-hand-inference` 的 `7db7fbf`，集成基线为
`feature/orbbec_ego_4` 的 `9c50e98`。每个同步时刻、每只手只拟合一份三维骨架和
MANO 网格，然后用各路原始标定投影到四路鱼眼画面。无有效模型的时刻保留空帧。

## 安装与输入

沿用根目录 README 的环境安装和模型准备步骤。新入口使用相同的 `ego-hand` 环境、
`third_party/WiLoR`、WiLoR 基础权重和 `models/mano/MANO_RIGHT.pkl`。

输入可以是录制根目录、`part0001` 分段目录，或含四路 H264 的 Orbbec MCAP。
MP4 必须同时具有四路视频和对应的 PTS CSV；多分段录制须分别指定分段和输出目录。
标定优先从录制元数据读取，也识别配套 `*_calibration_camera.yaml`、
`*_params_cam_*.yaml`。可通过 `--calibration` 指定 YAML 或四路 JSON 标定目录。

| 相机 | MP4 流名 | MCAP topic | 标定 ID |
| --- | --- | --- | --- |
| camera0 | left | /camera/left | cam_0 |
| camera1 | right | /camera/right | cam_1 |
| camera2 | right_side | /camera/right_side | cam_2 |
| camera3 | left_side | /camera/left_side | cam_3 |

## 运行

```bash
bash scripts/run_hand_inference.sh \
  --input /path/to/recording/part0001 \
  --detector /path/to/hand_detector.pt \
  --output output/shared_hand
```

MCAP 使用同一命令，将 `--input` 指向 MCAP 文件或单分段目录。
`--session` 和 `--mcap` 也可分别显式指定输入类型。完整参数见 `--help`。
默认 Conda 环境是 `ego-hand`；可设置 `--conda-env NAME`，或使用
`EGO_HAND_PYTHON=/path/to/python` 直接选择解释器。

默认处理完整录制，使用 FP32、检测批大小 4、共享拟合 100 步、拟合批大小 32。
`--max-frames 60` 可检查短片，`--gpu-profile rtx5090d` 可启用既有 FP16 路径。
默认同步容差 1500 微秒，每路帧不复用，只处理四路均匹配的时刻。

## 输出与重跑

输出包括 `final_shared_4view.mp4`、`joints3d.jsonl`、`hand_model.pt`、二维观测、
共享模型回投、标定、源时间戳、同步表和运行配置，字段见 [输出说明](outputs.md)。
时间戳单位是纳秒，三维单位是米，坐标系为标定参考系。
这些字段与实时链路的接口分别定义；下游应显式处理时间单位和坐标变换。

中间文件保存在输出的 `.cache/` 中，完整性校验成功后清理。失败时保留缓存，
同参数重跑可恢复已完成的阶段；成功后同参数重跑仅验证交付文件。
输入、模型、代码或参数变化时使用新输出目录。

## 与既有功能共存

新链路代码位于 `scripts/ego_hand_inference/`。相机模型、标定读取和推理模块通过
明确的包路径加载，避免与原有 `camera_models`、`ego_data` 等同名模块混用。
原有 `run_multiview_wilor_experiment.sh`、实时、VIO、NOKOV、标签导出和训练评估入口保持其原有调用方式。
两条链路共用已固定版本的 WiLoR 子模块和本地模型资产。
相同的时间戳配对、顺序视频读取、MANO 规范常量和 NumPy/chumpy 兼容辅助直接复用既有代码。

## 验证

```bash
PYTHONPATH=scripts conda run -n ego-hand python -m pytest -q tests
```

完整测试需要根目录 README 中的运行环境；MANO 几何测试还需要模型资产。
测试同时覆盖既有功能和新增的同步、MCAP、共享几何、模型重建、渲染、导出及失败恢复。
