# 四机手部推理

从仓库根目录运行 [run.sh](../inference/run.sh)。WiLoR 四机重建使用 [ego_hand_inference/](../inference/ego_hand_inference/)，每个同步时刻、每只手输出一份三维骨架和 MANO 网格，再按相机标定回投。FAH 和 GEM-X 使用各自的后端和参数。[FAH 使用说明](../docs/multiview_fah_experiment.md) · [GEM-X 使用说明](../docs/multiview_gem_experiment.md)

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

## WiLoR 手套权重推理

从仓库根目录选择一条命令。两条路线都用 `2.0_glove_wilor_3680.pt` 在原图上检测手框，`--adapter` 决定手框的处理方式。2D 关节使用原图像素坐标，四机视频将骨架画回原图。[模型准备](../models/README.md)

### 路线 A · 原图手框（现用 v4）

直接裁剪原图手框。下面使用现用的第 48,300 步 adapter。

```bash
bash inference/run.sh wilor \
  --input /path/to/recording/part0001 --output output/shared_v4 \
  --detector models/wilor/2.0_glove_wilor_3680.pt \
  --checkpoint models/wilor/coarse_start.ckpt \
  --model-config training/pretrained_models/model_config.yaml \
  --mano-model-dir models/mano --adapter models/wilor/v4_step48300.pt
```

### 路线 B · 原图手框去畸变

在原图上检测后，按 KB4 标定将手框转为透视图，预测关节和 MANO 姿态，再映回原相机。下面使用去畸变训练得到的第 15,000 步 adapter。

```bash
bash inference/run.sh wilor \
  --input /path/to/recording/part0001 --output output/shared_v4_bbox_pinhole \
  --detector models/wilor/2.0_glove_wilor_3680.pt \
  --checkpoint models/wilor/coarse_start.ckpt \
  --model-config training/pretrained_models/model_config.yaml \
  --mano-model-dir models/mano --adapter models/wilor/v4_bbox_pinhole_15000.pt
```

自行训练时，可直接把上面命令的 `--adapter` 换成对应输出目录的 `best_trainable.pt`。例如路线 B 使用 `training/output/glove_v4_bbox_pinhole_15000/best_trainable.pt`。[两条训练路线](../training/README.md#wilor-训练)

### v5 · 原始四目录像与精度评估

以下命令从 **ego_hand_system 根目录**运行。将 v5 完整权重传给 `--checkpoint`，例如
`wilor_pinhole_best_261126.ckpt` 或 `best_mpjpe_full.pt`。程序从导出文件恢复
LoRA 和 `physical_perspective` 相机头，使用本仓库 `training` 的模型与裁剪代码。
每个原始 KB 手框生成 256 × 256 局部针孔图。模型以 FP32 推理，再将 2D、3D 和 MANO 朝向变回原相机坐标，供跨视角关联和共享拟合使用。

```bash
bash inference/run.sh wilor --input /path/to/recording/part0001 \
  --checkpoint /path/to/wilor_pinhole_best_261126.ckpt --detector /path/to/detector.pt \
  --mano-model-dir /path/to/mano --output /path/to/v5_delivery
```

### WiLoR 通用选项

MCAP 使用同一命令，将 `--input` 指向 MCAP 文件或单分段目录。
`--session` 和 `--mcap` 也可分别显式指定输入类型。完整参数见 `--help`。
默认 Conda 环境是 `ego-hand`；可设置 `--conda-env NAME`，或使用
`EGO_HAND_PYTHON=/path/to/python` 直接选择解释器。

默认处理完整录制，使用 FP32、检测批大小 4、共享拟合 100 步、拟合批大小 32。
`--max-frames 60` 可检查短片，`--gpu-profile rtx5090d` 可启用既有 FP16 路径。
默认同步容差 1500 微秒，每路帧不复用，只处理四路均匹配的时刻。

## 原生 WiLoR 和其他后端

省略 `--adapter` 可运行原生 WiLoR，默认后端也是 WiLoR。

```bash
bash inference/run.sh --input /path/to/recording/part0001 \
  --detector /path/to/hand_detector.pt --output output/shared_hand
```

FAH 和 GEM-X 接受 `--session` 或 `--mcap`，完整选项见 `bash inference/run.sh fah --help` 或 `bash inference/run.sh gem-x --help`。[FAH 参数](../docs/multiview_fah_experiment.md) · [GEM-X 参数](../docs/multiview_gem_experiment.md)

## 输出与重跑

输出包括 `final_shared_4view.mp4`、`joints3d.jsonl`、`hand_model.pt`、二维观测、
共享模型回投、标定、源时间戳、同步表和运行配置，字段见 [输出说明](../docs/outputs.md)。
时间戳单位是纳秒，三维单位是米，坐标系为标定参考系。
这些字段与实时链路的接口分别定义；下游应显式处理时间单位和坐标变换。

中间文件保存在输出的 `.cache/` 中，完整性校验成功后清理。失败时保留缓存，
同参数重跑可恢复已完成的阶段；成功后同参数重跑仅验证交付文件。
输入、模型、代码或参数变化时使用新输出目录。

## 与既有功能共存

新链路代码位于 `inference/ego_hand_inference/`。相机模型、标定读取和推理模块通过
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
