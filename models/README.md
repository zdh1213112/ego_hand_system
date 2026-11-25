# 模型来源与目录

四目共享推理使用 WiLoR、手部检测器和 MANO_RIGHT。完整双目/实时等流程还可能需要其他资产，以下按运行时目录列出。

| 本地文件 | 来源 | 用途 |
| --- | --- | --- |
| `wilor/wilor_final.ckpt` | [下载](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt) | WiLoR 基础权重 |
| `wilor/model_config.yaml` | [下载](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/model_config.yaml) | 与基础权重配套的模型配置 |
| `wilor/detector.pt` | [下载](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) | 公开手部检测器；也可通过 `--detector` 使用自己的检测器 |
| `mano/MANO_RIGHT.pkl` | [MANO 项目页](https://mano.is.tue.mpg.de/) | 新共享推理以右手规范模型重建双手 |
| `mano/MANO_LEFT.pkl` | [MANO 项目页](https://mano.is.tue.mpg.de/) | 完整资产安装、依赖检查及原有流程的模型准备 |
| `hand_landmarker.task` | [MediaPipe 模型文件](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) | 双目 MediaPipe 路线 |

上表中的公开 WiLoR 权重保存在 `models/wilor/`，也可从上游下载。[公开模型目录](https://huggingface.co/spaces/rolpotamias/WiLoR/tree/main/pretrained_models)

使用项目的 v4 手套模型时，将下列项目权重单独部署到 `models/wilor/`。两条路线都使用仓库自带的 `training/pretrained_models/model_config.yaml`。[训练命令](../training/README.md#wilor-训练) · [推理命令](../inference/README.md#wilor-手套权重推理)

| 文件 | 用途 |
| --- | --- |
| `coarse_start.ckpt` | 两条路线共用的 WiLoR 基础权重 |
| `2.0_glove_wilor_3680.pt` | 两条路线共用的原图手部检测器 |
| `v4_step48300.pt` | 路线 A 的已训 adapter，也是路线 B 的训练起点 |
| `v4_bbox_pinhole_15000.pt` | 路线 B 的已训 adapter，对应去畸变训练输出的 `glove_v4_bbox_pinhole_15000/best_trainable.pt` |

MANO 需要在项目网站注册并按其许可获取，安装方法见 [mano/README.md](mano/README.md)。本项目手部流程使用 MANO 手模型，无需 SMPL-H 全身资产。

MediaPipe 模型可以通过以下命令安装并校验：

```bash
bash scripts/install_mediapipe_model.sh
```

准备完整环境后，从仓库根目录检查子模块版本和资产：

```bash
python scripts/check_third_party.py --require-mano --require-wilor
```

固定版本、已验收资产的校验和与其他运行时信息见 [third_party/manifest.json](../third_party/manifest.json)。
