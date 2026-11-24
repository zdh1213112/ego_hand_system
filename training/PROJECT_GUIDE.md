# WiLoR 项目智能体快速导览

> 用途：供后续开发智能体在进入仓库后快速建立全局认知，定位入口、核心模块、数据流、配置、运行资产与本地扩展。
>
> 分析基线：2026-08-28。分析时工作区已有用户修改和未跟踪运行资产，本文按“当前工作区实际状态”描述，不假设仓库是干净的。

当前手套训练框架、使用命令、目录职责和扩展规范请优先查看 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)。手套数据协议及训练参数详见 [`GLOVE_TRAINING.md`](GLOVE_TRAINING.md)。

## 1. 项目一句话概述

WiLoR 是一个从单张 RGB 图像或逐帧视频中完成多手检测、左右手判定、MANO 参数回归、3D 手部网格/关节重建和可视化的两阶段系统：

1. Ultralytics YOLO 检测器定位每只手并给出左右手类别。
2. ViT-Large 重建器在手部裁剪上先粗略回归 MANO 与相机参数，再通过多尺度、顶点对齐的特征采样模块细化结果。

本地手套训练支持 OpenTouch Stage 1、相机对齐的 Stage 2，以及联合训练
ViT LoRA、coarse heads 和 RefineNet 的 `joint_lora_finetune`；还保留物理相机及
marker 微调 recipe。它们共用 `train_glove.py` 和 `GenericTrainer`，详见
`GLOVE_TRAINING.md`。本导览其余内容为历史结构快照；当前配置、辅助工具和
实验文档见 [文档索引](docs/README.md)。

对应论文为仓库根目录的 `2409.12259v2.pdf`，论文方法主体见第 4 节，训练细节见补充材料第 7 节。

## 2. 最重要的运行链路

```mermaid
flowchart LR
    A[RGB 图像或视频帧] --> B[YOLO detector.pt]
    B --> C[手框 + 置信度 + 左右手]
    C --> D[ViTDetDataset 裁剪、左手镜像、归一化]
    D --> E[ViT-Large 粗回归 MANO/相机]
    E --> F[RefineNet 多尺度顶点对齐特征细化]
    F --> G[MANO 778 顶点 + 21 关节]
    G --> H[全图相机换算与 2D 投影]
    H --> I[图片/视频叠加、JSONL、OBJ]
```

主调用关系如下：

```text
demo.py / video_demo.py / gradio_demo.py
  -> wilor.models.load_wilor()
     -> wilor.configs.get_config()
     -> WiLoR.load_from_checkpoint()
  -> YOLO(pretrained_models/detector.pt)
  -> ViTDetDataset
  -> WiLoR.forward()
     -> ViT backbone
     -> MANO coarse mesh
     -> RefineNet
     -> MANO final mesh/joints
  -> cam_crop_to_full()
  -> Renderer / OpenCV drawing / JSONL / OBJ
```

## 3. 仓库目录总览

以下覆盖业务目录和当前工作区本地目录，不展开 `.git/` 内部结构。

```text
WiLoR/
├── assets/                  # README 展示素材
├── demo_img/                # 官方图片 Demo 输入
├── mano_data/               # MANO 均值参数与本地 MANO 官方模型
├── output/                  # 运行时生成的推理结果，默认被 Git 忽略
├── pretrained_models/       # 检测器、重建器权重及模型/数据配置
├── pre_opentouch_dataset/   # OpenTouch HDF5 解码工具
├── configs/glove/           # 两阶段 schema-v3 训练配置
├── whim/                    # WHIM 数据集说明和 YouTube 视频清单
└── wilor/                   # 核心 Python 包
    ├── configs/             # YACS 配置默认值和加载器
    ├── datasets/            # 推理/训练数据集、OpenTouch manifest 与变换
    ├── models/              # WiLoR、MANO、损失、判别器
    │   ├── backbones/       # ViT 主干及实验性 FastViT 分支
    │   └── heads/           # 多尺度细化头
    ├── training/            # GenericTrainer、recipe、loss 与评估
    └── utils/               # 几何、渲染、评估、日志与掌心坐标工具
```

## 4. 每个顶层文件夹的作用

### `assets/`

- 存放 README 使用的静态图片。
- `teaser.png` 是项目概览图，不参与推理或训练计算。

### `demo_img/`

- 官方提供的 8 张示例图片。
- 被 `demo.py --img_folder demo_img` 和 Gradio 示例列表直接使用。
- 适合做安装后的最小图片推理检查。

### `mano_data/`

- `mano_mean_params.npz`：上游跟踪文件，保存平均相机、手姿态和形状参数，用于初始化 ViT 中的 camera/pose/shape token。
- `mano_v1_2/models/MANO_RIGHT.pkl`：WiLoR 右手规范空间推理和训练必需模型。
- `mano_v1_2/models/MANO_LEFT.pkl`：原生左右手数据的 MANO 重放校验必需模型；
  第一阶段不会把其参数直接作为 WiLoR 参数监督。
- 原始压缩包、SMPL+H 和旧版 Chumpy 示例不参与当前 WiLoR 运行，已从本地工作区清理。
- `wilor.models.load_wilor()` 的当前本地修改可接受 MANO 模型文件、`models/` 目录或整个解压目录，并自动查找 `MANO_RIGHT.pkl`。
- MANO 有独立许可，不能把模型文件视为普通开源代码随意再分发。

### `output/`

- 这是运行时生成目录，已被 `.gitignore` 排除，不属于核心源码；清理后可以不存在。
- 常见文件：
  - `*_wilor.mp4`：检测框、21 点骨架和可选 MANO 网格的标注视频。
  - `*_wilor.jsonl`：每帧一行的结构化预测。
  - `preview.jpg`：本地快速检查预览图。
- 后续智能体应把这里视为可再生成的结果目录，而不是修改源代码的位置。

### `pretrained_models/`

- `detector.pt`：Ultralytics YOLO 手部检测与左右手分类权重，当前约 52 MB。
- `wilor_final.ckpt`：PyTorch Lightning 格式的 WiLoR 重建器权重，当前约 2.4 GB。
- `model_config.yaml`：重建器结构、输入归一化、MANO、数据集权重、损失权重和优化超参数。
- `dataset_config.yaml`：论文使用的数据集 WebDataset tar 分片位置和样本规模记录；路径指向外部 `wilor_training_data/`，不在本仓库中。
- 权重是本地下载的大文件并由 `.gitignore` 排除；配置 YAML 是上游跟踪文件。
- `load_wilor()` 会删除配置中的 `MODEL.BACKBONE.PRETRAINED_WEIGHTS`，避免先加载单独 ViTPose 权重，再从完整 checkpoint 恢复模型。

### `whim/`

- `Dataset_instructions.md`：WHIM 标注下载地址和通过 YouTube 恢复图像的说明。
- `train_video_ids.json`：1431 个训练视频条目，记录视频 ID、候选分辨率、长度和 FPS。
- `test_video_ids.json`：26 个测试视频条目。
- 此处只有下载说明和视频元数据，不包含论文所述 2M 图像或完整标注。
- 根目录 `download_videos.py` 使用这些 JSON 清单下载视频并抽取标注帧。

### `wilor/`

- 项目的核心 Python 包，模型、数据预处理和可视化都在这里。
- 没有独立安装清单或 `pyproject.toml`；通常从仓库根目录直接执行脚本，使根目录自然进入 Python import path。
- 详细子目录见下一节。

### 各处的 `__pycache__/`

- Python 自动生成的字节码缓存，与源码逻辑无关。
- 仓库历史上意外跟踪了部分 `.pyc`；新生成的缓存已由 `.gitignore` 排除。
- 分析或修改逻辑时应读取对应 `.py`，不要把 `.pyc` 当成源文件，也不要为了功能改动手工编辑它们。

## 5. `wilor/` 核心包详解

### `wilor/configs/`

主要文件：`__init__.py`

- 定义 YACS 默认配置，包括通用运行、训练批量、增强、模型输入尺寸和焦距。
- `default_config()` 返回默认配置副本。
- `get_config(config_file, merge=True, update_cachedir=False)` 把 YAML 合并进默认配置并冻结。
- `dataset_config()` 和 `dataset_eval_config()` 分别默认寻找 `datasets_tar.yaml`、`datasets_eval.yaml`；这两个文件当前不在仓库中。
- 推理的实际配置入口是 `pretrained_models/model_config.yaml`。

### `wilor/datasets/`

#### `vitdet_dataset.py`

- 定义 `ViTDetDataset`，是图片、Gradio 和视频推理共同使用的检测后裁剪数据集。
- 输入为一张 OpenCV BGR 图像、N 个检测框和 N 个左右手类别。
- 根据框中心和大小扩框，统一为模型需要的比例，再生成 `256 x 256` patch。
- 左手会水平翻转为右手规范空间，模型输出后由调用脚本再把 x 轴翻回。
- 对大幅下采样的手框先做高斯模糊，减少 aliasing。
- 使用 ImageNet mean/std 归一化；快速模式可直接返回 FP16 tensor。
- 返回的关键字段：`img`、`personid`、`box_center`、`box_size`、`img_size`、`right`。
- 文件显式断言 `train == False`，因此该类的职责是推理输入封装。

#### `utils.py`

- 提供通用图像裁剪、仿射变换、宽高比扩展和 HWC→CHW 转换。
- 包含训练数据常见的随机缩放、旋转、平移、颜色抖动、翻转和 extreme crop 逻辑。
- 能同步变换 2D/3D 关键点和 MANO axis-angle 参数。
- `get_example()` 是带标注样本的综合预处理函数。
- 文件后半部分还保留了从人体关键点构造不同局部裁剪的通用代码，部分明显继承自人体网格项目，不是当前手部推理主链路。

### `wilor/models/`

#### `__init__.py`

- 导出 `MANO`、`WiLoR`、`Discriminator`。
- `load_wilor()` 是三个推理脚本共同使用的模型加载入口。
- 加载时补充 `MODEL.BBOX_SHAPE=[192,256]`，移除单独 backbone 预训练权重路径，解析 MANO 路径，并以 `strict=False` 加载 Lightning checkpoint。

#### `wilor.py`

- 定义顶层 `WiLoR(pl.LightningModule)`。
- 组装 ViT backbone、`RefineNet`、MANO layer、损失函数、可选判别器和可视化器。
- `forward_step()` 是核心前向过程：
  1. 从 `256 x 256` patch 左右各裁掉 32 像素，得到 `256 x 192` ViT 输入。
  2. backbone 输出粗 MANO 参数、弱透视相机、参数特征和 `16 x 12` 图像特征。
  3. 用粗 MANO 参数生成初始网格。
  4. `RefineNet` 预测 pose/shape/camera residual。
  5. 最终 MANO 前向得到 778 顶点和 21 关节。
  6. 计算 crop 空间相机平移和 2D 投影。
- `forward_coarse_step()` 复用同一个 coarse 实现但不调用 2D 投影或 RefineNet，
  供 OpenTouch Stage 1 使用；正常推理接口和完整 forward 保持不变。
- 还包含损失计算、Lightning optimizer/training/validation hooks 和 TensorBoard 可视化逻辑。

#### `mano_wrapper.py`

- 扩展 `smplx.MANOLayer`。
- 从 MANO 网格顶点补充指尖关节，并重排为 21 点 OpenPose 风格顺序。
- 输出关节顺序为：腕点，然后依次为拇指、食指、中指、无名指、小指，每根手指 4 点。

#### `losses.py`

- `Keypoint2DLoss`：带置信度掩码的 L1/L2 2D 关键点损失。
- `Keypoint3DLoss`：先以腕点为根节点中心化，再计算带置信度的 L1/L2 损失。
- `ParameterLoss`：对有效 MANO 参数计算 MSE。

#### `discriminator.py`

- HMR 风格的姿态与形状判别器。
- 分别判别每个关节旋转、整体关节组合和 10 维 betas，用于约束合理手姿与形状。

### `wilor/models/backbones/`

#### `__init__.py`

- `create_backbone(cfg)` 根据 `MODEL.BACKBONE.TYPE` 创建模型。
- 默认使用 `vit`。
- 还保留实验性 `fast_vit` 分支，期望 `pretrained_models/fastvit_ma36.pt`；当前主配置和权重不使用该分支。

#### `vit.py`

- 自定义 ViT-Large：patch size 16、hidden dim 1280、32 个 block、16 个 attention head。
- 输入尺寸为 `256 x 192`，产生 `16 x 12` 图像 token 网格。
- 在图像 token 前加入 16 个姿态 token、1 个形状 token和 1 个相机 token。
- 通过线性层粗回归：
  - 16 组 6D rotation，共 96 维；
  - 10 维 MANO betas；
  - 3 维弱透视相机参数。
- `skip_blocks=True` 时跳过预设的 5 个后段 Transformer block，是 `--fast` 层裁剪加速的一部分。

### `wilor/models/heads/`

主要文件：`refinement_net.py`

- `DeConvNet` 将 `16 x 12` ViT 特征生成高、中、低三个尺度的特征图。
- `RefineNet` 把粗 MANO 的全部 778 个顶点投影到各尺度特征图。
- 使用 `grid_sample` 获取每个顶点的图像对齐特征，并在顶点维执行 max aggregation。
- 拼接多尺度特征后，分别预测 96 维 pose residual、10 维 shape residual 和 3 维 camera residual。
- 这是论文 Figure 4 紫色 refinement 模块的主要代码实现。

### `wilor/utils/`

#### `geometry.py`

- axis-angle、quaternion、6D rotation 到旋转矩阵的转换。
- 批量透视投影函数。

#### `renderer.py`

- 当前图片和视频 Demo 的主要 MANO 渲染器。
- `cam_crop_to_full()` 把 crop 空间弱透视相机换算成全图 3D 平移。
- `Renderer` 使用 Pyrender/Trimesh 完成单手、多手 RGBA 渲染和 OBJ mesh 构造。
- 会为手腕开口补面，并为左手反转三角面绕序。
- 默认使用 EGL，适合无显示器环境，但仍要求正确的 OpenGL/EGL 运行环境。

#### `mesh_renderer.py`

- 较早的单 mesh 渲染与 TensorBoard 网格可视化实现。
- `WiLoR.tensorboard_logging()` 使用其中的 `visualize_tensorboard()`。

#### `skeleton_renderer.py`

- 将预测/真值 3D 关节投影为 2D，并组织骨架可视化面板。
- 更偏向模型调试与评估，不是 `video_demo.py` 的当前 OpenCV 关键点叠加实现。

#### `render_openpose.py`

- 从 OpenPose C++ 逻辑移植的 2D 关键点、骨骼连线绘制代码。
- 同时保留手部和人体关键点配色/拓扑。

#### `pose_utils.py`

- Procrustes 对齐、MPJPE/重建误差和 PCK 评估辅助类。
- 包含针对 FreiHAND/HO3D 关节顺序和结果导出的处理。

#### `misc.py`、`rich_utils.py`、`pylogger.py`

- 来自 Hydra/PyTorch Lightning 工程模板的任务包装、日志、配置树、回调和 logger 实例化工具。
- 当前图片/视频推理入口基本不依赖这些高级运行框架。

#### `__init__.py`

- 汇总导出 Renderer、MeshRenderer、SkeletonRenderer、评估器。
- `recursive_to()` 递归把嵌套 batch 中的 tensor 移到目标设备。

## 6. 根目录关键文件

| 文件 | 作用 | 备注 |
|---|---|---|
| `README.md` | 安装、权重下载、图片/视频/Gradio 使用说明 | 当前已有本地视频推理说明修改 |
| `2409.12259v2.pdf` | WiLoR arXiv v2 论文 | 当前为本地未跟踪文件 |
| `demo.py` | 文件夹图片批量推理、叠加渲染、可选 OBJ 导出 | 官方主 CLI；支持 `--fast` |
| `video_demo.py` | 单视频或目录递归视频推理 | 当前本地扩展；输出 MP4 + JSONL，可选 mesh/vertices |
| `gradio_demo.py` | 本地 Gradio 图片交互界面 | import 时立即加载权重并在文件末尾启动服务 |
| `download_videos.py` | 按 WHIM 视频清单下载 YouTube 视频并抽帧 | 数据恢复辅助脚本，不是推理入口 |
| `requirements.txt` | Python 依赖列表 | PyTorch/torchvision 按 README 单独安装；当前本地增加 `dill` |
| `license.txt` | WiLoR 模型/代码仓库声明的 CC BY-NC-ND 4.0 文本 | 外部依赖和 MANO 另有许可 |

## 7. 推理数据契约

### 检测器约定

- `det.boxes.xyxy`：全图像素坐标的 `[x1, y1, x2, y2]`。
- `det.boxes.conf`：检测置信度。
- `det.boxes.cls`：`0 = left`，`1 = right`。

### `WiLoR.forward(batch)` 主要输出

| 键 | 典型形状 | 含义 |
|---|---:|---|
| `pred_cam` | `[B, 3]` | crop 空间弱透视相机 `[scale, tx, ty]` |
| `pred_cam_t` | `[B, 3]` | crop 相机平移 |
| `focal_length` | `[B, 2]` | 模型 crop 使用的焦距 |
| `pred_mano_params.global_orient` | `[B, 1, 3, 3]` | 全局旋转矩阵 |
| `pred_mano_params.hand_pose` | `[B, 15, 3, 3]` | 15 个 MANO 关节旋转矩阵 |
| `pred_mano_params.betas` | `[B, 10]` | MANO 形状参数 |
| `pred_vertices` | `[B, 778, 3]` | MANO 网格顶点 |
| `pred_keypoints_3d` | `[B, 21, 3]` | 21 个 3D 手部关节 |
| `pred_keypoints_2d` | `[B, 21, 2]` | 模型 crop 空间投影 |

### 左右手坐标处理

1. `ViTDetDataset` 将左手 crop 水平镜像为右手规范空间。
2. 模型只依赖右手 MANO 模型进行统一回归。
3. Demo 在输出阶段按 `2 * right - 1` 把左手顶点、关节和相机 x 分量翻回。
4. 若新增导出格式或后处理，必须保持这套镜像逻辑一致，否则左右手会反向或 mesh winding 错误。

### `video_demo.py` JSONL 结构

每一行对应一个处理过的视频帧，核心字段：

```text
frame_index
processed_frame_index
source_time_seconds
num_hands
hands[]
  detection_index
  bbox_xyxy
  confidence
  handedness / is_right
  camera_translation
  joints_3d
  joints_2d
  mano.global_orient / hand_pose / betas
  vertices              # 仅 --save-vertices 时存在
```

## 8. 三个主要推理入口如何选择

### 图片批处理：`demo.py`

```bash
python demo.py \
  --img_folder demo_img \
  --out_folder demo_out \
  --save_mesh
```

- 面向一个图片目录。
- 保存 mesh 叠加 JPG；可选每只手的 OBJ。
- `--fast` 使用 FP16、编译 backbone 和跳层。

### 视频批处理：`video_demo.py`

```bash
python video_demo.py \
  --input /path/to/video-or-directory \
  --output output/video_results \
  --mano-path mano_data/mano_v1_2
```

- 递归查找 `.mkv/.mp4/.avi/.mov/.webm/.m4v`。
- 每个视频输出 JSONL；默认输出检测框和 21 点骨架视频。
- `--render-mesh` 增加 mesh 叠加；`--save-meshes` 导出 OBJ；`--save-vertices` 把 778 顶点写入 JSONL。
- `--max-frames`、`--frame-stride` 适合快速检查和抽帧。
- 默认输入和 MANO 路径带有当前机器的绝对路径，跨机器运行时应显式传参。

### 交互图片：`gradio_demo.py`

```bash
python gradio_demo.py
```

- 单张图上传与可视化。
- 模型、检测器和 Renderer 在模块 import 阶段创建，启动和重载成本高。
- 更适合演示，不适合作为可复用服务端库直接 import。

## 9. 常见改动应去哪里

| 目标 | 优先查看/修改 |
|---|---|
| 调整图片检测阈值或批量流程 | `demo.py` |
| 调整视频递归、抽帧、JSONL、绘制 | `video_demo.py` |
| 调整 Gradio UI | `gradio_demo.py` |
| 修改手框扩张、裁剪、左手镜像 | `wilor/datasets/vitdet_dataset.py` |
| 修改数据增强或带标注预处理 | `wilor/datasets/utils.py` |
| 修改主模型前向和输出 | `wilor/models/wilor.py` |
| 修改 ViT token、层数或快速跳层 | `wilor/models/backbones/vit.py` |
| 修改多尺度细化策略 | `wilor/models/heads/refinement_net.py` |
| 修改 21 关节顺序或 MANO 输出 | `wilor/models/mano_wrapper.py` |
| 修改相机从 crop 到全图的转换 | `wilor/utils/renderer.py::cam_crop_to_full` |
| 修改 mesh 外观、光照、OBJ | `wilor/utils/renderer.py` |
| 修改损失或判别器 | `wilor/models/losses.py`、`discriminator.py` |
| 修改结构和超参数 | `pretrained_models/model_config.yaml` |
| 修改 WHIM 下载/抽帧 | `download_videos.py`、`whim/` |

## 10. 当前工作区的重要注意事项

1. 工作区不是干净状态。已有暂存内容和 MANO 路径加载逻辑修改；后续智能体必须先运行 `git status --short`，不要覆盖用户修改。
2. `.gitignore` 已排除本地权重、MANO 模型、输入、输出、编辑器配置和 Python 缓存；其中部分资产仍是运行必需文件，只是不应提交。
3. 当前目录约 2.6 GB，主要体积来自约 2.4 GB 的重建器 checkpoint；代码检索时应优先限定 `*.py/*.yaml/*.md`。
4. `gradio_demo.py` 第 23 行的设备选择条件当前是反向的：有 CUDA 时选择 CPU，无 CUDA 时选择 CUDA。若运行 Gradio，应先核对并修正此处。
5. `download_videos.py` 使用了 `cv2`，但当前文件没有导入 `cv2`；README 示例也没有传入脚本要求的 `--root`。该辅助脚本运行前需要检查。
6. `video_demo.py` 的默认绝对路径绑定当前机器，自动化或迁移时始终显式传 `--input`、`--output`、`--mano-path`。
7. `--fast` 会改变数值精度并跳过部分 Transformer block，适合吞吐优先场景；需要精确复现默认模型结果时不要使用。
8. Pyrender 默认走 EGL。无头机器需要可用的 EGL/OpenGL 驱动；若只需要 JSONL，可用 `video_demo.py --no-video --no-render` 并避免 mesh renderer。
9. `load_wilor(..., strict=False)` 会容忍 checkpoint 键不完全匹配。修改模型结构后应显式检查 missing/unexpected keys，避免静默漏载参数。
10. 当前系统 Python 环境未安装 PyTorch、Lightning、Ultralytics、SMPL-X、Pyrender、YACS 等核心依赖；实际运行应进入项目原有 Conda/虚拟环境或按 README 重建环境。

## 11. 建议的智能体阅读顺序

如果任务是推理或输出改造：

1. `README.md`
2. `video_demo.py` 或 `demo.py`
3. `wilor/datasets/vitdet_dataset.py`
4. `wilor/models/__init__.py`
5. `wilor/models/wilor.py`
6. `wilor/utils/renderer.py`

如果任务是模型结构理解：

1. 论文 Figure 4 和补充材料 7.2
2. `pretrained_models/model_config.yaml`
3. `wilor/models/wilor.py`
4. `wilor/models/backbones/vit.py`
5. `wilor/models/heads/refinement_net.py`
6. `wilor/models/mano_wrapper.py`

如果任务是数据或标注处理：

1. `wilor/datasets/utils.py`
2. `pretrained_models/dataset_config.yaml`
3. `whim/Dataset_instructions.md`
4. `download_videos.py`

## 12. 论文与代码的关键对应

| 论文概念 | 代码位置 |
|---|---|
| 实时手检测、左右手与框 | 外部 Ultralytics 运行时 + `pretrained_models/detector.pt` |
| 检测后紧致手部 crop | `wilor/datasets/vitdet_dataset.py` |
| ViT-Large、pose/shape/camera tokens | `wilor/models/backbones/vit.py` |
| 粗 MANO 解码 | `vit.py::ViT.forward_features()` |
| 多尺度反卷积特征 | `wilor/models/heads/refinement_net.py::DeConvNet` |
| 全网格顶点投影与特征采样 | `RefineNet.forward()` |
| pose/shape/camera residual | `RefineNet.dec_pose/dec_shape/dec_cam` |
| MANO 网格与 21 关节 | `wilor/models/mano_wrapper.py` |
| 2D/3D/MANO/对抗损失 | `wilor/models/losses.py`、`wilor/models/discriminator.py`、`wilor/models/wilor.py` |
| 图像与视频全图渲染 | `wilor/utils/renderer.py`、根目录 Demo 脚本 |

这份导览应作为入口索引使用；涉及具体行为时，以当前 `.py` 源码、YAML 配置和 `git diff` 为最终事实来源。
