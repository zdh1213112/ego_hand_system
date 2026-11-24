# 用 NOKOV/MANO 标签微调 FAH-S 和 GEM-X EgoHand

在 `wilor_training` 根目录运行 `train_hand.py`。FAH-S 和 GEM-X EgoHand 分别加载自己的发布权重，共用现有的 `train.jsonl` 和 `val.jsonl`。默认数据目录是 `output/group_001_20261116_193922/dataset`，也可用 `--dataset` 指向具有相同原图标签格式的目录。

## 在新机器上安装

使用 Linux x86_64、Python 3.11 和支持 CUDA 的 NVIDIA GPU。克隆本项目后，在项目根目录为两个模型分别创建环境。安装清单从官方 GitHub 的固定提交构建 Python 包。FAH 使用 PyTorch 2.6，GEM-X 使用 PyTorch 2.10。若集群预装 CUDA 版 PyTorch，可以在对应环境中沿用匹配版本。

```bash
python3.11 -m venv .venvs/fah
.venvs/fah/bin/python -m pip install -r requirements/hand-fah.txt
python3.11 -m venv .venvs/gem-x
.venvs/gem-x/bin/python -m pip install -r requirements/hand-gem-x.txt
```

在有网络的机器上下载发布权重和手部几何资产。已有相同文件也可按下表的目录放置。准备完成后训练可离线运行。

```bash
.venvs/fah/bin/python -m tools.hand_finetune.assets fah
.venvs/gem-x/bin/python -m tools.hand_finetune.assets gem-x
```

| 资产 | 项目内位置 |
| --- | --- |
| FAH-S 权重 | `pretrained_models/hand/weights/fah-s/model.pt` |
| EgoHand 权重 | `pretrained_models/hand/weights/egohand_s060000.ckpt` |
| SOMA v0.3.3 手部资产 | `pretrained_models/hand/assets/soma/`，保留其中的 `MHR/` 子目录 |

将已导出的数据集放到新机器。训练需要 `train.jsonl`、`val.jsonl` 以及清单中引用的图片和 NPY 标签，文件相对路径应与清单一致。若导出目录使用软链接，跨机复制时需复制实际文件。Git 仅分发代码、依赖清单和说明。

## 运行

依次运行下列命令，每个模型会打乱训练集并遍历一轮，随后在全部验证样本上评估。每次运行使用一个新输出目录。

```bash
run_dir=output/group_001_20261116_193922/finetune
data_dir=output/group_001_20261116_193922/dataset
.venvs/fah/bin/python train_hand.py fah --dataset "$data_dir" --epochs 1 --output "$run_dir/fah_epoch1"
.venvs/gem-x/bin/python train_hand.py gem-x --dataset "$data_dir" --epochs 1 --output "$run_dir/gem_x_epoch1"
```

默认使用 CUDA、batch 4、AdamW、学习率 `1e-5` 和随机种子 42。加 `--max-steps 100 --val-samples 256` 可复现短程训练。验证子集按种子固定，两模型及微调前后均可复用。省略 `--val-samples` 会评估整份验证集。

| 模型 | 输出权重 | 更新哪些部分 |
| --- | --- | --- |
| FAH-S | `model.pt` | 点预测头的共享特征、2D 热图、相对三维位置和腕深。冻结学生编码器、相机编码器及无监督的分割和可见性分支。 |
| GEM-X EgoHand | `egohand.ckpt` | SOMA 姿态输出层、相机输出层及直接 2D 坐标输出层。冻结图像编码器、Transformer 和手型分支。 |

默认通过当前环境导入已安装的算法包。`--init` 可指定已有权重，`--soma-assets` 可指定已有资产目录。开发上游算法时，仍可显式使用 `--upstream` 指定源码检出目录，并沿用该目录的权重和资产布局。每轮结束会保存最新权重和 `metrics.json`。从 `--init` 开始下一轮微调时，优化器重新初始化。FAH 原版 Python 接口可通过 `finetune_ckpt` 加载导出权重。GEM-X 可用 `gem.pose.api.load_pose_checkpoint` 加载导出权重。

单独复评已经生成的权重可执行

```bash
.venvs/fah/bin/python train_hand.py fah --dataset "$data_dir" --evaluate \
  --init "$run_dir/fah_100steps/model.pt" --output "$run_dir/fah_reeval"
.venvs/gem-x/bin/python train_hand.py gem-x --dataset "$data_dir" --evaluate \
  --init "$run_dir/gem_x_100steps/egohand.ckpt" --output "$run_dir/gem_x_reeval"
```

## 标签如何进入模型

1. 每条样本取一张原图和一只手的标签。根据手框、相机内参及 KB 畸变系数，将相机朝向手框中心，重采样得到透视裁剪。裁剪的朝向和大小仅取决于框和标定。评估也使用标签中的手框。
2. 将 `joints_3d` 和 `vertices` 加上 `trans` 得到物理相机坐标，再旋转到裁剪相机，用裁剪内参重新投影。FAH 把左手翻到标准右手，GEM-X 把右手翻到标准左手，同时镜像图像和三维横坐标。
3. FAH 使用 320 像素裁剪，监督 21 个关节、40 个骨段插值点、发布权重指定的 400 个 MANO 顶点和腕部深度。GEM-X 使用 256 像素裁剪，将预测姿态通过原生 SOMA 手模型转换为关节，取 25 个点中与 MANO 对应的 21 个，监督腕部对齐 3D、直接 2D 和相机中的腕部位置。其余四个掌部关节没有直接标注。

## 本次训练与验证

数据来自 `group_001_20261116_193922`。现有时间划分包含 22,830 个训练手样本和 1,484 个验证手样本，按源录制和同步帧检查了集合互斥。每个模型在 RTX 4090 上独立训练 100 步，共使用 400 个训练手样本。下表由重新加载权重并评估完整验证集得到。

| 模型 | 原始 3D 误差 | 微调后 3D 误差 | 原始 2D 误差 | 微调后 2D 误差 |
| --- | ---: | ---: | ---: | ---: |
| FAH-S | 18.34 mm | 15.19 mm | 10.39 px | 8.29 px |
| GEM-X EgoHand | 18.74 mm | 10.76 mm | 8.97 px | 8.15 px |

3D 误差是各自减去腕部坐标后，21 个关节到拟合标签的平均欧氏距离。2D 误差按各模型自身的裁剪像素报告。验证来自同一录制的留出时段，尚待跨录制评估。导出摘要中的 `approved_for_training` 为 `false`，标签贴合质量仍待正式验收。
