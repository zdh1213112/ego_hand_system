# 在自己的 KB 手套录制上比较 FAH-S V0–V5

正式训练在指定训练服务器上进行。从该服务器的仓库根目录运行以下命令，准备真实录制的原图、标签、固定检测器和独立的输出目录。`merged` 仅用于理解数据格式。所有版本均显式接收实际数据目录。V0–V5 在加载模型前校验 FAH-S 发布文件的 SHA-256，须为 `f469c80356391c2993c58b4421ef3ee81c36549af932cffed1824a038147484c`。相同字节的副本可放在任意路径。已有微调权重可用无 `--version` 的 `train_hand.py fah --init` 继续微调，或用独立评估入口直接打分。[固定版本的发布文件](https://huggingface.co/noahcao/FAH/blob/9f8a86f47a83cb23226ae7ab696c0500dabf6c82/fah-s/model.pt)

| 版本 | 相对 V1 的实验条件 |
| --- | --- |
| V0 | 原样评估初始权重，优化器步数为 0 |
| V1 | 用标注框训练原有点预测头和腕深分支 |
| V2 | 以固定概率混入匹配的检测框，漏检样本可使用训练集实测的框偏移重新裁剪 |
| V3 | 增加绝对腕部位置约束，默认权重 1.0。用发布解码方式恢复腕部，再计算与物理腕部等距的裁剪空间误差 |
| V4 | 同时训练已有的视觉编码器 adapter，默认学习率为头部的 1/4 |
| V5 | 用 `--combine` 显式列出验证后选定的 V2、V3、V4 条件，重新从同一初始权重训练 |

## 1. 准备真实录制和验证口径

每个输入导出目录含 `index.jsonl` 和其中引用的原图及每只手的 NPY 标签。用真实录制 ID 标识来源，同一录制 ID 覆盖该录制的所有相机与双手。下面的 ID、路径和权重均需替换为实际值。使用带 CUDA 的 FAH Python 环境，标准检测器缓存还需要 Ultralytics。每次准备清单、生成缓存和启动实验都使用新输出路径。[FAH 训练环境](FAH_GEM_X_zh.md)

```bash
py=.venvs/fah/bin/python
fah_repo=/path/to/FAH
det_py="$fah_repo/.venv/bin/python"   # 已能运行实际视频检测器的 Python
init="$fah_repo/weights/fah-s/model.pt"
detector=/path/to/deployed_glove_detector.pt
work=/path/to/new_kb_experiment
data="$work/splits"
cache="$work/train_val_proposals.jsonl"

"$py" -m tools.hand_finetune.splits --output "$data" \
  --train recording_A /path/to/recording_A_export \
  --train recording_B /path/to/recording_B_export \
  --val recording_C /path/to/recording_C_export \
  --test recording_D /path/to/recording_D_export
```

若目前只有一段真实录制，用下面的命令替代上述划分命令。按时间先训练、再验证、最后测试。可用 `--test-fraction 0` 暂不留测试集。有共享的 `reference_timestamp_ns` 时，每个同步组须至少有一条时间记录，组内声明值须一致。没有共享时间戳时，多机位录制使用 `sync_index`。单机位可使用一致的 `source_frame_index`，缺省使用 `sync_index`。相同排序时间必须同属一段，各相机的源帧计数允许有偏移。这给出同录制留出估计，跨录制泛化需要新的独立录制。

```bash
"$py" -m tools.hand_finetune.splits --output "$data" \
  --time-holdout recording_A /path/to/recording_A_export --val-fraction .15 --test-fraction .15
```

`splits.json` 记录来源、策略和每段样本数。分割目录通过符号链接引用原导出。在一次对照期间，保持源图像、标签、部署资产和软件版本固定。所有同步组必须互斥，跨录制实验还要求各录制只出现于一个集合。

标签使用相机系米制的 21 个关节、778 个 MANO 顶点以及独立平移 `trans`，物理左/右手用 0/1。加载器按各自的 KB 标定投影并镜像到标准右手。用下面的命令确认训练和验证标签均可读取，额外抽查指尖顺序、左右手、遮挡及原图叠加。若标签保存 MANO 姿态，可通过同一 MANO 资产前向检查关节和顶点的定义。几何数组检查无法量出标签与真实手套的贴合误差。

```bash
"$py" -c '
import sys
from tools.hand_finetune.data import EvaluationFrames
from tools.hand_finetune.fah_experiments import split_protocol
from pathlib import Path
root = Path(sys.argv[1])
print(split_protocol(root))
for split in ("train", "val"):
    frames = EvaluationFrames(root, split)
    print(split, len(frames), "native frames,", sum(len(frame["hands"]) for frame in frames), "hands")
' "$data"
```

用部署时的固定检测器跑训练和验证原图，两组实验共用同一缓存。缓存的一行对应 `recording_id, image, camera, sync_index` 唯一确定的原图，包含全部候选框、置信度和检测器输出的左右手。漏检帧写入空列表，未知手侧写入 `null`。默认与标准视频路径使用相同的 Ultralytics 检测后端。使用快速检测部署时同时指定 `--fast`。

```bash
"$det_py" -m tools.hand_finetune.boxes --dataset "$data" --split train val \
  --detector "$detector" --confidence .3 --nms-iou .5 --device cuda --output "$cache"
```

V2 在每只标注手自己的 KB 标定下检查候选框，只将能生成有效裁剪且监督目标有限、位于虚拟相机前方的候选参与 IoU 匹配，阈值为 0.5。默认一半样本保留标注框，另一半优先使用匹配框。未匹配时按种子、轮次和样本确定顺序，在目标手经传感器边界裁剪的标注框上最多尝试四个不同的实测偏移。若均无法生成有效监督，则使用该手已验证的标注框。改框时重新计算透视裁剪、内参和监督目标。

若训练缓存中没有可用匹配（包括只有验证缓存的情况），须用 `--offset-samples train_offsets.json` 提供从训练录制实测的偏移，否则 V2 会在训练前报错。JSON 为 N×4 数组，四列依次为中心偏移除以标注框宽、高，以及检测框/标注框宽、高比的自然对数。

## 2. 运行 V0 和独立消融

部署评估需要 `$fah_repo/weights/fah-s/config.json`、配置指定的原生网格拟合权重、`data/mano/MANO_{LEFT,RIGHT}.pkl` 和 `data/sam-3d-body-dinov3/{model_config.yaml,assets/mhr_model.pt}`。也可在已安装 FAH 的环境中用 `--release-weights`、`--mano-dir`、`--sam3d` 显式给出相应目录，省略 `--upstream`。

```bash
common=(--dataset "$data" --detector-cache "$cache" --upstream "$fah_repo" --init "$init"
        --batch-size 4 --lr 1e-5 --seed 42 --epochs 1 --max-steps 100 --workers 4 --eval-every 25)

# 单独检查 V0，完整评估验证集的标注框和固定检测框
"$py" train_hand.py fah --version V0 "${common[@]}" --output "$work/check_V0"

# 单卡顺序执行，V0 已完成时可从 --versions 中删去 V0
"$py" -m tools.hand_finetune.fah_suite run --versions V0 V1 V2 V3 V4 \
  --gpus 0 --output "$work/ablations" -- "${common[@]}"
```

单版本可通过 `train_hand.py fah --version V1` 等入口复用相同参数。有多张本地 GPU 时，可将上面的队列命令替换为以下命令。每张卡独立运行一个版本。若作业环境已设置 `CUDA_VISIBLE_DEVICES`，从该掩码中选卡。省略 `--gpus` 时使用第一张可见卡。

```bash
"$py" -m tools.hand_finetune.fah_suite run --versions V0 V1 V2 V3 V4 \
  --gpus 0,1 --output "$work/ablations" -- "${common[@]}"
```

步数为 `min(epochs × ceil(train_hands / batch_size), max_steps)`，省略 `--max-steps` 时遍历全部指定轮数。所有训练版本共用样本顺序和优化步数。默认每轮验证，也可按上述方式每 25 步验证。第 0 步和最后一步均验证。损失、标注框裁剪二维误差和腕部对齐 MPJPE 写入各次记录。两种框的同一份验证驱动完整原生推理，在鱼眼原图量指尖误差，在物理相机系量腕部误差和拟合后的网格顶点误差。

## 3. 选组合和最终权重

需要从 V1–V5 自动选出一个版本继续训练时，筛选队列可使用以下入口。`common` 提供相同的数据、检测缓存和模型资产。请从 `common` 中去掉示例的 `--max-steps 100`。

```bash
"$py" -m tools.hand_finetune.fah_suite run --versions V1 V2 V3 V4 V5 --combine V2 V3 V4 \
  --gpus 0,1,2,3,4 --output "$work/screen" --promote-to-steps 40000 --promote-eval-every 1400 -- \
  "${common[@]}" --epochs 3 --batch-size 32 --eval-every 3189 --seed 42
```

五个版本全部完成相同预算后，队列按各自最低的合格检测框腕部对齐 MPJPE 排名，精确持平时优先较小版本号。每个版本的第三轮末保存模型参数、AdamW 和随机数状态。胜出版本从第三轮末继续到累计目标步数，保留前期验证历史和最佳权重。`promotion.json` 记录排名、晋级版本和总步数，`final/best_model.pt` 保存两阶段合计的最佳权重，`final/model.pt` 保存最后一步。也可用 `train_hand.py fah --version … --resume-run <原运行目录> --output <新目录>` 恢复使用 `--save-training-state` 生成的完整验证断点，需提供一致的数据、组件和优化参数。

`jobs.json` 记录每个版本的参数、GPU 与退出状态，`V*.log` 保存各自日志。一项失败后，队列继续其余作业，最终返回非零退出码。每个成功训练目录有 `model.pt`（最后一次导出）、`best_model.pt`（验证最优）、`metrics.json` 和逐次 `val_step_*.json`。按验证集检测框下的腕部对齐 MPJPE（`detected.metrics.wrist_relative_mpjpe_mm`）选择最优权重。预测和标注各自减去腕关节坐标，计算 21 个关节的三维欧氏距离，先按关节、再按可评分的手取平均，以毫米计，越小越好。网格 MPVPE 作为诊断指标保留。第 0 步也参与选择，持平时保留较早权重。

每个版本在第 0 步记录标注框（`gt`）和检测框（`detected`）各自的标注手数、可评分网格手数及这些手的身份摘要 `mesh_cohort_sha256`。只有两组手的数量和身份均与第 0 步一致、检测框腕部对齐 MPJPE 有效的检查点才可更新最优权重。例如，第 0 步两只手平均 55 mm，后续仅一只手得到 20 mm，这次结果只供诊断。即使数量相同，只要换了手也不能获选。该次权重仍写入 `model.pt`，验证报告及 `metrics.json` 的 `latest` 记录 `selection_eligible=false` 和 `selection_exclusion`，保留已有最优权重，之后符合条件的检查点仍可获选。若第 0 步就没有可评分检测，V0 仅保存基线权重与诊断，V1–V5 保存诊断后在优化前停止。模型推理失败、拟合结果损坏等错误会使作业失败。

阅读验证结果时，两组框各有以下分母，并细分相机和视野中央/边缘。

- 网格和三维误差按手平均。`counts.labeled` 为全部标注手数，`counts.scored` 为可评分网格手数，`coverage` 为两者的比值。漏检、手侧未知或检测框无法形成有效 KB 裁剪（`invalid_proposal`）会影响覆盖。手侧错误时按模型实际给出的手侧评分。
- 原图二维误差按可投影的关节或指尖点平均。`counts.scored_2d` 是全部 21 个关节均可投影的手数，`coverage_2d` 为该手数除以全部标注手数。`counts.invalid_projection` 是网格可评分但部分或全部关节无法投影的手数。这些手仍计入网格和三维误差，其可投影关节仍计入二维误差。
- `metric_denominators` 给出每项误差实际使用的手数或点数，`joint_2d_count_by_index` 给出 21 个关节各自的二维点数。比较二维误差时同时查看这些分母。例如，两只手都有有限网格，其中一只缺一个指尖投影，网格覆盖率为 100%，完整二维覆盖率为 50%，二维关节和指尖均值的分母分别为 41 和 9。

根据完整验证结果指定 V5。以下用 V2+V3 演示语法。

```bash
"$py" train_hand.py fah --version V5 --combine V2 V3 "${common[@]}" --output "$work/chosen_V5"
"$py" -m tools.hand_finetune.fah_suite compare "$work/ablations"/V{0,1,2,3,4} "$work/chosen_V5" \
  --output "$work/validation_comparison.json"
```

对照表记录每个版本的最优步和最后一步、两种框下的指标和样本分母。`release_mesh_spread_mm` 给出两种框下相同发布权重在不同作业间的第 0 步网格 MPVPE 极差。对照按实际训练和验证图像/标签的内容、缓存、发布权重、拟合器、MANO、SAM-3D 资产和 FAH 实现识别输入。每次原生评估记录实际评分的权重、清单、缓存和输入摘要。训练阶段只检查测试清单和文件身份，最终测试时才读取测试图像和标签内容。每组对照使用只读、版本固定的输入。首次计算摘要须顺序读取训练和验证数据，之后的验证复用文件身份、大小及修改时间一致的摘要。

共享的步数、种子、批量大小、头部学习率和验证间隔等须相同。对照命令还会逐一核对各版本的最优检查点与其第 0 步，以及各版本之间两组框的网格评分手数及身份。每行只记录该版本实际使用的组件参数。例如，V1 可直接与使用外部偏移的 V2 或自定 adapter 学习率的 V4 比较。若 V2 和 V5 都启用框混合，则这两行的混合概率和偏移来源须一致。协议版本 4 按腕部对齐 MPJPE 选模。协议版本 3 的历史结果仍可单独对照，显示当时选模使用的网格 MPVPE。对照命令要求所有作业使用相同协议。单个旧权重可以独立复评。需要按新协议比较版本最优结果时，重新运行相应版本。固定组合和最终权重后，再用相同检测器生成独立测试缓存并评估一次。

```bash
selected="$work/chosen_V5/best_model.pt"   # 替换为验证后选定的权重
"$det_py" -m tools.hand_finetune.boxes --dataset "$data" --split test --detector "$detector" \
  --confidence .3 --nms-iou .5 --device cuda --output "$work/test_proposals.jsonl"
"$py" -m tools.hand_finetune.fah_evaluation --checkpoint "$selected" --dataset "$data" --split test \
  --detector-cache "$work/test_proposals.jsonl" --upstream "$fah_repo" --output "$work/final_test.json"
```

## 在统一 FAH 环境中运行合成回归测试

准备好上述发布文件、部署资产、CUDA GPU 及 `pretrained_models/hand/weights/fah-s/model.pt`。让测试进程从头到尾优先导入同一份指定的 FAH 检出，运行包括 GPU 前向、导出与拟合的测试。

```bash
"$py" -m pip install -r requirements-dev.txt
PYTHONPATH="$fah_repo${PYTHONPATH:+:$PYTHONPATH}" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  WILOR_RUN_GPU_TESTS=1 FAH_EVAL_SMOKE_UPSTREAM="$fah_repo" "$py" -m pytest -q \
  tests/test_fah_experiment_entrypoints.py tests/test_external_hand_crops.py \
  tests/test_fah_variant_data.py tests/test_fah_variant_model.py tests/test_fah_evaluation.py
```
