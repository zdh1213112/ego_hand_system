# 最新汇总标签 trainset_v1 训练

本文对应本机 2026-11-14 准备的 `datasets/trainset_v1`，不是 11 月 4 日旧数据。
在仓库根目录执行命令；数据、模型和训练结果仍由 Git 忽略，需在运行机器上单独准备。

## 数据与配置

数据来自两段 2026-11-11 录制的最新 joint_refit 原图标签：

- `pair_20261111_191855_native_20261113_211401/optimized/dense/fit/final/export/merged`
- `pair_20261111_192916_native_20261113_190658/optimized/dense/fit/final/export/merged`

本机对应工作目录位于同级 `../work/`。
合并后为 10451 张原图、13874 份手标签（左手 3423、右手 10451），涉及 camera0～camera3。
`splits.json` 使用 `time_holdout`：每段录制最后约 10% 的时间用于验证，
共 12468 个训练实例、1406 个验证实例；这不是整段录制隔离的验证。

`dataset_profile.json` 记录几何投影校验和全部 MANO 标签重放通过。
`summary.json` 仍为 `approved_for_training: false`，表示贴合质量尚未完成正式验收；
几何一致性和训练链路通过不等于标签已达到最终质量。

可提交配置为 [`configs/glove/trainset_v1_joint_lora.yaml`](../../configs/glove/trainset_v1_joint_lora.yaml)：

- 完整 joint LoRA + coarse heads + RefineNet + 物理相机头训练；原始 ViT 和 MANO 冻结。
- 原始鱼眼图像配合 `camera_model: physical_kb`。
- 所有 ViT block 的 `attn.qkv` / `attn.proj`，LoRA rank 4、alpha 8、dropout 0.05。
- 4000 个优化步，batch 4 × 梯度累积 8，有效 batch 32。
- LoRA / coarse heads / RefineNet 学习率均为 1e-5，物理相机头学习率 3e-4。
- 每 100 步保存、全量验证，结束后全量生成验证可视化。

这些参数来自本机 `output/trainset_v1_full.yaml`，仅将路径改为等价相对写法、
将输出目录改为 `output/trainset_v1_joint_lora/`，保留已有 `output/trainset_v1_check/` 结果。
原文件中关于“仍为 200 步”的旧注释与实际 `max_steps: 4000` 不一致，此处以有效配置和运行报告为准。

## 初始化权重

沿用此前真实训练使用的 `output/native_kb_training_20261111/coarse_start.ckpt`。
该文件基于 `pretrained_models/wilor_final.ckpt`，只将 RefineNet 的 pose / cam / shape
三个输出层的六个权重、偏置张量清零，使初始预测从 coarse 分支开始。
它与直接使用原始 WiLoR 初始化不同；复现本次训练时须保持一致。

本机文件已经存在。新检出仓库在准备好原始权重后，可按以下方式生成缺失的初始化文件；
已有文件不会被覆盖，原始 checkpoint 也不会修改：

```bash
python - <<'PY'
from pathlib import Path
import torch

source = Path('pretrained_models/wilor_final.ckpt')
target = Path('output/native_kb_training_20261111/coarse_start.ckpt')
if target.exists():
    raise SystemExit(f'Already exists: {target}')
payload = torch.load(source, map_location='cpu', weights_only=False)
keys = [f'refine_net.{head}.{field}'
        for head in ('dec_pose', 'dec_cam', 'dec_shape')
        for field in ('weight', 'bias')]
assert all(key in payload['state_dict'] for key in keys)
for key in keys:
    payload['state_dict'][key].zero_()
target.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, target)
print(target)
PY
```

## 启动新一轮训练

激活已安装运行依赖的训练环境后执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_glove.py --mode joint_lora \
  --config configs/glove/trainset_v1_joint_lora.yaml
```

本机后台运行示例：

```bash
mkdir -p output/trainset_v1_joint_lora
nohup env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py \
  --mode joint_lora --config configs/glove/trainset_v1_joint_lora.yaml \
  > output/trainset_v1_joint_lora/train.log 2>&1 &
tail -f output/trainset_v1_joint_lora/train.log
```

该配置 `resume` / `init_weights` 均为 null，从 coarse_start 重新开始。
再次从头运行时为 `paths.output` 选择新目录；配置固定的输出目录不会自动创建新的实验版本。
单独执行 `--mode joint_lora` 仍走 11 月 4 日旧配置，最新数据必须显式指定上述 YAML。

## 已完成训练及其评估

已有报告 `output/trainset_v1_check/train_report.json` 记录：

- `status: passed`、`current_step: 4000`。
- 最佳验证 loss 对应 step 3900，`best_val_loss: 0.02601623617292807`。
- 最佳与最后权重分别为 `output/trainset_v1_check/best_trainable.pt` 和 `last_trainable.pt`。

虽然目录名包含 `check`，报告对应完整的 4000 步运行，不能仅凭目录名认定是 200 步短训。
原配置 `output/trainset_v1_full.yaml` 指向该目录，直接重跑会写入已有结果，因此新配置使用独立输出目录。

单独评估已有最佳权重：

```bash
python evaluate_glove.py \
  --root datasets/trainset_v1 \
  --checkpoint output/native_kb_training_20261111/coarse_start.ckpt \
  --adapter output/trainset_v1_check/best_trainable.pt \
  --camera-model physical_kb --batch-size 4 --device cuda:0 \
  --output output/trainset_v1_evaluation_new
```

评估新一轮训练时，将 `--adapter` 改为 `output/trainset_v1_joint_lora/best_trainable.pt`，并使用新的评估输出目录。
