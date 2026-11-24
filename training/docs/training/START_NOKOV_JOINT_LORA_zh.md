# NOKOV 联合 LoRA 训练（2026-11-06）

源目录：`/home/p3/data_sda1/WiLoR`（大写）；集成目录：`/home/p3/data_sda1/EGO_glove/wilor_training`。
新增recipe为 `joint_lora_finetune`：相机对齐NOKOV→MANO标签驱动完整coarse→MANO→RefineNet→MANO路径，同时更新ViT LoRA、pose/shape/camera heads和RefineNet，冻结原始ViT及MANO权重。无需先跑OpenTouch。

## 正式启动

在192.168.111.183执行：

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
mkdir -p output/ego_nokov_20261104_joint_lora
nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --mode joint_lora --config configs/glove/ego_nokov_20261104_joint_lora.yaml > output/ego_nokov_20261104_joint_lora/train.log 2>&1 &
tail -f output/ego_nokov_20261104_joint_lora/train.log
```

前台可用 `/home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py --mode joint_lora`，默认已选择同一本机配置。旧的 `--config configs/glove/ego_nokov_20261104_original.yaml` 仍为冻结ViT基线。
本次仅跑3步短训练，没有启动2000步正式联合训练。

## 本机配置

- 数据：`../work/20261104/wilor_training_195844_195732_cam01`；2979张训练，562张验证，按录制场次隔离。保留当前NOKOV数据加载器和左右手标准化，不重导出标签。
- 原始 `pretrained_models/wilor_final.ckpt` 初始化；init_weights和resume均为null。
- 32个ViT block的attn.qkv/attn.proj/mlp.fc1/mlp.fc2，共128个LoRA模块；沿用源方案rank16、alpha32、dropout0.05。
- 共18826528可训练参数：LoRA10485760、heads24339、RefineNet8316429。
- batch4×梯度累积8，有效32；2000优化步、AMP。每50步对完整562张验证，结束输出前32张验证可视化。
- LoRA/head/RefineNet学习率分别1e-5/1e-4/1e-5。相对源模板的LoRA 1e-4，先降为1e-5，效果需实测；batch和输出数量也按当前机器/基线调整。
- 与基线相同监督：coarse/final 2D、手腕相对3D和profile允许的MANO参数。

## 输出及续训

独立输出到 `output/ego_nokov_20261104_joint_lora/`，包含best_trainable.pt、last_trainable.pt、train_report.json、loss_curves.png、resolved_config.yaml、val_visualization/。
best按最低val_loss保存，同时检查val_metric（手腕相对MPJPE，mm）、投影和图片，与原基线在同一验证集比较。权重包含LoRA结构声明和训练参数，推理仍需原始WiLoR权重和MANO资产。
续训请复制正式配置，仅将paths.resume改为该联合实验last_trainable.pt，保持数据、优化器、loss、batch/累积等设置一致；跨训练方式继承权重用paths.init_weights。重复从头运行同目录会覆盖日志和同名输出，新实验应修改experiment.name与paths.output。
评估可使用本项目evaluate_glove.py的--root、--adapter和--output参数。

## 检查证据

目录：`../work/20261106/joint_lora_integration_154423/`。

- smoke.log、smoke_output/train_report.json：公共CLI真实GPU短训3步，前向/反向/验证/保存/最终可视化通过。短训每次验证仅8张、可视化4张，不代表完整验证精度或收敛。
- gradient_and_freeze_audit.json：128个LoRA模块的A/B（256参数张量）、三个head和RefineNet均获得非零有限梯度，3步后所有可训练参数张量均实际更新；395个冻结参数张量前后SHA-256一致。保存权重从原始模型重新加载后与训练结束参数逐项一致。
- 本次进程峰值PyTorch allocated约2948MiB、reserved约3012MiB，不含其他进程。
- cpu_tests.log：66 passed、1 failed、3 deselected，另有5个subtest通过。唯一失败为原有test_glove_config.py要求旧stage2_mano的batch16，但配置本来是256。配置、测试与旧engine均与集成前/HEAD相同，见legacy_test_mismatch.json；本次保留旧设置。
- before/保存原文件，source/保存源方案快照，integration_manifest.json记录来源和哈希。

源方案的lora_stage1/lora_stage2/two_stage入口与模板同时保留，但需要另外准备OpenTouch数据和阶段权重，本次未跑该两阶段训练。
