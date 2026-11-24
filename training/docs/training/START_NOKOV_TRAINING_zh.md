# NOKOV → MANO 标签训练检查与启动（2026-11-06）

结论：当前标签可以直接用于本项目的 refiner_finetune 流程，已完成真实 GPU 短训练检查。正式长训练尚未启动。

## 已核验的数据

- 数据目录：`/home/p3/data_sda1/EGO_glove/work/20261104/wilor_training_195844_195732_cam01`
- 3541 对图片/标签，camera0 1771 对、camera1 1770 对，全部为物理左手。
- 训练 2979 对：195844/195852 场次的原始结果（未使用 3 mm、RGB 或表面附着试验结果）。
- 验证 562 对：195732/195743 独立录制场次；训练/验证场次没有交集。
- 全部图片可解码；标签形状、有限数值、正深度、图片对应关系、投影和 MANO 回放均通过。
- 训练 MANO_RIGHT.pkl 的 SHA-256 与标签导出时一致。
- 左手几何由加载器镜像到右手标准姿态，MANO 参数已是 right_canonical，不能再次手动镜像参数。
- `joints_2d` 字段虽然命名为关节，实际为 778 顶点投影；加载器通过 `joints_3d + trans` 和 `K` 重新投影得到训练用的 21 个关节。

## 环境和短训练

本机 `wilor` conda 环境没有 PyTorch。已验证的环境为 `/home/p3/miniforge3/envs/ego-hand/bin/python`：PyTorch 2.11.0+cu128、CUDA 12.8、RTX 5090 D v2（约24 GB）。

短训练配置：`configs/glove/ego_nokov_20261104_smoke.yaml`。
已完成3个优化步，每步4张×8次梯度累积，验证限制为2个batch（8张），最终可视化4张。
程序退出码0；2D/3D/coarse/MANO监督全部生效，loss和梯度有限，best/last检查点、曲线、验证可视化已生成。
短训练只证明数据加载、前向、反向、验证和保存可运行，不代表收敛或最终精度。

检查报告：`../work/20261106/wilor_training_check/label_audit.json`
短训练日志：`../work/20261106/wilor_training_check/smoke.log`
短训练输出：`../work/20261106/wilor_training_check/smoke_output/`

## 正式启动

在192.168.111.183服务器上运行：

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
mkdir -p output/ego_nokov_20261104_original
nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   /home/p3/miniforge3/envs/ego-hand/bin/python train_glove.py   --config configs/glove/ego_nokov_20261104_original.yaml   > output/ego_nokov_20261104_original/train.log 2>&1 &
tail -f output/ego_nokov_20261104_original/train.log
```

配置使用原始 `pretrained_models/wilor_final.ckpt` 初始化，`init_weights: null`，冻结ViT，训练pose/shape/camera heads和RefineNet；batch_size=4、梯度累积8（有效batch32）、2000优化步、AMP；每50步保存并对完整562张验证集评估；最终可视化前32张验证样本。

输出目录 `output/ego_nokov_20261104_original/` 内包含：

- `best_trainable.pt`：按验证loss选择的最佳权重；
- `last_trainable.pt`：最后一次保存的权重和优化器状态；
- `train_report.json`、`loss_curves.png`、`resolved_config.yaml`；
- `val_visualization/`：最终验证可视化。

本次只运行短训练，没有启动2000步正式训练。重复从头运行会写入同一个正式输出目录；保存独立实验时同时修改experiment.name和paths.output。续训时复制正式YAML、仅将paths.resume设为last_trainable.pt的绝对路径，保持数据、优化器、loss等设置一致，再用该配置启动。输出是本项目的trainable adapter格式，推理应通过本项目load_wilor(..., adapter_path=...)加载，并保留原始WiLoR权重。

## 标签质量的实际范围

这些标签是NOKOV约束下的MANO拟合结果。几何自洽不等于相对真实皮肤表面的独立精度验证。
导出清单记录的逐样本NOKOV关节拟合中位误差，其中位数约4.74 mm；逐样本95分位误差的中位数约18.17 mm。
图像重投影的逐样本中位误差，其中位数约2.98 px。数据目前只有两段录制、单侧手，适合先做该场景的微调基线，泛化能力需要新增独立录制、人物和手势来评估。

上游参考：https://github.com/rolpotamias/WiLoR 。本次启动结论以服务器实际代码与上述实测报告为准。
