# 左右手个人手型＋固定 marker 附着标定：第一阶段结果

当前状态：已完成 marker 驱动的分阶段标定和对照；**未接入独立原图轮廓／指尖约束，未通过贴合验收，不可当作正式真值标签**。保留全部基线输出。没有自动加入标签导出流程。

## 做了什么

- 从有效 NOKOV24 中排除非有限点、单帧跳变大于 50 mm 的样本，按姿态和整体朝向差异选取每手 48 个标定样本。该筛选不等于完成了原图运动模糊筛选。
- 整个 sync [512,704) 留出，包括 632 帧；验证时固定个人手型及附着参数，只估计有界姿态、旋转和平移。基线初始化本身曾用过验证帧 marker，故不是完全独立的精度评测。
- A：固定基线手型和姿态，拟合共享附着位置；B：固定附着与姿态，拟合共享 betas；C：在此初始化上联合细化，并对姿态变化加界限。
- 附着点按用户确认的解剖区域选择三角面，每手共享面片和重心坐标；本轮离散面片选定后固定。三个腕点采用根部局部坐标，权重 0.2，不强制当作手掌表面点。
- 附着法向偏移允许 0～4 mm，仅向外；不固定减去 2 mm。这里是有效表面到球心偏移，不是测量得到的手套厚度。
- 每帧平移参数增量限制 8 mm，旋转轴角增量范数限制 0.06 rad，手指参数各分量变化限制 0.12 rad。界限是实验保护条件，不是生理精度证明。
- 同流程的固定手型对照，以及使用互不重叠的 128 帧时间块进行两组标定，分别检查手型贡献与稳定性。

## 结果

| 项目 | 左手 | 右手 |
|---|---:|---:|
| 个人手型方案：验证 marker 残差中位数 | 2.86 mm | 3.17 mm |
| 固定手型对照：验证 marker 残差中位数 | 4.03 mm | 4.58 mm |
| 两组标定所得中性手型表面差异中位数 | 0.32 mm | 0.20 mm |
| 两组标定所得中性手型表面差异 P95 | 0.70 mm | 0.33 mm |

残差只衡量当前附着模型对 marker 的拟合，不是 mesh 到真实手套的距离。两组标定共享初始化、模型与先验，参数接近不等于真实手型已确定。相对原基线，中性手型表面改变中位数约 6 mm，必须进一步由真实图像约束检验。

632 帧右手：相对基线整体旋转变化约 0.80°、腕点移动约 4.04 mm。旧 pose-locked 实验是 6.93°／21.54 mm；这次存在明确运动界限，因此不能仅凭移动减小宣称更准确。原图叠加仍可见指尖和轮廓偏差。

左手 17 个、右手 14 个表面 marker 的法向偏移落在零边界。这提示仅靠当前 marker 模型仍不足以解释所有表面关系；可能包含附着位置、手型表达、手套厚度或时空对齐等因素，尚未唯一定位。没有扩大参数范围来强行压低残差。

初版 signed-offset v1 出现向网格内部拉球心，已标记为被 v2 替代。生产标签从未使用这些实验。

## 看结果

页面：http://127.0.0.1:8766/personal_calibration/ 。原图／基线／新模型三栏同步查看；不是要求用户逐点复核。

- `review/camera0_632_crop.jpg`：问题帧放大对比。
- `review/camera0_comparison.mp4` ～ `camera3_comparison.mp4`：四路采样对比视频，每路 20 个共同验证时刻。3.75 张采样图／秒，不是原视频连续速度。
- `left_pilot.npz`、`right_pilot.npz`：共享参数、每帧拟合结果和原始观测。
- `shared_calibration.json`：可读的固定参数；包含实验状态和坐标说明。
- `verification.json`：参数重建、训练验证分离、上下界及稳定性检查。

## 干净命令

当前脚本针对 pair_20261111_192916，尚非任意录制的通用生产入口。新输出目录必须不存在。

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
PY=/home/p3/miniforge3/envs/ego-hand/bin/python
$PY tools/label_pipeline/calibration/personal_hand_calibration/run.py --enable-personal-calibration   --output output/native_kb_training_20261111/personal_calibration_new   --steps 120 --frames 48
$PY tools/label_pipeline/calibration/marker24_pilot/build_review.py   --pilot output/native_kb_training_20261111/personal_calibration_new
```

对照参数：`--freeze-shape` 固定手型；`--subset even` 或 `--subset odd` 使用不同时间块（建议各 32 个标定样本）。默认基线导出命令不受影响。

## 已验证及下一步

八份双手拟合产物通过参数重建检查，重建 mesh 和球心与保存数据一致；检查了样本分离、方向、参数边界、固定手型对照以及输出有限性。没有宣称通过独立图像精度验收。

下一步必须接入可靠的原图观测：在多视角下区分可见轮廓、指尖和遮挡区域，先验证观测提取质量，再用于标定／验证。当前仅 marker 阶段的结果暂不接入正式标签导出，也不应通过自由表面变形来掩盖上述偏差。


## 已接入显式配置入口

原始录制入口现可通过 `--optimization-config configs/native_personal_marker24.json` 启动完整实验链路。默认关闭，左右手各自共享一套 betas。各阶段支持 context 和 source，不再要求修改代码中的历史实验目录。

[统一命令与配置](/home/p3/data_sda1/EGO_glove/ego_hand_system/docs/EGO_NOKOV_PERSONAL_MARKER24_COMMANDS_zh.md)。
