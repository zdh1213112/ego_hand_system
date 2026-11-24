# 24 点附着位置试拟合（显式开启的实验）

**20261112 更正：旧版选择了错误的手心侧附着区域。原先关于模型退化原因的判断不完整，不能据此否定 24 点方案。**

当前修正版：http://127.0.0.1:8766/marker24_fixed/ ，页面标题为 `DORSAL-FIX-1`。已重跑双手并重新生成四路视频。输出目录为 `output/native_kb_training_20261111/marker24_pilot_hemisphere_minus`。

默认手背候选使用 `--surface-hemisphere -1`，并根据已确认的手背照片检查掌部与非拇指指根面朝向相机。检查失败会报错，不能继续拟合。两手的旧错误区域均被回归检查拒绝，新区域通过；见 `check_surface_direction.py` 与输出中的 `direction_regression.json`。

方向修正前→后：细化时留出的指尖残差中位数，左手 4.70→2.77 mm，右手 5.43→2.40 mm。此指标仍不是独立的解剖精度，当前依然不自动替换基线标签。照片点击坐标仍不作为独立像素真值。

下方 `v1_run2` 数值与目录为保留的旧版历史记录，已因方向错误失效。

已生成的对照：http://127.0.0.1:8766/marker24_pilot/ 。由 `label_export/pipeline/calibration/marker_placement/start_viewer.sh` 提供本地服务。

重新运行本次录制的实验，自动使用新的输出目录：

```bash
bash label_export/pipeline/calibration/marker24_pilot/run.sh
```

当前脚本针对 `pair_20261111_192916`，使用 `marker24_calibration_input/placement_user_confirmed.json`。不是所有新录制通用入口。基线导出命令及默认行为没有改变。

算法与限制：

- 从原始 NOKOV 读取全部 24 点，沿用已验证的时间映射、头部刚体与四目外参。重建出的原有 21 点与既有对齐结果差异小于 0.0001 mm。
- MANO 使用原有左右手 canonical 约定，冻结基线 betas。用户确认的解剖区域决定手指候选网格区域；HandOffset 限定在掌部。照片中的点击坐标没有作为独立 RGB 真值参与损失。
- 21 个表面 marker（HandOffset + 20 手指点）各采用跨帧共享的三角形及重心坐标；在限定区域内交替更新三角形、重心坐标和逐帧姿态。
- 三个腕部 marker 使用各自相对于腕部根节点的固定局部坐标，权重为 0.2。这是袖口运动的近似模型，不能认为它们实际严格构成刚体。
- 表面法线偏移始终为 0，不扣除 2 mm。腕部局部坐标不是球半径补偿。姿态保留基线锚定和姿态先验，本轮未优化手型，未增加时间平滑项。
- 每手 80 帧标定。sync 512～703 不参与附着位置标定；验证段左手 20 帧、右手 24 帧。在验证段姿态优化中排除五个指尖 marker，检查预测误差。
- 注意：基线手型和初始姿态本身曾使用全片段及指尖。上述划分只是附着位置训练/检查分离，不是整个系统严格独立的测试集。点选数据又来自 NOKOV 投影预填，不能证明真实内部关节中心准确。

本轮结果目录：

`output/native_kb_training_20261111/marker24_pilot_v1_run2`

- `left_pilot.npz` / `right_pilot.npz`：候选模型、固定附着位置、基线、目标 24 点及划分。
- `summary.json`：拟合与检查指标，`approved_for_full_export=false`。
- `verification.json`：合成多帧附着恢复、划分互斥、重心坐标与有限性、基线投影一致性检查。
- `review/index.html`：原图 / 基线 / 新方案三栏交互检查。
- `review/camera0_comparison.mp4` ～ `camera3_comparison.mp4`：四路 H.264 对照视频，每路 20 个共同验证时刻。按采样顺序等间隔播放，不表示原始录制连续时间轴。

查看器包含部分未达到原训练标签导出条件的局部视图，用于诊断；它们不是新训练标签。

结论：marker 拟合残差下降，但尚未证明 mesh 更贴合。新模型关节相对基线移动约 11 mm，排除于新姿态优化之外的指尖没有一致改善。关闭腕部约束的对照实验也未解决指尖问题，因此不能只归因于腕部点。保留实验结果，不替换基线，不进行全量标签导出。

## 保留正确手指姿态的对照（POSE-LOCK-1）

用户指出方向修正版仍改变了原本正确的手指姿态。已补做两种对照，均未替换基线：

- 全 24 点自由姿态：`http://127.0.0.1:8766/marker24_all/`。附着标定结果与指尖留出版本逐项相同，只恢复五个指尖约束；指尖残差约 0.41 mm，但这是拟合残差，不能证明姿态正确。
- 锁定基线手指姿态：`http://127.0.0.1:8766/marker24_pose_locked/`，标题 `POSE-LOCK-1`。固定 betas 和每帧手指参数，只调整整体旋转、平移及跨帧共享附着参数。查看器可以切换到“全24点自由姿态”对照，保持同一帧和缩放。

锁定版本命令：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python \
  label_export/pipeline/calibration/marker24_pilot/run_pilot.py \
  --enable-marker24 --fit-all-markers --freeze-finger-pose \
  --output /你的新实验目录
```

当前结果在 `output/native_kb_training_20261111/marker24_pilot_pose_locked`。`pose_lock_verification.json` 确认两手手指参数与手型逐项等于基线，消除整体刚体变换后的网格差异不超过 0.00013 mm（浮点误差量级）。这不等于二维轮廓或整体位置通过验收；仍为实验结果。
