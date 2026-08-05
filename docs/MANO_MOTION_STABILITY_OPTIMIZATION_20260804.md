# MANO 运动稳定性优化（2026-08-04）

## 目标

本轮针对手部移动时 MANO 网格出现的整手旋转、平移和关节姿态突变。输入仍是已完成鱼眼校正、PTS 同步、双目关联和三角化的 21 点，不回退为单目 MANO。

## 已实现策略

- 将手指 PCA 姿态时序项与整手全局旋转/平移时序项分开计算。
- 根据有效 3D 点数量、深度置信度和左右图 2D 位移生成逐帧观测质量。
- 低质量帧提高时序约束，快速且高质量的真实运动适当放宽更新范围。
- 对姿态、全局旋转和平移增加速度及加速度约束。
- 滑动窗口的重叠帧固定为上一窗口的显式边界锚点，避免窗口切换产生新解。
- 使用 SO(3) 相对旋转而不是直接比较 axis-angle 向量，避免正负 pi 表示造成误判。
- 全局旋转、平移和极端 PCA 姿态变化设置自适应信赖域，拒绝单帧局部最优解切换。
- 实时拟合器使用同一观测质量策略，并在 2D 几乎不动时收紧刚体更新；观测不足时主要冻结手指姿态。

曾测试掌心驱动的最终图像空间平移，以及受限 2D 相似变换。它们会让掌心局部更贴，但会显著增大整手和指尖误差，因此未启用为正式默认策略。

## 398 帧离线回归

正式候选使用 `mano_preparation_follow` 和上一版 `mano_fit_follow` 作为 warm start。中位运动跟随残差基本保持：右手 4.43 -> 4.13 px，左手 4.88 -> 4.88 px。

明显突变得到抑制：

| 指标 | track 0 旧版 | track 0 新版 | track 1 旧版 | track 1 新版 |
|---|---:|---:|---:|---:|
| 最大单帧全局旋转 | 167.88° | 79.02° | 173.31° | 77.98° |
| 最大单帧平移 | 160.29 mm | 87.39 mm | 208.36 mm | 83.59 mm |
| 最大关节姿态 RMS 变化 | 1.0717 rad | 0.6964 rad | 1.0626 rad | 0.5723 rad |
| 中位左图重投影误差 | 10.87 px | 11.28 px | 7.61 px | 7.99 px |

代价是严重遮挡或高速运动帧不再追随不可信的瞬时检测，因此极端帧可能出现短暂滞后；这是为了消除更明显的模型翻转和跳跃。

## 复现命令

```bash
conda activate ego-hand
cd /home/zdh/ego_hand_system

python scripts/fit_mano_sequence.py \
  --input output/mano_preparation_follow/mano_input.npz \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --output output/mano_fit_motion_guard \
  --initial-output output/mano_fit_follow \
  --shape-iterations 0 \
  --pose-iterations 80 \
  --pose-window 24 \
  --pose-overlap 8 \
  --learning-rate 0.0015 \
  --w-3d 1.0 --w-2d 0.80 --w-pose 0.0025 \
  --w-temporal 0.012 --w-rigid-temporal 0.005 \
  --w-acceleration 0.006 --boundary-weight 0.04 \
  --max-orient-step-deg 75 \
  --max-translation-step-m 0.08 \
  --max-pose-step 6 \
  --no-image-rigid-alignment \
  --device cuda --no-video
```

实时启动命令不变：

```bash
cd /home/zdh/ego_hand_system/scripts
./run_ego_live.sh
```

实时策略新增可调参数：`--mano-max-orient-step-deg`、`--mano-max-translation-step-m` 和 `--mano-low-quality-freeze`。默认值分别为 75°、0.08 m 和 0.22。

