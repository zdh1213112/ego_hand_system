# MANO 21自由度双手仪表板

日期：2026-08-04

## 目标

参考外部演示视频，在原始 EGO 鱼眼画面上保留双手半透明 MANO 网格，并将右侧面板升级为：

- 右手21自由度弧度条；
- 左手21自由度弧度条；
- 左右手独立 MANO 3D 预览；
- 手离场时同时隐藏角度与3D模型。

## 21自由度定义

### 拇指（5）

1. `thumb_cmc_flex_rad`：CMC屈伸。
2. `thumb_cmc_abduction_rad`：CMC张合。
3. `thumb_cmc_opposition_rad`：CMC对掌/掌骨轴向旋转。
4. `thumb_mcp_flex_rad`：MCP屈伸。
5. `thumb_ip_flex_rad`：IP屈伸。

### 食指、中指、无名指、小指（各4）

1. MCP屈伸。
2. MCP张合。
3. PIP屈伸。
4. DIP屈伸。

总数为 `5 + 4 × 4 = 21`。

## 计算方法

1. 使用每条轨迹已拟合的 `betas` 生成同手型的 flat-hand MANO 关节。
2. 由掌根、食指 MCP 和小指 MCP 建立掌面法向。
3. 由各关节到子关节的骨方向与掌面法向建立屈伸轴。
4. 将 MANO mean-pose 相对的局部轴角向量投影到屈伸轴、掌面法向和拇指掌骨轴。
5. 对左右手镜像屈伸轴统一符号方向，再使用前后各2帧局部中值抑制单帧跳变。

这些值是基于 MANO 指数坐标旋转向量的运动学分量，不是对旋转矩阵进行临床解剖轴标定后的医疗量角。

## 正式结果

- 处理同步帧：398。
- 可见手实例：729。
- 输出视频：1920×1080，398帧，13.27 s。
- CPU渲染速度：28.64 FPS。
- 21自由度数值范围：右手 `[-1.21, 2.99] rad`，左手 `[-1.92, 2.17] rad`。
- 平滑后逐帧变化的 P95：右手约 `0.20 rad`，左手约 `0.23 rad`。
- 原始鱼眼与校正投影链最大差异：`2.54e-13 px`。

遮挡区、轨迹结束前后抽查通过；右手轨迹结束后，右手角度卡和3D预览均显示 `NOT VISIBLE`，左手继续输出。

## 产物

- `output/mano_overlay_21dof/mano_overlay_21dof.mp4`
- `output/mano_overlay_21dof/preview_montage.jpg`
- `output/mano_overlay_21dof/mano_joint_angles_21dof.csv`
- `output/mano_overlay_21dof/mano_joint_angles.csv`
- `output/mano_overlay_21dof/mano_pose_axis_angle.csv`
- `output/mano_overlay_21dof/summary.json`

## 复现

```bash
cd /home/zdh/ego_hand_system
conda activate ego-hand
unset PYTHONPATH

python scripts/render_mano_overlay_angles.py \
  --session data/recordings/Orbbec_Ego_AZER764008C_20260803_190034 \
  --mano-fit output/mano_fit_refined \
  --mano-source third_party/MANO \
  --model-dir models/mano \
  --stereo-frames output/mediapipe_stereo/stereo_frames.csv \
  --output output/mano_overlay_21dof
```
