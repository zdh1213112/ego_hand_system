# EGO-4B 四路 WiLoR 关节点与 mesh 连续视频

使用 `scripts/render_ego4_wilor_mesh_video.py`，直接对原始 EGO-4B 四路 MP4 运行项目中的 WiLoR。支持 `part0001` 等分段录制，通过各路 PTS 时间戳同步；每组同步帧均写入视频，检测不到手时继续保留原画面。

输入目录 `/home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand` 共有 7 段录制。本次按用户最后要求，在前 5 段完成后停止后续推理，最终视频包含 12,209 组同步帧，30 FPS，时长约 6 分 47 秒。最终结果位于 `output/20261111_ego4b_wilor_2906_full`。

单独复现一段到新目录的命令：

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system
/home/p3/miniforge3/envs/ego-hand/bin/python -u \
  scripts/render_ego4_wilor_mesh_video.py \
  --input /home/p3/data_sda1/EGO_glove/data/20261111/20261111_nokov_hand \
  --detector models/wilor/2.0_glove_2906.pt \
  --session-filter pair_20261111_174236 \
  --output output/ego4b_wilor_2906_single_example
```

检测器为用户指定的 `2.0_glove_2906.pt`，其类别为 `0=left, 1=right`；关节和 MANO mesh 由 `models/wilor/wilor_final.ckpt` 重建。左右手分别保留最高置信度检测，检测阈值 0.3。使用项目现有批量推理实现及 FP16，不跳过 backbone 层。各相机分别预测，未应用 NOKOV 标签拟合或多视角三维融合。

四宫格为 `camera0=left`、`camera1=right`、`camera2=right_side`、`camera3=left_side`。每路显示检测框、21 个关节点、彩色骨架、半透明 MANO 表面；右上角给出固定斜视角的独立三维 mesh。左手青色，右手橙色。输出为 H.264 MP4、1920×1304、30 FPS。

输出目录包括：

- `ego4b_wilor_joints_mesh_all_4view.mp4`：已完成选段按录制顺序无损拼接的长视频，本次包含前 5 段。
- `play_loop.sh`：用 VLC 循环播放长视频。
- `pair_*/wilor_joints_mesh_4view.mp4`：每段完整视频。
- `pair_*/preview_*.jpg`：每段开头、中间和结尾的预览。
- `pair_*/wilor_geometry.npz`：四路全部帧的预测几何。
- `pair_*/synchronized_frames.csv`：输出帧与四路原始帧、时间戳的对应表。
- `summary.json`、`run_config.json`：帧数、检测数量、运行参数及模型 SHA-256。

`wilor_geometry.npz` 的前 3 个维度依次为 `[同步帧, camera0..3, left/right]`。`valid` 标识当前帧有预测的手；无预测时浮点数组为 NaN。包含 21 点二维/三维关节、778 点 MANO 顶点、相机平移、检测框、置信度、MANO 三角面以及投影所用焦距。顶点使用 FP16 存储，关节和相机平移为 FP32。三维关节及顶点在各相机局部预测坐标下，投影前加 `camera_translation`，不代表跨相机统一坐标。

mesh 渲染可使用同目录的 `wilor_mesh_rasterizer.cpp` 加速：自动用 `g++` 编译到临时缓存，使用深度缓冲处理遮挡；没有编译器时自动回退到 OpenCV 三角面绘制。此加速不改变模型预测。

可使用 `--session-filter pair_20261111_174236 --max-frames 120` 在新输出目录生成 4 秒验证短片。相同参数重复运行会复用已完成的整段；不同参数应指定新输出目录。
