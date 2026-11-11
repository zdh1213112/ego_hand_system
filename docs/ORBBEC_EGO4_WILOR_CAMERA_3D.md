# Orbbec EGO Pro：相机坐标下的 WiLoR 3D 手部与网格

输入录制：`EGO_Pro_AZLR86000F7_20260928_144343`。
主入口为 `scripts/run_orbbec_wilor_3d.py`，输出为**每个相机独立的单目 WiLoR 估计**，
包括 21 个三维关节、778 个网格顶点、1538 个三角面和手腕参考位姿。

## 本机执行

```bash
cd /home/lff/xense_project/ego_hand_system

/home/lff/miniforge3/envs/wilor/bin/python scripts/run_orbbec_wilor_3d.py \
  --session /home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO_Pro_AZLR86000F7_20260928_144343 \
  --output output/EGO_Pro_AZLR86000F7_20260928_144343_wilor_3d_repeat \
  --wilor-root /home/lff/xense_project/WiLoR \
  --device cuda --obj-stride 30
```

默认处理四个相机的全部 885 帧。输出目录必须不存在，换参数使用新目录。
仅测试前 8 帧时，添加 `--cameras cam_0 --max-frames 8`，并使用另一个输出目录。

模型使用本机已有文件：

- `/home/lff/xense_project/WiLoR/pretrained_models/wilor_final.ckpt`
- `/home/lff/xense_project/WiLoR/pretrained_models/model_config.yaml`
- `/home/lff/xense_project/WiLoR/pretrained_models/detector.pt`
- 当前工程的 `models/mano/MANO_RIGHT.pkl`；WiLoR 以右手 MANO 为基础，左手进行一次镜像恢复并反转三角面绕序。
- WiLoR 源码目录中的 `mano_data/mano_mean_params.npz`。

本机 `wilor` Conda 环境的 NumPy 为 2.2.6，但原有 scikit-image/pandas 是不兼容的旧二进制。
已将兼容版本放入工程的 `python_deps/wilor_runtime`，没有替换 Conda 环境中的软件包。
新入口自动加载该目录。以后从干净工程恢复时，可执行：

```bash
/home/lff/miniforge3/envs/wilor/bin/python -m pip install --no-deps \
  --target python_deps/wilor_runtime \
  -r requirements/wilor_camera3d_overlay.txt
```

模型运行需要 GPU。自动化沙箱不暴露 GPU 时会显示 CUDA 不可用；本次实际推理在可访问 GPU 的执行环境运行。

## 实际处理方法

1. 从原始厂商相机 YAML 读取四目 KB 鱼眼内参；读取每路真实帧时间戳。
2. 每个镜头生成 1000×750、水平视场角 130° 的中央虚拟针孔视图，使用真实鱼眼模型重采样。
3. 去畸变使用单位旋转 `R=I`：虚拟视图与原始相机的三维坐标轴一致，只改变投影，不改变三维参考系。原始鱼眼外围超出该虚拟视场的区域不参与预测。
4. 逐相机执行手部检测，每帧左右手分别保留该类别中置信度最高的一个框；WiLoR 输出 MANO 几何、关节、旋转和形状参数。
5. 对左手恢复物理镜像，避免把右手模板直接作为左手输出。
6. WiLoR 原始全图相机平移使用假设焦距，不能直接当作本设备的真实相机深度。
   当前入口保留 WiLoR 估计的手形与朝向，使用已知虚拟相机 K，拟合三维平移，使预测关节的投影尽量匹配 WiLoR 自身预测的二维关节。
7. 将关节和网格导出为原始相机光学坐标下的米制数值，并用原始 KB 投影将网格投回原视频，生成核对视频。

这是**经过相机标定处理的单目模型估计**，不包含四目三角化、几何融合、VIO 世界坐标转换或时序补帧。
平移拟合使用的二维点来自 WiLoR 自身，不是外部独立测量；因此 `placement_fit_rmse_px` 只是内部投影拟合残差，不代表真实定位误差。
手的米制大小继承 MANO 的模型尺度，绝对深度仍受单目歧义、手套、遮挡和模型误差影响，不能作为经过真值验证的测量。

## 坐标定义与相机编号

每个相机各有一套输出坐标，原点为该相机光学中心：

```text
+X 向图像右方
+Y 向图像下方
+Z 向镜头前方
长度单位：米
```

编号与原始录制、此前 VIO 输出保持一致：

| 目录 | 原始视频 | 含义 |
|---|---|---|
| `cam_0` | `_camera_left.mp4` | 左前相机 |
| `cam_1` | `_camera_right.mp4` | 右前相机 |
| `cam_2` | `_camera_right_side.mp4` | 右侧相机 |
| `cam_3` | `_camera_left_side.mp4` | 左侧相机 |

各路结果不共享同一个相机原点。同一只手在不同相机下的 XYZ 数值不同，且单目预测可能彼此不一致。
若要统一到 cam_0，需要使用相机外参进行坐标变换；如果还要改善几何一致性，需要另做多视角关联与融合。

## 输出结构

```text
output/..._wilor_3d/
  run_config.json
  summary.json
  cam_0/ ... cam_3/
    hands_3d.npz
    hand_poses.csv
    camera_calibration.json
    undistortion.json
    mesh_overlay.mp4
    meshes/frame_000000_left.obj ...
    summary.json
```

`hands_3d.npz` 保留所有已处理帧的网格。OBJ 默认每 30 个源帧导出一次，方便直接打开查看。
需要每帧 OBJ 时使用 `--obj-stride 1`；`--obj-stride 0` 关闭 OBJ，仍保留 NPZ 中的全部网格。

## NPZ 字段

设 F 为处理帧数，手维度固定为 `[left, right]`，即第 0 项左手、第 1 项右手。

| 字段 | 形状 | 含义 |
|---|---|---|
| `frame_index` | F | 原始视频解码帧号 |
| `timestamp_us` | F | 原始相机采集时间，微秒 |
| `valid` | F×2 | 是否存在有效手部预测 |
| `confidence` | F×2 | 检测框置信度，不是三维精度 |
| `joints_camera_m` | F×2×21×3 | 相机坐标下的 21 个手部关节 |
| `vertices_camera_m` | F×2×778×3 | 相机坐标下的完整网格顶点 |
| `faces_right` | 1538×3 | 右手网格三角面顶点索引，0 起始 |
| `faces_left` | 1538×3 | 左手三角面索引，已修正镜像绕序 |
| `joints_2d_native_px` | F×2×21×2 | 3D 关节经原始鱼眼标定投影后的像素 |
| `T_camera_hand` | F×2×4×4 | 手腕参考坐标到相机坐标的变换 |
| `mano_global_orient_right_canonical` | F×2×3×3 | 模型原始右手规范下的全局旋转 |
| `mano_hand_pose_right_canonical` | F×2×15×3×3 | 15 个 MANO 关节的局部旋转矩阵 |
| `mano_betas` | F×2×10 | MANO 手形系数，不是角度或位置 |
| `geometry_translation_camera_m` | F×2×3 | 对模型几何施加的平移，已经应用到导出的顶点/关节 |
| `placement_fit_rmse_px` | F×2 | 内部二维投影的拟合残差 |

**`vertices_camera_m` 和 `joints_camera_m` 已经是最终相机坐标，读取后不再加平移。**
未检测到或无法生成有效几何时，`valid=false`，对应几何为 NaN，不用前一帧伪造数据。

21 点顺序为：0 手腕，1–4 拇指，5–8 食指，9–12 中指，13–16 无名指，17–20 小指。

`T_camera_hand` 的原点为关节 0（手腕）。旋转采用 MANO 根部坐标约定，并将左手镜像转换为适当旋转。
它不是按手掌法线重新定义的解剖坐标，也不能直接当作机器人夹爪坐标。
原始 canonical MANO 参数与物理左手几何是不同约定；重建左手时必须应用相同镜像规则，不能再次镜像已导出的物理网格。

`hand_poses.csv` 仅保存有效预测：源帧号、采集时间、左右手、置信度、手腕 XYZ、四元数 `qw,qx,qy,qz` 和内部拟合残差。

## 用 Python 读取或导出某一帧网格

```python
from pathlib import Path
import numpy as np
import trimesh

root = Path("output/EGO_Pro_AZLR86000F7_20260928_144343_wilor_3d/cam_0")
data = np.load(root / "hands_3d.npz")
side = 0  # 0 左手，1 右手
i = int(np.flatnonzero(data["valid"][:, side])[0])

joints = data["joints_camera_m"][i, side]       # (21,3)，米
vertices = data["vertices_camera_m"][i, side]  # (778,3)，米
T_camera_hand = data["T_camera_hand"][i, side]
faces = data["faces_left" if side == 0 else "faces_right"]
trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export("hand_camera.obj")
print("source frame:", data["frame_index"][i], "wrist:", joints[0])
```

此流程不依赖此前 VIO。手部处理使用全部 885 帧；此前 VIO 因 IMU 尾部覆盖限制使用 884 帧，
两者数量不同是正常的。如果后续组合为世界坐标手部位姿，应按原始相机时间关联；图像—IMU 时间修正统一固定为 0 ms。
