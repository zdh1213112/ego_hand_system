# 输出字段

WiLoR 使用共享 MANO 模型。三维长度单位米，二维点与框为原图像素。四路原图不去畸变。`side=0` 左手，`side=1` 右手。

## 视频与骨架（A、B）

`final_shared_4view.mp4` 是四路同步原图的 2×2 拼接，各视角投影同一共享模型。帧率为同步时间戳间隔中位数的倒数；精确时间关系以同步表为准。

`joints3d.jsonl` 每同步时刻一行，含 `sync_index`、camera0 `timestamp_ns` 和 `hands[]`。每只手包含 `side`、`handedness`、`joints_base_m: [21,3]`，顺序为腕、拇指、食指、中指、无名指、小指，各指由根到尖。无有效模型时 `hands=[]`，不丢帧。当前每帧最多一只左手和一只右手，不提供多人身份跟踪。

## 网格和参数（C、D）

用 `torch.load(path, map_location='cpu', weights_only=False)` 读取 `hand_model.pt`：

| 字段 | 含义 |
| --- | --- |
| format | `shared_wilor_mano_v1` |
| export_schema | `ego_hand_model_v1` |
| backend / units / coordinate_system | 固定算法标签 wilor、米、标定参考系 |
| topology.left / topology.right | 各侧 `[F,3]` 三角面索引，全序列共享 |
| frames | 每时刻 `sync_index`、`timestamp_ns`、`hands` |

每只手的字段：

| 字段 | 含义 |
| --- | --- |
| side | 0 左，1 右 |
| joints_base_m | WiLoR 21 个参考系关节 |
| vertices_base_m | `[V,3]` 参考系顶点；MANO 为 778 顶点 |
| parameters | 拟合后的原生模型参数 |
| rotation_base_from_model | 模型到参考系的旋转矩阵 |
| translation_base_m | 参考系平移 |
| model_scale | 额外整体尺度 |
| mano_convention | WiLoR 右手规范标识 |

WiLoR 参数：`global_orient`、`hand_pose`、`betas`，旋转以矩阵保存。物理左手在 MANO 前向后执行一次 X 镜像，再应用共同尺度、旋转和平移。已保存的顶点和关节包含全部变换，可直接使用。

网格拓扑只需保存有有效模型的手侧；无手序列允许空拓扑和全部空帧。

## 二维结果（E）

`observations_2d/cameraN.jsonl` 每同步时刻一行，含 `sync_index`、`source_frame_index`、`timestamp_ns`、`hands[]`。每个假设含 `detection_index`、`hypothesis_index`、`bbox_xyxy`、`confidence`、`detector_is_right`、`is_right`、`joints_2d: [21,2]`。

同一检测框具有左右手两种姿态假设，不代表两只物理手。检测器类别与假设手侧分别保存。二维姿态由 WiLoR 推理得到。

`projections_2d.jsonl` 含 `sync_index`、camera0 `timestamp_ns`、`cameras.cameraN[]`。每只共享手保存 `side`、`shared_model_id`（`同步索引:手侧`）、`joints_2d: [21,2]`。无效投影点为 `null`；投影有效但超出图像的点保留真实坐标。无共享手时各相机为 `[]`。

## 标定和配置（G、H）

每份 `calibration/cameraN.json` 包含 `schema_version`、`camera_id`、`frame_id`、`model`、`image_size: [宽,高]`、`K: [3,3]`、`distortion`、`T_base_camera: [4,4]`。KB 畸变为 `[k1,k2,k3,k4]`。`T_base_camera` 把相机光学坐标变换到共同参考系，平移单位米。Orbbec YAML 的毫米平移及 reference→camera 变换已在加载时转换。外部 JSON 必须显式提供每路真实外参。

`timestamps/cameraN.csv` 保存源流帧号与时间戳。MP4 保存整路 PTS，MCAP 保存本次解包帧的时间戳和消息编号。`multiview_frames.csv` 保存实际处理的同步索引、各路帧号、时间戳、相对 camera0 的 `delta_us`。

`manifest.json` 是源文件哈希、同步与标定的溯源快照，其源视频存储描述属于临时归一化阶段，最终目录不含归一化视频。`source_meta.json` 在源录制提供时保存。

`run_config.json` 保存参数及输入、标定、模型、代码的路径、大小和修改时间，供重跑一致性判断。`video.json` 保存帧数、帧率、分辨率、编码和渲染耗时。`summary.json` 保存完成状态、基本计数、耗时及交付文件 SHA256；不含质量误差和拒绝原因。
