# 多视角 FAH 手部姿态与网格重建

## 数据与输出

输入仍为原来的 MCAP 或 EGO4 多相机 MP4 会话，无需转换为图片。
沿用原脚本的视频读取、同步帧号、纳秒时间戳、相机顺序和 MP4 编码/拼接布局。
对于标定标记为 KB 的原始视频，检测器在原图找手。每个检测框按原相机 K 和四个畸变系数生成手部透视小图，
与训练使用相同的留边、内参和左右手镜像。FAH 在小图内预测关节并拟合 MANO，随后将二维关节映回鱼眼像素，
把三维关节、腕部、网格和 MANO 根朝向转回真实相机坐标。网格叠加读取归档中的 KB 标定。
标定为 PINHOLE 的视频沿用透视输入流程，DS 输入沿用原针孔近似。

默认从同级 `wilor_training` 加载共享裁剪代码，两个仓库分开放置时设置 `WILOR_TRAINING_ROOT`。
每个机位的 `summary.json` 会记录 `camera_model_for_pose`，原始 KB 视频应为 `KB_local_perspective`。

```text
OUTPUT/
  run_config.json
  normalized_multiview/               与原流程相同的数据
  fah_multiview/
    summary.json
    camera*/predictions.jsonl         原流程字段与 FAH MANO 拟合参数
    camera*/meshes.npz                真实相机网格、面片、原相机 K 和 KB 标定（若适用）
  fusion_multiview_shared_fah/
    shared_pose.pt
    final_shared_Nview.mp4            shared/both + --final-video
  fusion_multiview/
    final_only_Nview.mp4              legacy/both + --final-video
    diagnostic_Nview.mp4              legacy/both，沿用原开关
```

网格为 778 个 MANO 顶点，3D 单位为米，坐标为相机系 x 向右、y 向下、z 向前；
21 个关节点沿用 wrist、thumb、index、middle、ring、pinky 的原有顺序。
左右手网格各自保留正确的三角面绕序。
共享拟合使用 FAH 拟合出的旋转与形状作为初值，最终网格与骨架来自同一个 MANO 模型。
原生网格单独保存实际 FAH 像素对齐后的顶点，避免用 WiLoR 的焦距或网格参数错误重放。
