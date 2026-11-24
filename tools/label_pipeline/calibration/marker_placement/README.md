# 手套 marker 贴点确认

在数据所在电脑的桌面终端运行：

```bash
xdg-open tools/label_pipeline/calibration/marker_placement/index.html
```

也可以用浏览器直接打开同目录的 `index.html`，无需安装依赖或联网。

1. 选择左手或右手，逐个选择 NOKOV marker 名称。
2. 选择手背或手心，在示意图上点击贴点位置。示意图不准确时，载入对应视角的手套照片再点选。照片必须未镜像。
3. 选择实际贴点区域和表面；不要把 Finger1～4 默认理解成关节编号。不清楚的点可以标记“不确定”。
4. 两手分别完成后点击“保存贴点配置 JSON”，得到 `glove_marker_placement.json`。允许导出未完成配置，文件会列出缺失项。
5. 之后用“载入配置继续编辑”恢复；照片需重新选择同名、同尺寸的原照片。不要用同名同尺寸的其他照片替换。

浏览器会暂存配置，但最终以下载的 JSON 为准。照片不上传、不嵌入配置。坐标是对应视图或照片内的归一化二维位置，不是毫米，也不是三维附着点。

配置用于确认后续 NOKOV 三维轨迹拟合的解剖区域约束，目前不会被现有标签导出命令自动读取。球直径记录为 4 mm；本工具不执行 2 mm 偏移扣除，不修改既有标签或拟合结果。

## 原图与 mesh 标签检查

启动两个工具：

```bash
bash tools/label_pipeline/calibration/marker_placement/start_viewer.sh
```

浏览器打开 http://127.0.0.1:8766/ 检查 mesh，打开 http://127.0.0.1:8766/marker_placement/ 确认 marker 贴点。远程访问时，在本地电脑执行 `ssh -L 8766:127.0.0.1:8766 p3@服务器地址`，再在本地浏览器打开上述地址。

当前查看器读取基线 `fourview_192916/native_labels/export/merged` 的 488 张原图、976 份标签。它不重新拟合，也不修改标签。切换相机时选择最接近当前 sync 的已导出帧，务必查看屏幕上的 sync 编号；不同相机并非每个 sync 都有标签。

- 左侧原图，右侧为同一原图上投影的 mesh；支持网格、半透明面、顶点和可选关节点。
- 滚轮缩放、拖动平移；“放大左手 / 右手”定位局部；左右箭头逐个切换已导出的帧；按住空格临时隐藏叠加。
- 可以记录问题类型和备注，导出 `mesh_label_review.json`；保存截图会保存当前叠加视窗。
- mesh 使用导出标签中相机坐标下的 `vertices + trans`，以各自的真实 `K` 和 KB `distortion` 投影；不额外旋转外参、不去畸变、不翻转原图。
- mesh 线框包括背面边；半透明面采用深度排序，不具备真实手部遮挡检测。照片内的贴合只能检查二维投影一致性，不能单独证明三维深度、关节中心或皮肤表面准确；手套表面与 MANO 裸手表面也可能存在厚度差异。
- 数据不插帧、不补标签。图像通过本地符号链接引用，源数据不能移动或删除。

为其他已导出的原图 KB 标签生成同样的查看器：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python \
  tools/label_pipeline/calibration/marker_placement/build_label_viewer.py \
  --dataset /你的导出目录/merged \
  --output /你的查看器输出目录

bash tools/label_pipeline/calibration/marker_placement/start_viewer.sh /你的查看器输出目录
```

## 可选：真实手套照片与 24 点预填

预填照片、建议 JSON 和 `prefill.js` 是本地录制的生成数据，由 Git 忽略，本机已有文件会继续使用。
新检出仓库可以直接使用示意图、上传照片及配置导入／导出；预填数据生成后，打开贴点工具即可看到左右手各 24 个 NOKOV 球心投影建议。
左手来自 camera1 / sync 616，右手来自 camera2 / sync 584，照片是现有标签原图的局部裁剪，没有去畸变或镜像。

- 橙色代表待确认，粉色圈为当前点，绿色代表已确认；编号采用 NOKOV 原始 0～23，与 MANO 21 点编号不同。
- 先选点名，再点击照片微调，检查区域与表面，然后点击“确认本点位置与区域”。支持“只显示当前点”和“还原本点建议”。
- 初始区域是按名称给出的粗略建议，不代表已完成解剖标定。遮挡点也可能有投影，不能把投影位置当作可见 marker；看不清时请选择“不确定”。
- 已有手工记录不被自动覆盖；“使用预填原图”可恢复内置参考照片。导出文件单独列出待确认点与缺失点。
- 初始建议由原始 24 点 NOKOV 轨迹、已有时间同步及空间标定、真实 KB 相机模型生成，未使用 MANO 关节点代替 marker。原始照片路径、裁剪范围、投影像素、marker 名称都保存在建议配置中。
- 这次只生成贴点辅助数据，不修改现有标签，不执行 marker 半径偏移扣除。

预填数据与来源：`prefill/suggested_placement.json`、`prefill/report.json`。准备好同级的
`ego_hand_system`、`nokov_ego_hand_validation`，以及脚本引用的 2026-11-11 录制和对齐结果后，
在训练环境中运行 `python tools/label_pipeline/calibration/marker_placement/prepare_marker_prefill.py` 重建本次录制的预填数据。

## 播放、放大与全屏

查看器新增播放／暂停（快捷键 P）、采样播放速度、循环和逐帧保持缩放。每个画面标题下都有“单独放大”“全屏”和播放按钮；全屏用 Esc 退出，单独放大用“恢复并排”返回。支持＋／－按钮和鼠标滚轮缩放，拖动平移。播放针对当前已加载的采样帧，不代表原始连续录像。

更新后浏览器按 Ctrl+Shift+R 刷新。播放与全屏控件直接保存在共享源码模板
`label_viewer.html` 中，之后新生成页面会自动包含这些功能。已有输出页面可以用对应的构建命令重新生成。
