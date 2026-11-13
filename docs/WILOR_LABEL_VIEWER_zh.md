# WiLoR 原图标签数据集查看器

复用项目已有的 `wilor_training/tools/marker_placement/build_label_viewer.py` 和交互页面，新增通用入口 `ego_hand_system/scripts/view_wilor_labels.sh`。读取的是实际导出的 `.npy` 标签，不从中间拟合文件重新生成标签。

## 查看当前最新优化标签

```bash
cd /home/p3/data_sda1/EGO_glove/ego_hand_system

bash scripts/view_wilor_labels.sh \
  --dataset /home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/explicit_pipeline_sampled_smoke_v2/export/merged \
  --sync 632
```

打开：http://127.0.0.1:8770/ 。保持终端运行，Ctrl+C 退出。已经启动时直接打开链接，不必重复启动。

## 加载其他导出的标签数据集

只替换 `--dataset`，指向含 `index.jsonl`、`images/`、`labels/` 的数据集根目录；不要指向 `labels/` 子目录。本入口目前支持当前路线的 `native_raw / KB` 标签。

```bash
bash scripts/view_wilor_labels.sh --dataset /你的数据集根目录
```

可选参数：

| 参数 | 作用 |
|---|---|
| `--sync 632` | 初始定位到最接近的有效同步帧；不提供则从第一帧开始 |
| `--port 8771` | 修改 HTTP 端口，默认 8770 |
| `--output /新的查看器缓存目录` | 指定 HTML、投影数据和原图链接位置；默认在数据集旁边生成 `<数据集名>_viewer/` |
| `--build-only` | 只生成查看器文件，不启动服务器 |

缓存目录如果已属于另一套数据集，会明确报错，需指定新目录。图像使用链接，原数据集需要保留在原路径。

## 交互功能

- 左侧真实原图，右侧同一原图叠加实际标签，缩放和平移同步。
- 切换录制和相机，上一帧／下一帧、时间滑块。
- 左右手分别显示，三角网格、半透明面、顶点或隐藏 mesh；可调透明度及显示 21 关节点。
- “放大左手／放大右手”聚焦对应区域；滚轮缩放，拖动平移。
- 每个面板可单独放大、全屏；播放支持速度选择、循环和保持缩放。
- 按住空格暂时隐藏叠加；P 键播放／暂停。
- 保存叠加截图，查看当前标签文件路径、K、畸变和索引信息；可导出个人检查记录。

仅显示数据集中真实存在的已导出标签。切相机时定位到最接近的有效帧，界面显示实际 sync；没有导出的帧不补插。播放是采样帧播放，不是原始连续视频速度。

网格线包含背面三角形，透明面按深度排序；真实手和其他物体对 mesh 的遮挡不会自动推断。该查看器适合核对投影和贴合，不等同于测量真值。

## 校验与本次结果

构建时按真实 KB 参数重新投影 `vertices + trans`，检查与标签中 778 个顶点投影一致，然后生成页面；21 关节投影从 `joints_3d + trans` 单独计算。

本次加载 483 张原图、966 份标签，四路相机均可查看；最大顶点投影差约 0.0000883 px。浏览器检查通过实际数据加载、初始 632 帧、左右手、叠加切换、切帧、camera3 图像、缩放、单独放大和播放／暂停。

查看器只读取标签；截图、缓存和个人检查记录不修改 `.npy` 标签。
