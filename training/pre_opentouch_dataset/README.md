# OpenTouch HDF5 解码

这个目录只负责把 OpenTouch HDF5 解码成可随机访问的 RGB 图像、逐 clip 数值数组和索引；它不会把 Rokoko 关键点投影到 RGB，也不会假设关键点已经处于相机坐标系。

## 输出结构

```text
opentouch_decoded/
├── decode_config.json
├── summary.json
├── frames.jsonl
├── clips.jsonl
├── calibration/
│   └── <source>.json
├── images/
│   └── <source>/<clip>/<source_frame_index>.jpg
└── clips/
    └── <source>/<clip>.npz
```

JPEG 直接复制 HDF5 中的原始字节，不重新编码。每个 NPZ 保存所选帧的：

- `source_frame_indices`
- `timestamps`
- `right_hand_landmarks`
- `right_wrist_pos`
- `right_palm_pos`
- `right_pressure`
- `camera_poses`

`frames.jsonl` 将每张图像映射到对应 NPZ 和 `clip_data_index`；路径均相对于输出根目录。

## 环境

当前机器的 `xense-taccap` 环境已经包含 `h5py`，可直接使用：

```bash
conda run --no-capture-output -n xense-taccap \
  python pre_opentouch_dataset/decode_opentouch.py --check-only
```

也可以在其他 Python 环境安装本目录的依赖：

```bash
python -m pip install -r pre_opentouch_dataset/requirements.txt
```

## 冒烟解码

先解码一个文件、一个 clip、两帧：

```bash
conda run --no-capture-output -n xense-taccap \
  python pre_opentouch_dataset/decode_opentouch.py \
  --max-files 1 \
  --max-clips-per-file 1 \
  --max-frames-per-clip 2 \
  --output-dir /tmp/opentouch_decode_smoke
```

## 完整解码

```bash
conda run --no-capture-output -n xense-taccap \
  python pre_opentouch_dataset/decode_opentouch.py \
  --input-dir datasets/opentouch_raw \
  --output-dir datasets/opentouch_decoded
```

默认每帧都解码；当前 327,030 帧的完整输出实测约 21 GiB，建议至少预留 25 GiB。中途中断后，使用完全相同的参数加 `--resume`；脚本会复用大小匹配的 JPEG 和结构完整的 NPZ，并重新生成确定性的 manifest。

如果希望减少数据量，可以设置 `--frame-step 2` 等间隔采样。`--compress-arrays` 会减小 NPZ，但完整数据解码会明显变慢。

## 后续 bbox 与训练数据准备

解码完成后，用仓库根目录的统一准备入口生成待标注清单：

```bash
conda run -n wilor python prepare_glove_dataset.py \
  --data-type opentouch \
  --decoded-root datasets/opentouch_decoded \
  --bbox-annotations datasets/opentouch_bbox_annotations.jsonl \
  --init-empty-bboxes
```

只给“完整手在图内”的有用样本填写 `bbox_xyxy`，无用样本保持 `null`。合并键固定为
`(source_file, clip_id, source_frame_index)`；不能使用会在多段视频中重复的
`sync_index`。完整标注和训练命令见根目录 `GLOVE_TRAINING.md`。
