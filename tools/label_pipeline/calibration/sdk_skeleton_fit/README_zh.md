# 假定 SDK 已补偿表面到内部偏移：MANO 标签对照

## 先看结果

- [交互对照：原图／当前标签／SDK 假设标签](http://127.0.0.1:8766/sdk_skeleton_labels/review/)
- [Hand 腕根：四路双手近景视频](http://127.0.0.1:8766/sdk_skeleton_labels/closeups_hand/)
- [Finger0 公共腕根：四路双手近景视频](http://127.0.0.1:8766/sdk_skeleton_labels/closeups_finger0/)
- [实际导出的 Hand 腕根标签](http://127.0.0.1:8766/sdk_skeleton_labels/sdk_hand/export/viewer/)
- [实际导出的 Finger0 腕根标签](http://127.0.0.1:8766/sdk_skeleton_labels/sdk_finger0/export/viewer/)

上排左手、下排右手；视频每排为原图 → 当前标签 → SDK 骨架假设拟合标签。交互页可切换两种腕根、放大、播放，或开启紫色 SDK21 目标。绿色为已有可靠分割；分割不参与本轮动作拟合。

本次按用户明确指定的假设执行：**NOKOV 模型已经配置 marker 表面到内部关节的偏移，SDK 输出的骨骼原点可作为内部关节候选。** 这个假设没有得到模板文件核实，也没有因此修改 NOKOV 工程。没有再次减去球半径或增加统一偏移。

## 方法与两种映射

1. 固定当前 `full_silhouette_v1/shape_only` 左右手各自的一套 betas，以及既有 marker 附着关系。
2. 复用已检查的同步、头部刚体与真实相机外参，将 SDK 世界坐标转换到 camera0；物理左手只在送入 MANO 时转换为右手规范坐标。
3. 各指 SDK `*1、*2、*3、*End` 依次作为对应 MANO 四点的候选目标。没有把重复的五个 `Finger0` 当作五个独立 MCP。
4. 两组唯一的目标区别是 MANO 腕根：`Hand`，或五个重合 `Finger0` 的公共位置。节点名称与真实内部关节的语义仍属于实验假设。
5. 每手对已有 194 个采样时刻拟合动作，350 步：稳健 SDK21 三维距离、相对初值的姿态先验和相邻采样帧增量平滑。betas 与附着参数始终固定。SDK 四元数没有直接复制到 MANO，因为两套模型的静止关节轴尚未核定。
6. 为实际标签构建单独的 SDK21 融合观测，质量参考使用 SDK 投影，不再拿旧 marker 当作内部关节目标。使用原有原图导出器、双手配对和完整21点筛选，再运行严格 MANO 参数重建及 KB 投影检查。

动作限制：手指各轴角增量 ±0.5 rad；整体旋转增量范数≤0.15 rad；平移增量≤20 mm。本轮没有为降低残差继续扩大限制或调整共享手型。Hand 组部分帧整体旋转触界，不能视为已充分求解。

## 实际导出文件

结果根目录：

```text
/home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/sdk_skeleton_assumed_internal_v1
```

| 方案 | 原图数 | 双手标签数 | 实际数据集相对路径 |
| --- | ---: | ---: | --- |
| SDK Hand 腕根 | 458 | 916 | `sdk_hand/export/merged/` |
| SDK Finger0 公共腕根 | 519 | 1038 | `sdk_finger0/export/merged/` |

每张导出图像各有一份左手和右手 `.npy`。输出格式为现有 WiLoR 原图 KB 标签格式，未添加破坏严格 schema 的字段；实验假设保存在 `summary.json`、逐标签 `index.jsonl` 及 `assumptions.json`。每份索引明确 `sdk_internal_offset_assumed=true`、`sdk_internal_offset_verified=false`、`approved_for_training=false`。

标签数量差异来自完整21点、双手配对和几何筛选，不能用来判断精度优劣。比较页面包含部分未通过实际标签导出条件的视图；实际标签入口只显示已通过格式与几何筛选的样本。

两组全部标签通过格式检查、真实 KB 投影检查和 MANO 参数重建。最大顶点重建误差小于 `1.2e-7 m`，最大投影误差约 `6.1e-5 px`。这些是内部一致性检查，不代表标签贴合真实手套。

## 对照结果：该假设下整体视觉贴合变差

在同一批未参加前期共享手型标定的可靠分割图像上，使用固定的原图腕部区域评分：

| 手 | 图像数 | 当前标签 IoU | SDK Hand 根 | SDK Finger0 公共根 |
| --- | ---: | ---: | ---: | ---: |
| 左手 | 64 | 79.74% | 74.35% | 74.52% |
| 右手 | 126 | 81.77% | 76.62% | 76.42% |

Hand 根方案中，左手 52/64、右手 106/126 份图像的 IoU 下降超过2个百分点；Finger0 根方案为51/64、108/126。分割是图像证据，不是独立解剖学真值，但足以说明此假设＋当前映射没有改善整体表面贴合。

与此同时，SDK 关节拟合残差中位数确实下降：

| 手 | Hand 根：拟合前→后 | Finger0 根：拟合前→后 |
| --- | ---: | ---: |
| 左手 | 8.83→5.27 mm | 8.79→4.72 mm |
| 右手 | 8.29→4.94 mm | 8.03→4.27 mm |

这说明更贴近 SDK 节点不等于更贴近真实手套。不能据此单独判定“SDK 必然没有偏移”，也可能存在节点语义、骨长或静止轴不一致；但当前两套结果不应直接替换现有正式标签。

## 复现命令

```bash
bash /home/p3/data_sda1/EGO_glove/wilor_training/tools/sdk_skeleton_fit/run.sh \
  --assume-sdk-internal-offsets
```

默认使用本机当前 `pair_20261111_192916` 的同步、骨架审核、个人手型和可靠分割，创建新的实验目录。`--output 新目录` 可指定结果位置；`--context`、`--source`、`--skeleton`、`--selection`、`--previous-review` 可覆盖已经准备好的实验输入。这不是原始录制入口的新默认模式。

如果 8766 服务未运行：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python -m http.server 8775 \
  --bind 127.0.0.1 \
  --directory /home/p3/data_sda1/EGO_glove/wilor_training/output/native_kb_training_20261111/sdk_skeleton_assumed_internal_v1
```

打开 http://127.0.0.1:8775/review/ ，视频在 `/closeups_hand/` 与 `/closeups_finger0/`。

## 主要产物

- `sdk_hand/`、`sdk_finger0/`：每手194个采样时刻的 MANO 参数与 mesh，SDK21 目标、对应关系、残差和拟合记录。
- 各组 `mano_fit/`：标准拟合 track；各组 `sdk_observations/`：明确标记假设来源的 SDK21 观测。
- 各组 `export/merged/`：实际原图双手标签；`export/viewer/`：实际标签查看器；`export/logs/check.log`：全量一致性检查。
- `review/`：两种腕根可切换的拟合对照；两个 `closeups_*`：各4路、每路158帧约42秒的近景视频。
- `comparison_metrics.json`：逐图像的视觉比较；`verification.json`：固定参数、SDK目标映射、双手配对和来源标记验证。

当前训练权重、默认标签路线和旧标签均未替换，也没有训练 WiLoR。
