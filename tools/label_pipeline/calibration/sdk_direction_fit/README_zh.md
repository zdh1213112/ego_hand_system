# SDK 方向辅助与 Finger0 腕根候选

本实验复用 `pair_20261111_192916` 的当前共享手型、固定 marker 附着和已审核分割。用户选择的 **SDK Finger0 公共根**保留为优先腕根候选；原来的两种腕根对照页已默认选择 Finger0。它仍不是经过解剖标定确认的 MANO 腕关节位置。

## 执行

```bash
bash tools/label_pipeline/calibration/sdk_direction_fit/run.sh
```

默认生成新目录，不覆盖历史输出。可用 `--output /新目录` 指定结果，`--help` 查看 context、source、skeleton、selection 等输入。该入口依赖本机已有标定产物，并非新录制从零导出的通用入口。

本轮结果：

```text
output/native_kb_training_20261111/sdk_direction_v1
```

- [原图／当前标签／新对照](http://127.0.0.1:8766/sdk_direction/review/)
- [四路方向辅助近景视频](http://127.0.0.1:8766/sdk_direction/closeups_direction/)
- [仅 marker 重拟合的四路视频](http://127.0.0.1:8766/sdk_direction/closeups_marker_only/)
- [方向辅助实际导出标签](http://127.0.0.1:8766/sdk_direction/direction/export/viewer/)
- [原来的 Finger0／Hand 腕根对照](http://127.0.0.1:8766/sdk_skeleton_labels/review/)

若未运行现有 8766 服务：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python -m http.server 8776 --bind 127.0.0.1 \
  --directory output/native_kb_training_20261111/sdk_direction_v1
```

打开 http://127.0.0.1:8776/review/ 。此独立服务中的“原来的腕根对照”链接需要原有 8766 服务，其他本轮产物使用相对路径。

## 本次做了什么

1. 从 6077 个原始 SDK 时刻恢复父子骨骼的局部固定偏移，估计 24 个 marker 相对候选骨段的局部偏移与残差，写入 `profile/mapping.json`。**这是录制数据推得的关系，不是读取到了原始 NOKOV 模板的内部关节偏移。**
2. 先用头部刚体和已接受外参，把 SDK 世界旋转转到真实 camera0，再处理左手 canonical 镜像。Finger0 提供参考原点，SDK Hand 提供刚性掌部坐标方向；不能把可运动的 Thumb0 方向直接当作手掌方向。
3. 只在每手 16 个优质关键时刻估计 SDK 到 MANO 的固定方向校正，约束每指的三条节段方向，共 15 条。排除 Hand／Finger0 到 MCP 的位置对应，不采用 SDK 骨长。
4. 同样从当前共享手型组起步，运行两组各手 194 个实际采样时刻：`marker_only` 只重新拟合 marker；`direction` 额外加入较弱的 SDK 方向和掌部旋转辅助。两组共享完全相同的自由度和正则。
5. 固定每手 betas、表面附着面片／重心坐标／有效法向偏移、腕部 marker 局部关系和逐帧根平移。只开放动作小幅变化：每个手指旋转向量增量范数 ≤0.10 rad（约5.73°），整体旋转向量增量范数 ≤0.035 rad（约2.01°）。没有额外减去2mm。
6. 用同一组可靠原图分割评分，通过原生 KB 导出器输出标签，执行 MANO 参数重建和逐份原图投影检查，生成可播放四目对照。

**本次没有按 Finger0 的绝对位置硬拉 MANO 腕关节，也没有再次假定 SDK 节点就是已补偿的内部关节。** `profile/mapping.json` 中 Finger0 到 MANO 腕的局部偏移仅供诊断，不参与拟合位置目标。新方向实验使用当前共享手型组初始化，与原来的 SDK Finger0 位置拟合是两种实验，查看器中有各自入口。

## 本轮结果

190 份未参与本次方向映射及此前手型标定的可靠视图，其中左手64份、右手126份。它们的 RGB 分割没有参与本次动作拟合，但对应 SDK／marker 数据仍用于动作拟合，因此不是独立动作真值验证。

| 手 | 当前标签 IoU | 仅 marker 重拟合 | marker＋SDK方向 | SDK方向相对 marker 对照 | 新结果相对当前标签 |
|---|---:|---:|---:|---:|---:|
| 左 | 79.74% | 78.82% | 79.21% | +0.39 个百分点 | −0.53 个百分点 |
| 右 | 81.77% | 82.13% | 82.20% | +0.07 个百分点 | +0.43 个百分点 |

说明 SDK 方向辅助在这一受限实验中带来小幅增益，但不能据此替换当前路线。降低 SDK 拟合残差不等于 mesh 更贴合；手型、附着、解剖语义尚未得到独立确认。当前输出继续为 `approved_for_training=false`。

## 输出

| 位置 | 内容 |
|---|---|
| `profile/mapping.json` | SDK 局部骨长／偏移、候选 marker 附着、15条方向映射与可信度 |
| `profile/direction_targets.npz` | 掌部局部单位方向、可信度、Finger0参考根 |
| `{marker_only,direction}/*_pilot.npz` | 左右手共享 betas、逐帧姿态／mesh、固定附着与 marker 残差 |
| `{marker_only,direction}/export/merged/` | 每组505张原图、1010份双手标签；严格原生 WiLoR 标签格式 |
| `{marker_only,direction}/export/logs/check.log` | 原生标签和 MANO 重建检查 |
| `review/` | 四目交互对照，支持缩放／全屏／采样播放 |
| `closeups_{direction,marker_only}/camera*_comparison.mp4` | 每组4路近景视频，各158个同步采样时刻、3.75帧/秒 |
| `comparison_metrics.json` | 固定腕部评分区域的轮廓 IoU、边界距离和逐图比较 |
| `verification.json` | 镜像／坐标不变性、冻结参数、增量上限、MANO重建和双手导出分组验证 |

紫色方向箭头锚在 MANO 关节、采用 MANO 骨长，仅用于展示方向，不代表 SDK 内部关节位置；紫色圆圈是 SDK Finger0 参考根。实际标签只包含通过完整性与重投影筛选的图像，交互拟合页还包括未通过导出的视图。
