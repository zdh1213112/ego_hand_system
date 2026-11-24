# NOKOV 26 段骨架：原图与模型语义快速核查

打开 http://127.0.0.1:8766/nokov_skeleton/review/ 。可切换左右手与四路相机、放大／全屏、播放预选样本，独立开关 SDK 骨架、marker、当前 MANO 和节点编号。

颜色：紫色为原始 SDK 26 段骨架；绿色圆圈为真实 24 个 marker；青色／橙色为当前左右手 MANO 关节点。未对 SDK 骨架重新对齐或拟合，也没有修改原有标定或训练标签。

## 当前结果

结果目录：

```text
output/native_kb_training_20261111/nokov_skeleton_audit_v1
```

完成原始 6077 个 NOKOV 帧的数值检查，并生成 8 个代表时间点 × 4 相机共 32 张原图核查。检查覆盖整段录制的骨架数据；图像核对是代表帧，不代表已逐帧确认所有动作。

| 项目 | 左手 | 右手 | 含义 |
| --- | ---: | ---: | --- |
| 每帧骨架段 | 26 | 26 | 包含根、Finger0、各指骨段原点与 End |
| 骨架 Hand 根到 WristM 距离中位数 | 25.70 mm | 21.19 mm | 二者不是同一个定义的点 |
| 按名称选出的候选21点到 marker21 距离中位数 | 2.59 mm | 2.73 mm | 是模型与表面 marker 的对应差异，不是真实内部关节误差 |
| 最大全段骨长标准差 | 0.0000173 mm | 0.0000172 mm | 反映固定骨长约束，不能当作测量精度 |
| 五个 Finger0 原点最大分离 | 0 mm | 0 mm | 全部共享同一位置，不能映射为五个独立 MCP |

SDK 旋转解释为世界旋转时，父骨骼旋转下的子节点局部偏移在数值精度内恒定；结合其位置与同时采集 marker 的对应关系，支持当前“全局位置／旋转”解释。但采集元数据没有明确记录 XINGYING 的 global/local 开关，应结合原工程最终确认。

代表帧中，骨架整体跟随张手、弯指、对指姿态；SDK 末端和手指节点往往接近表面 marker，因此不应自动把这些点当成独立测量的真实内部关节。

**判断：骨架具备作为动作／固定骨长先验的候选价值，但尚不能直接拿26点或按名称截出的21点替换MANO关节真值。** 下一步需核对 NOKOV 工程或手部模板中的节点定义、静止坐标轴和 marker 附着偏移，再建立到 MANO 的映射。优质轮廓继续约束个人手型与表面。

## 坐标链路与输入

1. 骨架来自本次真实 `nokov_skeleton_segments.csv`，以米保存位置；父子关系读取 `asset_descriptions.json`。
2. 使用已验收链路的时间映射，位置做相邻帧插值，头部刚体姿态使用 Slerp，复用原有 `T_cam0_head`。
3. SDK 世界位置 → 头部刚体局部 → camera0 → 各路真实相机；最后使用真实 K 和 KB 畸变投影到原图。
4. marker 使用同一坐标链，并与既有 `aligned_nokov21.npz` 逐点一致性核验。物理左手在本链路不做 MANO 规范化镜像。
5. `aligned_skeleton.npz` 中的 SDK 四元数保留世界坐标语义，没有静默改成相机坐标或 MANO 局部关节旋转。

当前 MANO 对照来自 `full_silhouette_v1/shape_only`。三栏静态图为原图／SDK＋marker／SDK＋MANO；只是骨架位置比较，没有重新拟合 MANO mesh。节点编号、名称和 End 标记均未自动认证为解剖学关节。

## 一个命令复现

```bash
bash label_export/pipeline/calibration/nokov_skeleton_audit/run.sh
```

默认使用当前录制的既有 context、原图审核缓存和当前 MANO 对照；创建新目录，不覆盖已有结果。`--output 新目录` 可以指定保存位置。命令不会重新采集、训练或修改默认标签流程。

原来的 8766 服务未启动时：

```bash
/home/p3/miniforge3/envs/ego-hand/bin/python -m http.server 8774 \
  --bind 127.0.0.1 \
  --directory output/native_kb_training_20261111/nokov_skeleton_audit_v1
```

打开 http://127.0.0.1:8774/review/ 。数值报告为 `report.json`，接入判断为本次目录中的 `decision.json`。
