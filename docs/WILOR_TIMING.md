# 每帧与 WiLoR 前向平均耗时

独立入口：`scripts/profile_multiview_wilor.py`。原实验脚本、原推理代码和第三方
WiLoR 源码均不需要修改。

## 运行

在项目根目录执行：

```bash
python scripts/profile_multiview_wilor.py \
  --output output/wilor_timing_1009_simple \
  --device cuda --max-frames 60
```

默认使用：

- 录制：`recordings/20261009/EGO_Pro_AZLR86000F7_20261009_171149`
- 检测模型：`models/wilor/detector_1009.pt`
- conda 环境：`wilor`
- 关闭 backbone 编译，以便分别测量 ViT 和 MANO 参数解码器。

支持原实验入口的参数，例如 `--session`、`--detector`、`--conda-env`、
`--batch-size`、`--gpu-profile`、`--no-video`、`--stop-after-inference`。
每次运行使用新输出目录；省略 `--output` 会自动生成带时间戳的目录。

## 只看两张表

打开 `OUTPUT/timing/report.md`。

第一张表展示各环节的平均耗时：图像解码 → YOLO 手部检测 → 裁剪与预处理 →
数据传输 → WiLoR 前向 → 后处理 → 结果写出，以及推理流程的实测合计。
录制同步、融合、视频渲染也显示在同一张表，并明确标出统计单位。

- **单路图像**：一台相机的一张图。每帧平均 = 环节累计耗时 ÷ 所有相机图像总数，
  没有检测到手的图像也计入。例如 4 路相机各 60 帧，分母为 240。
- **同步帧组**：同一时刻所有相机的图像。同步、融合、渲染属于多视角操作，
  按同步帧组摊销；上述例子的分母为 60。这些摊销值含进程启动和文件读写。
- 推理流程合计取模型加载完成后处理全部图像的实测时间，再除以图像数。
  解码可能与推理重叠，因此不应把每个环节直接相加作为实际总耗时。

第二张表只保留 WiLoR 的主要模块，按执行顺序排列：

| 主要模块 | 测量范围 |
| --- | --- |
| ViT 骨干网络 | 完整特征提取 |
| MANO 参数解码器 | 从特征解码手部姿态、形状和相机参数 |
| 初始 MANO 网格计算 | 生成精修前的手部网格 |
| RefineNet | 完整精修网络 |
| 最终 MANO 网格与关节点计算 | 生成最终手部结果 |
| 其他张量与相机运算 | forward 总耗时减去以上模块，含计时开销 |
| **一次 WiLoR 前向总耗时** | **完整模型的一次 forward** |

平均每次前向 = 完整 forward 累计耗时 ÷ 成功的 forward 次数。
所有模块使用同一批成功 forward 作为分母，包含首次调用。
一次 forward 是一个批次；报告注明平均每次处理多少个手部裁剪。
**一次前向的均值与分摊到每张图像的均值是两个不同的量。**

CUDA 设备使用 CUDA Event 测量网络模块；CPU 使用墙钟时间。
逐帧流程区间包含设备同步等待。计时本身会产生少量开销。
模型初始化不计入逐帧推理，但计入报告末尾的程序总时间。

显式使用 `--compile-backbone 1` 时，ViT 和参数解码显示为一个编译整体，
其余主要模块仍分别统计。

## 输出文件

- `report.md`：两张中文平均耗时表，直接阅读这份文件即可。
- `frame_average.csv`：各环节的每帧平均耗时。
- `wilor_forward_average.csv`：主要模块及完整 forward 的每次平均耗时。
- `report.json`：相同的简化统计，便于程序读取。
- `samples.csv`、`components.json`、`stages.jsonl`、`run.log`：排查问题用的原始记录。
  首次调用、分位数、预热后统计等细节不在主报告中展开。
  `--warmup-batches` 仅影响原始明细的预热后统计，不改变主报告的全样本平均值。

## 把已有记录转换成简化报告

无需重新推理：

```bash
python scripts/profile_multiview_wilor.py \
  --summarize output/wilor_timing_1009_full_cpu_smoke/timing
```

只更新统计报告，不改动原始预测或计时样本。

CPU 短验证与测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/profile_multiview_wilor.py \
  --output output/wilor_timing_simple_cpu_smoke \
  --device cpu --max-frames 1 --no-video --stop-after-inference

conda run -n wilor python -m unittest discover -s tests -p test_wilor_timing.py -v
```
