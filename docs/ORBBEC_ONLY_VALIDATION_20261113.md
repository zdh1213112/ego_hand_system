# Orbbec 专用入口清理验证（2026-11-13）

本次以 `1f8155a` 为基线，移除已停用的 GEN 相机入口，保留 Orbbec EGO4 / EGO-4B 工作流。

## 调整范围

- 删除 GEN MCAP 解码器、双目标准化入口、环境诊断脚本、配置样例及五份 GEN 六目操作文档。
- 双目 `run_offline.sh` 默认使用 Orbbec；MediaPipe、WiLoR、parallel、校正、稳定化、MANO 和渲染阶段继续保留。
- `run_multiview_wilor_experiment.sh` 继续提供 Orbbec 四目 MP4 融合，并将 MCAP 分支接到 Orbbec 读取器；默认 camera0..3、参考 camera0、锚点 camera0/camera1。
- 从旧 GEN 模块提取通用 H.264 写入工具到 `scripts/ego_data/h264.py`，原 Orbbec MCAP 读取器仅更换导入位置。
- 保留共享推理、原始四目 WiLoR、NOKOV 标签、实时、VIO、SAM3/亮点辅助及通用几何。输出中的历史算法标识继续保留，避免影响既有数据的读取。
- README 和进阶文档改为 Orbbec 工作流；已有 Orbbec 自动化测试保留，旧 GEN 测试中的通用时间配对、外参和 H.264 检查迁移至 `test_sensor_data.py`。

## 自动检查

完整 Python 回归：**305 passed, 1 skipped**。轻量 CI 所用的六个测试文件：**25 passed**。Ruff 致命错误检查、Shell 语法检查和本地 Markdown 链接检查通过。

补充回归实际执行公共 MCAP 标准化入口，检查四路 H.264 解码、同步、标定保留和缺失相机报错；另检查四目 MP4/MCAP 启动默认值及旧配置中闲置 camera4/camera5 条目对继续运行的兼容。

对拆分前后的 Python 语法树进行比较，确认 `H264MatroskaWriter`、`h264_nal_types` 和通用 `synchronize_rows` 的执行逻辑一致。原始四目网络、MP4 标准化、共享推理主入口、环境依赖及相机采集/VIO 的 C++ 源码保持基线内容。

## 真实数据检查

| 输入 | 检查内容 | 结果 |
| --- | --- | --- |
| `EGO-4B_AZLCA610091_20261113_150156_part0001.mcap` | 公共多目入口，自动发现配套 YAML，最多 5 帧 | 四路各解码 5 帧、同步 5 组、KB 标定保留；最大同步差 780 微秒 |
| `pair_20261111_192916` 的 Orbbec MP4 | 共享推理、MANO 导出、四目渲染，最多 8 帧 | 8 帧、16 个手模型，1920×1200 四目视频及结果完整性检查通过 |
| 同一份 EGO-4B MCAP | 共享推理完整流程，最多 8 帧 | 8 帧四目视频及结果完整性检查通过；开头片段未接受手模型，不作为三维手部质量验收 |

使用既有手套检测器和授权 MANO，GPU 配置为 `rtx5090d`、假设批大小 16、帧批大小 8。原图保持鱼眼像素空间。

测试关闭宿主 ROS 的 pytest 自动插件加载，使用项目完整 Python 环境。此次没有接入实时相机验证 SDK/采集，也没有重新安装外部 SAM3 运行时；这些链路保留原实现，回归结果不代表重新完成硬件验收。

本机原始日志及数据产物位于 `work/ego_hand_orbbec_only_20261113_x2q95k2c/`，不随仓库提交。此前的长片、固定输入数值比较和模型约定见[融合验证记录](INTEGRATION_VALIDATION_20261113.md)。
