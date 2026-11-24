# 手套标签导出内部实现

对外命令为 `bash scripts/export_ego_nokov_labels.sh --recording <录制> --optimization-config configs/native_nokov_v4.json`，在仓库根目录运行。[使用者命令和结果路径](../../docs/EGO_NOKOV_NATIVE_LABEL_COMMANDS_zh.md)

以下说明供维护和回归使用。`export_glove_labels.py` 实现签收拟合、五段冻结重发；`run_recordings.py` 负责历史批处理。二者均由维护流程调用。九段的进度汇总 `recordings/report.py` 只服务于既有批次的工作记录，与公共导出入口独立。

1. 填录制 ID、独立输出目录、四机 MCAP 与同期 NOKOV 所在目录、相机标定、空间初值和同曝光 WiLoR 观测。模板已填共用手套定义及 MANO 路径。WiLoR 观测须能按原曝光帧号索引四机及实际快动作片段，多份归档可填入 wilor_observations。新录制缺少观测时先确定原始路径和推理许可。原生求解器使用项目安装的 MANO_RIGHT.pkl，入口会核对配置文件内容与该模型一致。固定先验与来源摘要在 assets/。原生标定与导出沿用同级 NOKOV 解析工具、本仓库 calibration 和 training 目录以及已安装的 nvdiffrast 原生光栅器，其目录列在 defaults.yaml。
2. 新录制在 step06 按每只手的 NOKOV 三个同名腕带球、四机亮点和指尖位移自动选择外旋参照，并继续拟合。可在 `review/WRIST_REVIEW.md` 看参照、伸展和卷曲时的画面。候选帧及测量值在 `review/adaptive_wrist/selection.json`。若扩大采样后仍缺观测，程序在同目录写出 `unusable.json`，说明涉及哪只手及未通过的条件。历史重发使用保存的参照。
3. step12 后打开 `review/DATASET_REVIEW.md` 查看最终训练标签的四机图、camera0 整段视频和训练读取坐标核查。白圈标记训练腕点 MANO J0，黄色菱形标记外旋参照。最终数据签收对应 `dataset_targets.json` 的 SHA。用同一条入口命令加 `--publish` 生成已签收的独立发布目录。

重发五段时指定 `--config label_export/pipeline/configs/signed_5groups.yaml`。从冻结轨迹重建逐曝光标签后，入口检查全部字段、原图 SHA 和去除 PyTorch 存储编号后的严格及扩展 NPY 摘要。每段位于 legacy/录制名，五段合并结果位于 dataset/。输入和算法参数的修改记录在运行记录中。

已有非空输出用 `--resume` 续跑。可指定 `--from step09 --to step12`，上游记录与产物须能通过摘要核验。每步的运行记录在 legacy/录制名/records/stepXX/run_record.json，原生 GPU 日志在 records/legacy_seed，后续拟合在 records/legacy_solver 和 records/subprocess。005 在本机的首次实测为转存及对齐约 4 分钟，GPU 基线 9 分钟、步长 4 的原生拟合与导出 38 分钟。手形、腕点、解剖、挂点及后续检查另计，耗时随录制长度变化。五段冻结重发和训练读取使用 CPU。
