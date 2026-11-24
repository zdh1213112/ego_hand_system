# 辅助工具索引

这些工具用于数据标定、实验比较和标签检查。训练主入口仍是根目录的
`train_glove.py`，核心训练与已有图片／视频／Gradio 推理不会自动导入 `tools/`。
从仓库根目录运行下列脚本，具体输入和命令见各目录说明。

| 目录 | 用途 |
| --- | --- |
| [marker_placement](marker_placement/README.md) | 浏览器确认 marker 贴点，构建原图 / MANO 标签查看器；`label_viewer.html` 为多工具共享模板 |
| [native_marker_audit](native_marker_audit/README_zh.md) | 原始图像上的 marker 投影与检测检查 |
| [nokov_skeleton_audit](nokov_skeleton_audit/README_zh.md) | NOKOV 骨架、比例和多目投影诊断 |
| [marker24_pilot](marker24_pilot/README.md) | 24 marker 表面附着拟合与方向检查 |
| [personal_hand_calibration](personal_hand_calibration/README_zh.md) | 个人手型标定、轮廓约束、几何消融及上下文加载；被其他标定工具复用 |
| [keyframe_calibration](keyframe_calibration/README_zh.md) | 关键帧选择、标定 profile、应用与验收 |
| [joint_pose_attachment](joint_pose_attachment/README_zh.md) | 联合优化姿态与 marker 附着点，构建轮廓检查页面 |
| [full_silhouette](full_silhouette/README_zh.md) | 完整轮廓渲染、手型优化与质量评估 |
| [sdk_skeleton_fit](sdk_skeleton_fit/README_zh.md) | SDK 骨架约束拟合和标签导出 |
| [sdk_direction_fit](sdk_direction_fit/README_zh.md) | SDK 方向约束拟合和标签导出 |
| [manual_shape_tuner](manual_shape_tuner/README_zh.md) | 人工调整手型与姿态，保存多目确认结果 |
| `joint_refit/run.sh` | 转发到 `ego_hand_system/scripts/run_joint_refit_labels.sh`，保留旧命令兼容性 |

## 依赖和生成文件

`marker_placement/index.html` 可以独立打开，支持示意图、上传照片和 JSON 配置编辑。
其中 `prefill/` 及 `prefill.js` 属于指定录制的生成数据，已由 Git 忽略，本机已有预填仍可使用；
重建方法见该目录 README。

Python 标定工具使用本仓库的相机与 MANO 实现。NOKOV 原始轨迹处理还依赖
同级 `nokov_ego_hand_validation`。相关 MANO 模型、原始录制、对齐结果及 `output/` 下的拟合产物需单独准备。
Shell 脚本保留原有本机环境默认值；换机器时按各脚本的 `EGO_NATIVE_PYTHON`、
`KEYFRAME_PYTHON`、`FULL_MASK_PYTHON`、`SDK_FIT_PYTHON` 或 `SKELETON_PYTHON` 环境变量选择解释器。

HTML 查看器与轮廓控件是构建页面时读取的源码模板，应随代码提交。
生成的页面、截图、模型预测、拟合 NPZ 和 renderer 依赖安装结果放在已忽略的 `output/` 下。
播放、缩放与全屏代码已经内嵌于 `marker_placement/label_viewer.html`，不再需要一次性修补脚本。

三个工具 `test_*.py` 验证标定编辑、联合投影和外部流水线的行为，仍有回归价值，
其运行条件和命令见 [测试说明](../../../training/docs/README.md#测试)。

## `training/` 下的配套脚本

- `assemble_joint_refit_trainset.py`：收集已完成标签导出、合并数据，并按时间划分训练／验证集。
- `compare_marker_projection.py`：将矫正图预测映回原图，与 NOKOV marker 投影比较。
- `compare_native_rectified_predictions.py`：在原图相机坐标中比较原图／矫正图的预测。
- `render_glove_inference_video.py`：将已有预测和 marker 投影绘制为对比视频，需可执行的 `ffmpeg`。
