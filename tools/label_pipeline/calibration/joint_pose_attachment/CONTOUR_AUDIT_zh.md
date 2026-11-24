# 实际优化轮廓检查（CONTOUR-AUDIT-1）

入口：http://127.0.0.1:8766/contour_audit/ 。默认 camera0 / sync632 / 右手。

绿色区域和外边界是分割重放，紫色点从本轮 `observations.json` 原样读取，是实际用于轮廓损失的点。没有用 MANO 投影轮廓替代图像观测；没有在腕部裁剪处或稀疏点之间补线。绿色完整区域可能包含袖口，只有紫点用于优化。

支持切换四路相机、左右手、同步帧；筛选“实际参与联合优化／全部接受／被剔除／全部”；原图与叠加同步缩放和平移、单独放大、全屏、采样播放和截图。可以记录漏手指、背景污染、双手粘连、身份错误等检查结论并导出 JSON；不需要逐点标注。

## 发现的实际问题

**camera0、sync632、右手的分割连到了背景横梁。紫色实际优化点中也包含这段背景边界。**这不是查看器渲染错误，而是上游颜色分割及门控接受了错误区域。

因此此前报告的自动轮廓距离下降不能证明真实手套贴合改善。联合修正结果保留为实验，需先修复观测再重新评估。本页面保留原输入供检查，没有悄悄替换成修正后的分割。

## 来源与核验

共 800 张原图，对应 1552 份手／相机／时刻记录：接受 829 份，剔除 723 份；804 份实际进入全部可用视角的联合优化。

从原始 MP4 像素、相同 source frame index、同一版分割逻辑和原始 marker fit 重放，829 份接受样本的轮廓点均与保存观测逐点相同，0 份不一致。这证明来源一致，不表示分割正确。被剔除的样本显示原图及原始剔除原因，不伪造优化轮廓。

数据来源：`joint_pose_attachment_observations_v1/observations.json`。分割提示来自 `explicit_pipeline_sampled_smoke_v2/marker24`，不能用轮廓修正后的 mesh 代替它重放，否则会改变腕部裁剪等条件。

## 生成命令

```bash
cd /home/p3/data_sda1/EGO_glove/wilor_training
/home/p3/miniforge3/envs/ego-hand/bin/python tools/joint_pose_attachment/build_contour_audit.py   --context output/native_kb_training_20261111/explicit_pipeline_sampled_smoke_v2/context.json   --source output/native_kb_training_20261111/explicit_pipeline_sampled_smoke_v2/marker24   --observations output/native_kb_training_20261111/joint_pose_attachment_observations_v1/observations.json   --joint-fit output/native_kb_training_20261111/joint_pose_attachment_all_v1   --output /新的轮廓检查输出目录
```

当前已生成，无需重新执行。8766 服务未启动时可运行：

```bash
bash tools/marker_placement/start_viewer.sh
```

已验证实际点与原始输入完全一致、分割重放一致、默认632帧、覆盖显示切换、拒绝样本无伪造点、相机切换、播放／暂停及检查记录。
