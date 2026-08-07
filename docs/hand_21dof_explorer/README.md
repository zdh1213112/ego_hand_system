# EGO Hand 21-DOF 交互学习器

面向新用户的纯静态交互网页，用于理解本项目导出的 MANO 21 个手指局部自由度。

## 打开

直接用浏览器打开 `index.html`，或者在项目根目录运行：

```bash
python -m http.server 8000 --directory docs/hand_21dof_explorer
```

然后访问 `http://localhost:8000`。

页面没有 npm 或网络依赖，支持：

- 左右手切换；
- 掌心/手背一键切换，并用掌纹、指甲和文字提示区分正反面；
- 21 个自由度逐项选择和角度调节；
- 张开、握拳、捏取和指向预设；
- 拖动旋转、滚轮缩放及点击关节选择；
- EGO CSV 字段、弧度与角度的实时对应；
- 关节层级、正负方向以及 21-DOF 与 Hand 6D Pose 的区别说明。

教学骨架用于解释项目的运动学约定，不作为医学或临床量角工具。

可使用查询参数直接打开指定视图，例如：

```text
index.html?hand=Left&surface=back
```
