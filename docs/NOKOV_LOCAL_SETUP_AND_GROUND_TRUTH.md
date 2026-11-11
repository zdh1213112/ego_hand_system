# 本机 NOKOV 部署与手部真值采集记录

检查日期：2026-10-22。用途：手部动作采集，作为现有 EGO 手部重建系统的独立参考。
本记录来自本机只读检查和 NOKOV 官方文档。未安装软件、修改网络或连接相机。

## 本机环境

| 项目 | 实测结果 |
|---|---|
| 系统 | Ubuntu 24.04.4 LTS，Noble |
| 架构 | x86_64 / Debian amd64，非虚拟机 |
| 内核 | 7.0.0-38-generic |
| CPU | Intel Core i7-14650HX，24 逻辑线程 |
| 内存 | 约 15.3 GiB；检查时可用约 2.9 GiB，可用值会变化 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |
| NVIDIA 驱动 | 595.99.02 |
| 工作区所在分区剩余空间 | 约 340 GiB |
| 桌面 | X11，DISPLAY=:1 |
| 有线网卡 | enp12s0，链路 1000 Mb/s，192.168.111.72/24 |
| 无线网卡 | wlp18s0，192.168.111.158/24 |
| 虚拟网络接口 | Meta，198.18.0.1/30 |
| Python | Miniforge，检查时激活 base |
| 已有 Conda 环境 | TacFlow、loreal、opentouch、wilor、xense-taccap |
| 开发工具 | cmake、g++、tcpdump 可用 |
| NOKOV 状态 | dpkg 未发现 xingying/nokov/vrpn 匹配包；未发现同名运行进程 |
| ROS 状态 | 未发现 /opt/ros 下标准安装；PATH 中没有 ros2、colcon |

上述检查不能排除非标准路径里的 SDK/便携程序。USB 枚举不足以识别通过以太网连接的
NOKOV 相机，也未据此确认相机型号、数量、同步盒、授权和采集主机。
网络地址和可用资源属于检查时快照，实际部署前应重新读取。

## 官方兼容范围与部署选择

官网列出的 XINGYING-Linux 为 v3.4.0.6843，明确写明兼容 Ubuntu 22.04。
本次未找到 Ubuntu 24.04 的官方支持声明，也未验证 3.4 Linux 包含 4.6 文档中的
Left/Right Hand(24) 模板。因此本机 24.04 上直接安装主程序属于待验证方案。

- [官方下载中心](https://en.nokov.com/support/downloads.html)
- [主程序版本与安装说明](https://en.nokov.com/support/downloads/desc/Software.html)
- [星影 3.4 中文操作手册](https://docs.nokov.com/xingying/XingYing3.4-CN/)
- [星影 4.6 Windows 安装章节](https://docs.nokov.com/xingying/XingYing4.6-CN/san-ruan-jian-de-an-zhuang-yu-qi-dong/san-ruan-jian-an-zhuang/)

优先方案：如已有正常运行的 NOKOV 采集主机，由它完成相机连接、标定、手部模板
创建和原始数据保存；本机保留 Ubuntu 24.04，用官方 SDK 接收或离线读取导出文件。
SDK 本身在 24.04 的运行兼容性仍需用实际包验证。首次离线试验无需先安装 ROS。

若必须由本机直接连接相机，应先拿到实际 amd64 安装包和发行说明，核对 Ubuntu
24.04、相机型号、授权方式、24 点手部模板及导出功能，再决定部署。若不支持，
可使用独立的官方支持系统采集主机；不要为此直接重装当前开发系统。

官网有一处架构描述矛盾：安装环境写 arm64，示例文件名写 amd64。应使用
`dpkg-deb -f ./实际安装包.deb Package Version Architecture Depends` 检查实际包，
本机选择 amd64，不选择 arm64。安装前还应阅读包内依赖及启动说明。

## 现有工程衔接

已有方案见 [NOKOV_OPTICAL_MOCAP_TO_WILOR_LABELS.md](NOKOV_OPTICAL_MOCAP_TO_WILOR_LABELS.md)。
该文档是设计说明，不能视为 SDK 已接入、转换脚本已实现或精度已验证。

当前 EGO 主流程涉及 GEN/Orbbec、多目 WiLoR、MediaPipe 和 MANO；反光点辅助 WiLoR
是图像侧辅助，不能代替独立 NOKOV 数据。触觉工具箱说明见
[glove_toolbox README](../../xenseglovesdk/glove_toolbox/README.md)。触觉工具箱目前按
主机接收时间对齐，SDK 只提供最新帧，不保证设备每帧全部被记录，尚无硬件同步保证。

## 手部采集应读的章节

1. [Left/Right Hand(24)](https://docs.nokov.com/xingying/XingYing4.6-CN/jiu-chuang-jian-markerset/er-body-ren-ti/left-right-hand-24/)：贴点、冻结帧、建模姿态和朝向。
2. [基础标定](https://docs.nokov.com/xingying/XingYing4.6-CN/ba-biao-ding-fen-lei/san-ji-chu-biao-ding/)：相机空间与全局轴向。
3. [TRC 导出](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-san-shu-ju-dao-chu/wu-trc-wen-jian/)：点轨迹、帧、时间、时间戳以及反算点说明。
4. [C3D 导出](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-san-shu-ju-dao-chu/er-c3d-wen-jian/)：Markerset 命名点。
5. [数据广播](https://docs.nokov.com/xingying/XingYing4.6-CN/liu-shi-tu-jie-shao/san-shu-ju-guang-bo/)与 [SDK 概览](https://docs.nokov.com/xingying/XingYing4.6-CN/shi-qi-cha-jian-sdk/shi-san-sdk/)：实时接收入口。

## 真值的定义和独立性

采集至少保留三层数据，并记录来源，不能统称为同等精度的真值：

1. 实测 marker：有名称、左右手身份、时间戳的 3D 表面点；保留原始采集文件。
2. NOKOV 模型解算的关节/骨骼：含骨长、关节约束与表面点偏移等模型假设。
3. 转为 MANO/WiLoR 的参数及 21 关节：含映射与拟合误差的派生标签。

24 个表面 marker 不等于 21 个关节中心，不能直接取前 21 点，也不能默认星影
骨骼旋转或手型参数就是 MANO 参数。需要用实际导出的名称、静态姿态和标定确认。
NOKOV 参考精度应通过本场地静态噪声、已知长度和动态试验测定，不能仅引用相机规格。

TRC 文档说明可输出反算点，名称为 `RCalc_标记点名称`，丢失的原始点也可能被补齐。
应核对实际软件的补点设置与有效性标识，保留 observed/inferred/invalid 状态。
只有有限坐标值不等于该点在该帧被真实观测。无法区分时，不计入严格实测误差指标。

评估 WiLoR/视觉算法时，最终参考应独立于被评估算法。不要用测试序列中 WiLoR
预测优化 NOKOV 标签到相机的外参、时间偏移或关节位置后，再用该标签评估 WiLoR。
优先使用独立标定物/同步信号与单独校准序列；模型拟合与过滤规则在测试前固定。

## 时间、空间和网络

- 时间：优先核实各设备是否支持共享硬件触发；否则估计时钟偏移与漂移，并单独验证。
  保存设备采样时间、帧号和主机接收时间及其时钟域，不把接收时刻当作采样时刻。
- 空间：固定相机可标定固定外参。头戴 EGO 相机需跟踪头环刚体，结合固定安装外参
  逐帧求相机位姿；保留 GEN/Orbbec 实际鱼眼模型、单位和变换矩阵方向。
- 当前以太网和 Wi-Fi 同属 192.168.111.0/24，另有虚拟接口。接入时显式选择发送与
  接收网卡，并验证路由/组播路径；当前没有发现已配置的 10.1.1.0/24 直连路由。
  不应把当前正在使用的有线地址直接改成手册示例地址。先确认相机交换机和主机拓扑，
  再配置独立采集网口或专用网段。

## 第一轮实际验收

1. 确认相机型号/数量、星影实际版本、授权、同步盒及 Hand(24) 可用性。
2. 单手静态张手，按模板贴点并建模；保存名称与点位对应表。
3. 记录静止、逐指屈伸、捏合、握拳和短时遮挡恢复；再增加双手与物体交互。
4. 保存原始采集文件及 TRC/C3D，检查实测/补点、时间戳含义、单位、左右手和丢帧。
5. 用独立校准序列建立 EGO 与 NOKOV 的时间和空间关系，再评估正式动作。
6. 报告覆盖率、丢点率、静态抖动、时间残差、marker 残差、关节/拟合误差；
   不能仅以骨架动画流畅或重投影好看判断真值系统有效。

下一步依赖实际硬件拓扑、兼容安装包或一段带名称和时间信息的导出数据；当前尚未
进行主程序安装、SDK 链路验证或精度验收。
