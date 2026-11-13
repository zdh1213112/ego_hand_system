# MANO 模型资产

从 [MANO 项目页](https://mano.is.tue.mpg.de/)注册并按其许可下载模型。目录统一使用小写的 `models/mano/`。

四目共享推理、WiLoR 规范拟合和相应可视化使用：

```text
MANO_RIGHT.pkl
```

物理左手通过右手规范模型与镜像约定重建。完整资产安装脚本和依赖检查同时检查 `MANO_LEFT.pkl`，因此准备完整项目时保留两侧模型。

从用户提供的下载目录、ZIP 或 tar 安装（在仓库根目录执行）：

```bash
python scripts/install_mano_models.py --source /path/to/mano_v1_2.zip
python scripts/check_third_party.py --require-mano
```

拟合前检查模型和已准备的 EGO 输入：

```bash
python scripts/check_mano_assets.py \
  --model-dir models/mano \
  --mano-source third_party/MANO \
  --input output/mano_preparation/mano_input.npz
```

使用和分发模型应遵循 MANO 的许可约定。
