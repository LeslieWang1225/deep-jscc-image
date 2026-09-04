# 潜特征通道数消融

本实验在相同数据划分与训练设置下，独立运行潜特征通道数 C=4、8、32 三组模型，并与 C=16 基线比较模型性能和复杂度。

运行全部消融组：

```powershell
python 实验运行代码/潜特征通道数消融/run_latent_channels.py --latent-channels 4 8 32 --epochs 200 --device auto
```

只运行一组（以 C=8 为例）：

```powershell
python 实验运行代码/潜特征通道数消融/run_latent_channels.py --latent-channels 8 --epochs 200 --device auto
```

结果默认保存至系统临时目录下的 `deep-jscc-image/潜特征通道数消融/`。运行前需先完成基线训练，以生成固定数据划分文件。
