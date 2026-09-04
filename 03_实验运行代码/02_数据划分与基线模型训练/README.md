# 实验 02：数据划分与基线模型训练

本实验将 24 张 Kodak 图像固定划分为 18 张训练图像和 6 张测试图像，并训练潜特征通道数为 16 的随机 SNR 基线模型。

在项目根目录执行：

```powershell
python 03_实验运行代码/02_数据划分与基线模型训练/create_split.py
python 03_实验运行代码/02_数据划分与基线模型训练/train.py --experiment-name baseline --latent-channels 16 --train-snrs -10 -5 0 5 10 --epochs 200 --device auto
```

数据划分文件与基线模型默认保存至系统临时目录下的 `deep-jscc-image/02_数据划分与基线模型训练/`；模型测试和汇总绘图由实验 06 完成。
