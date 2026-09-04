# SNR 训练策略对比

本实验训练固定 5 dB SNR 模型，并与系统临时目录 `deep-jscc-image/数据划分与基线模型训练/baseline/` 中的随机 SNR 基线模型进行鲁棒性对比。

在项目根目录执行：

```powershell
python 实验运行代码/SNR训练策略对比/run_snr_robustness.py --fixed-train-snr 5 --epochs 200 --device auto
```

结果默认保存至系统临时目录下的 `deep-jscc-image/SNR训练策略对比/`。运行前需先完成基线训练，以生成固定数据划分文件。
