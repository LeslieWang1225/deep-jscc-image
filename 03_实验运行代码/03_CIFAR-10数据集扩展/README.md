# 实验 03：CIFAR-10 数据集扩展

本实验将基础任务的 Kodak 图像集替换为 CIFAR-10，从官方训练集和测试集中分别固定抽取 10,000 张和 1,000 张图像，不复用 Kodak 的 18:6 划分。

首次运行会自动下载 CIFAR-10 至 `03_实验运行代码/实验数据集/cifar10/`。在项目根目录执行：

```powershell
python 03_实验运行代码/03_CIFAR-10数据集扩展/run_cifar10.py --epochs 50 --max-train-samples 10000 --max-test-samples 1000 --device auto
```

入口脚本会依次完成训练、测试和绘图。权重、训练日志、五档 SNR 指标、重建图及汇总曲线默认保存至系统临时目录下的 `deep-jscc-image/03_CIFAR-10数据集扩展/cifar10_random_snr/`。

快速检查时可将样本数改为 `--max-train-samples 1000 --max-test-samples 200`；正式复现实验使用上述 10,000/1,000 样本与 50 轮训练命令。
