# CIFAR-10 数据集扩展

本流程使用 CIFAR-10 官方训练集和测试集，可独立于 Kodak 实验运行。以下命令分别固定抽取 10,000 张和 1,000 张图像；省略样本数参数时使用全部样本。

首次运行会自动下载 CIFAR-10 至 `实验运行代码/实验数据集/cifar10/`。在项目根目录执行：

```powershell
python 实验运行代码/CIFAR-10数据集扩展/run_cifar10.py --epochs 50 --max-train-samples 10000 --max-test-samples 1000 --device auto
```

入口脚本会依次完成训练、测试和绘图。权重、训练日志、五档 SNR 指标、重建图及汇总曲线默认保存至系统临时目录下的 `deep-jscc-image/CIFAR-10数据集扩展/cifar10_random_snr/`。

快速检查时可将样本数改为 `--max-train-samples 1000 --max-test-samples 200`。入口默认训练 200 轮，上述命令显式指定 50 轮；已有权重时会跳过训练，修改训练设置后可加 `--force-train` 重新训练。
