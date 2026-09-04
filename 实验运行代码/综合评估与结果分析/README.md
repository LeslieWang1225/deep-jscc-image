# 综合评估与结果分析

统一评估基线、通道数消融和固定 SNR 训练得到的五个 Kodak 模型，计算 MSE、PSNR、SSIM、LPIPS 和模型复杂度，并生成性能曲线、重建效果对比和模型对比图表。该流程加载已有权重，不会重新训练模型。

在项目根目录执行：

```powershell
python 实验运行代码/综合评估与结果分析/run_metrics_analysis.py --device auto --noise-repeats 5 --benchmark-runs 50
```

运行前需先完成基线训练、通道数消融与 SNR 策略对比，并确认对应 `model.pth` 文件位于系统临时目录下的 `deep-jscc-image/`。汇总结果默认保存至该临时目录的 `综合评估与结果分析/结果图表/`。
