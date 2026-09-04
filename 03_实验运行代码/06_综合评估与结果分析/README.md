# 实验 06：综合评估与结果分析

本实验对实验 02、04、05 训练得到的五个 Kodak 模型统一计算 MSE、PSNR、SSIM、LPIPS 和模型复杂度，并生成性能曲线、重建效果对比和模型改进对比图表；该流程不会重新训练模型。

在项目根目录执行：

```powershell
python 03_实验运行代码/06_综合评估与结果分析/run_metrics_analysis.py --device auto --noise-repeats 5 --benchmark-runs 50
```

运行前需先完成实验 02、04 和 05，并确认对应 `model.pth` 文件位于系统临时目录下的 `deep-jscc-image/`。汇总结果默认保存至该临时目录的 `06_综合评估与结果分析/结果图表/`。
