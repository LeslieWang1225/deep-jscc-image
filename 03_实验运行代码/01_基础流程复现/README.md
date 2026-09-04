# 实验 01：基础流程复现

本实验沿用教师提供的 `train.py`、`evaluate.py` 与 `plot_results.py` 流程，使用全部 24 张 Kodak 图像完成基础训练、测试和结果绘制，不划分训练集与测试集。

在项目根目录执行：

```powershell
python 03_实验运行代码/01_基础流程复现/train.py --epochs 200 --device auto
python 03_实验运行代码/01_基础流程复现/evaluate.py --device auto
python 03_实验运行代码/01_基础流程复现/plot_results.py
```

默认输出至系统临时目录下的 `deep-jscc-image/01_基础流程复现/`。该实验仅用于验证基础流程，不用于评价模型泛化能力。
