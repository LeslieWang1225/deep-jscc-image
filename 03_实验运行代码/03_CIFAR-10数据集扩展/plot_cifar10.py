"""绘制任务三 CIFAR-10 数据集扩展实验图表。"""

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
DEFAULT_EXPERIMENT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "03_CIFAR-10数据集扩展" / "cifar10_random_snr"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 CIFAR-10 扩展实验图表")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_curve(x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(x, y, marker="o", linewidth=2, color="#1f77b4")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_metric_triptych(snrs, metrics, path):
    """将三项 SNR 指标横向并列，作为报告中的三联图。"""
    specifications = [
        ("psnr_db", "PSNR（dB）", "PSNR"),
        ("ssim", "SSIM", "SSIM"),
        ("lpips", "LPIPS（越低越好）", "LPIPS"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    for axis, (metric, ylabel, title) in zip(axes, specifications):
        axis.plot(
            snrs,
            [float(row[metric]) for row in metrics],
            marker="o",
            linewidth=1.8,
            color="#1f77b4",
        )
        axis.set_xlabel("信噪比 SNR（dB）")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.45)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    figure_dir = experiment_dir / "结果图表"
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = read_csv(experiment_dir / "training_history.csv")
    metrics = read_csv(experiment_dir / "评价结果" / "metrics.csv")
    save_curve([int(row["epoch"]) for row in history], [float(row["loss"]) for row in history], "训练轮数（Epoch）", "MSE 损失", "CIFAR-10 训练损失曲线", figure_dir / "cifar10_training_loss.png")
    snrs = [float(row["snr_db"]) for row in metrics]
    save_curve(snrs, [float(row["psnr_db"]) for row in metrics], "信噪比 SNR（dB）", "PSNR（dB）", "CIFAR-10：SNR 与 PSNR 的关系", figure_dir / "cifar10_snr_psnr.png")
    save_curve(snrs, [float(row["ssim"]) for row in metrics], "信噪比 SNR（dB）", "SSIM", "CIFAR-10：SNR 与 SSIM 的关系", figure_dir / "cifar10_snr_ssim.png")
    save_curve(snrs, [float(row["lpips"]) for row in metrics], "信噪比 SNR（dB）", "LPIPS（越低越好）", "CIFAR-10：SNR 与 LPIPS 的关系", figure_dir / "cifar10_snr_lpips.png")
    save_metric_triptych(snrs, metrics, figure_dir / "cifar10_snr_metrics_triptych.png")
    print("图表已保存至: {}".format(figure_dir.resolve()))


if __name__ == "__main__":
    main()
