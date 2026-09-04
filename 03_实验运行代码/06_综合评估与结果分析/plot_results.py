"""汇总实验结果并绘制报告图表。"""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TASK2_CODE = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练"
sys.path.insert(0, str(TASK2_CODE))

from 公共模块.experiment_config import EXPERIMENTS


# 所有报告图表采用中文字体和中文标注，避免正文中文而图内仍为英文。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "06_综合评估与结果分析" / "结果图表"


def parse_args():
    parser = argparse.ArgumentParser(description="绘制语义通信实验结果")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--figure-dir",
        default=str(DEFAULT_FIGURE_DIR),
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_results(output_root):
    results = {}
    for experiment in EXPERIMENTS:
        experiment_dir = output_root / experiment["result_task"] / experiment["name"]
        metrics_path = experiment_dir / "评价结果" / "metrics.csv"
        summary_path = experiment_dir / "评价结果" / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            continue
        metrics = read_csv(metrics_path)
        for row in metrics:
            row["snr_db"] = float(row["snr_db"])
            row["mse"] = float(row["mse"])
            row["psnr_db"] = float(row["psnr_db"])
            row["ssim"] = float(row["ssim"])
            row["lpips"] = float(row["lpips"])
        results[experiment["name"]] = {
            "config": experiment,
            "metrics": metrics,
            "summary": read_json(summary_path),
            "directory": experiment_dir,
        }
    return results


def save_line_plot(series, metric, ylabel, title, output_path):
    plt.figure(figsize=(7.2, 4.8))
    for label, rows in series:
        rows = sorted(rows, key=lambda row: row["snr_db"])
        plt.plot(
            [row["snr_db"] for row in rows],
            [row[metric] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    plt.xlabel("信噪比 SNR（dB）")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metric_triptych(series, metric_specs, output_path):
    """将同一实验的 PSNR、SSIM、LPIPS 横向并列，便于比较趋势。"""
    figure, axes = plt.subplots(1, len(metric_specs), figsize=(15.2, 4.4))
    for axis, (metric, ylabel, title) in zip(axes, metric_specs):
        for label, rows in series:
            rows = sorted(rows, key=lambda row: row["snr_db"])
            axis.plot(
                [row["snr_db"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set_xlabel("信噪比 SNR（dB）")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.45)
        if len(series) > 1:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_required_curves(results, figure_dir):
    baseline = results.get("baseline")
    if baseline:
        save_line_plot(
            [("基线模型", baseline["metrics"])],
            "psnr_db",
            "PSNR (dB)",
            "基线模型：SNR 与 PSNR 的关系",
            figure_dir / "baseline_snr_psnr.png",
        )
        save_line_plot(
            [("基线模型", baseline["metrics"])],
            "ssim",
            "SSIM",
            "基线模型：SNR 与 SSIM 的关系",
            figure_dir / "baseline_snr_ssim.png",
        )
        save_line_plot(
            [("基线模型", baseline["metrics"])],
            "lpips",
            "LPIPS（越低越好）",
            "基线模型：SNR 与 LPIPS 的关系",
            figure_dir / "baseline_snr_lpips.png",
        )
        save_metric_triptych(
            [("基线模型", baseline["metrics"])],
            [
                ("psnr_db", "PSNR (dB)", "PSNR"),
                ("ssim", "SSIM", "SSIM"),
                ("lpips", "LPIPS（越低越好）", "LPIPS"),
            ],
            figure_dir / "baseline_snr_metrics_triptych.png",
        )

    latent_names = ["latent_4", "latent_8", "baseline", "latent_32"]
    latent_series = [
        (results[name]["config"]["label"], results[name]["metrics"])
        for name in latent_names
        if name in results
    ]
    if latent_series:
        save_line_plot(
            latent_series,
            "psnr_db",
            "PSNR (dB)",
            "潜特征通道数：SNR 与 PSNR 的关系",
            figure_dir / "latent_channels_psnr.png",
        )
        save_line_plot(
            latent_series,
            "ssim",
            "SSIM",
            "潜特征通道数：SNR 与 SSIM 的关系",
            figure_dir / "latent_channels_ssim.png",
        )
        save_line_plot(
            latent_series,
            "lpips",
            "LPIPS（越低越好）",
            "潜特征通道数：SNR 与 LPIPS 的关系",
            figure_dir / "latent_channels_lpips.png",
        )
        save_metric_triptych(
            latent_series,
            [
                ("psnr_db", "PSNR (dB)", "PSNR"),
                ("ssim", "SSIM", "SSIM"),
                ("lpips", "LPIPS（越低越好）", "LPIPS"),
            ],
            figure_dir / "latent_channels_metrics_triptych.png",
        )

    robustness_names = ["baseline", "fixed_snr_5"]
    robustness_series = [
        (results[name]["config"]["label"], results[name]["metrics"])
        for name in robustness_names
        if name in results
    ]
    if robustness_series:
        save_line_plot(
            robustness_series,
            "psnr_db",
            "PSNR (dB)",
            "随机与固定 SNR 训练策略对比",
            figure_dir / "random_vs_fixed_psnr.png",
        )
        save_line_plot(
            robustness_series,
            "ssim",
            "SSIM",
            "随机与固定 SNR 训练策略对比",
            figure_dir / "random_vs_fixed_ssim.png",
        )
        save_line_plot(
            robustness_series,
            "lpips",
            "LPIPS（越低越好）",
            "随机与固定 SNR 训练策略：LPIPS 对比",
            figure_dir / "random_vs_fixed_lpips.png",
        )
        save_metric_triptych(
            robustness_series,
            [
                ("psnr_db", "PSNR (dB)", "PSNR"),
                ("ssim", "SSIM", "SSIM"),
                ("lpips", "LPIPS（越低越好）", "LPIPS"),
            ],
            figure_dir / "random_vs_fixed_metrics_triptych.png",
        )


def plot_latent_tradeoff(results, figure_dir):
    names = ["latent_4", "latent_8", "baseline", "latent_32"]
    selected_snrs = [-10.0, 0.0, 10.0]
    available = [results[name] for name in names if name in results]
    if not available:
        return

    plt.figure(figsize=(7.2, 4.8))
    for snr in selected_snrs:
        channels = []
        psnr_values = []
        for item in available:
            row = next(
                (row for row in item["metrics"] if row["snr_db"] == snr), None
            )
            if row:
                channels.append(item["summary"]["latent_channels"])
                psnr_values.append(row["psnr_db"])
        if channels:
            pairs = sorted(zip(channels, psnr_values))
            plt.plot(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                marker="o",
                linewidth=2,
                label="SNR {:+g} dB".format(snr),
            )
    plt.xlabel("潜特征通道数 C")
    plt.ylabel("PSNR (dB)")
    plt.title("潜特征通道数与重建质量的折中")
    plt.xticks([4, 8, 16, 32])
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "latent_tradeoff.png", dpi=220)
    plt.close()


def plot_training_curves(results, figure_dir):
    plt.figure(figsize=(7.2, 4.8))
    plotted = False
    for item in results.values():
        history_path = item["directory"] / "training_history.csv"
        if not history_path.exists():
            continue
        history = read_csv(history_path)
        plt.plot(
            [int(row["epoch"]) for row in history],
            [float(row["loss"]) for row in history],
            linewidth=1.5,
            label=item["config"]["label"],
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("训练轮数（Epoch）")
    plt.ylabel("MSE 损失")
    plt.title("五个模型的训练损失曲线")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "training_loss.png", dpi=220)
    plt.close()


def plot_reconstruction_grid(results, figure_dir):
    baseline = results.get("baseline")
    if not baseline:
        return
    image_dir = baseline["directory"] / "评价结果" / "重建样例"
    originals = sorted(image_dir.glob("original_*.png"))
    if not originals:
        return
    snr_tags = [
        ("m10_0", "SNR=-10 dB"),
        ("m5_0", "SNR=-5 dB"),
        ("p0_0", "SNR=+0 dB"),
        ("p5_0", "SNR=+5 dB"),
        ("p10_0", "SNR=+10 dB"),
    ]
    rows = min(3, len(originals))
    figure, axes = plt.subplots(
        rows, 1 + len(snr_tags), figsize=(14.4, 2.8 * rows), squeeze=False
    )
    for row_index, original_path in enumerate(originals[:rows]):
        image_name = original_path.stem.replace("original_", "")
        axes[row_index, 0].imshow(Image.open(original_path))
        axes[row_index, 0].set_title("原图：{}（64×64）".format(image_name))
        axes[row_index, 0].axis("off")
        for column_index, (tag, label) in enumerate(snr_tags, start=1):
            reconstruction_path = image_dir / "reconstructed_{}_snr_{}.png".format(
                image_name, tag
            )
            if reconstruction_path.exists():
                axes[row_index, column_index].imshow(Image.open(reconstruction_path))
            if row_index == 0:
                axes[row_index, column_index].set_title(label)
            axes[row_index, column_index].axis("off")
    figure.suptitle("基线模型在五档 SNR 下的重建示例（3 个 Kodak 测试样例）")
    figure.tight_layout()
    figure.savefig(figure_dir / "reconstruction_comparison.png", dpi=220)
    plt.close(figure)


def plot_model_reconstruction_comparison(results, figure_dir):
    model_names = ["latent_4", "baseline", "latent_32"]
    if any(name not in results for name in model_names):
        return
    baseline_dir = results["baseline"]["directory"] / "评价结果" / "重建样例"
    originals = sorted(baseline_dir.glob("original_*.png"))
    if not originals:
        return

    rows = min(3, len(originals))
    figure, axes = plt.subplots(rows, 4, figsize=(10, 2.8 * rows), squeeze=False)
    column_titles = ["原图", "C=4", "C=16", "C=32"]
    for row_index, original_path in enumerate(originals[:rows]):
        image_name = original_path.stem.replace("original_", "")
        axes[row_index, 0].imshow(Image.open(original_path))
        axes[row_index, 0].set_title("原图：{}".format(image_name))
        axes[row_index, 0].axis("off")
        for column_index, model_name in enumerate(model_names, start=1):
            reconstruction_dir = (
                results[model_name]["directory"] / "评价结果" / "重建样例"
            )
            reconstruction_path = reconstruction_dir / (
                "reconstructed_{}_snr_p0_0.png".format(image_name)
            )
            if reconstruction_path.exists():
                axes[row_index, column_index].imshow(Image.open(reconstruction_path))
            axes[row_index, column_index].axis("off")
        if row_index == 0:
            for column_index, title in enumerate(column_titles):
                if column_index == 0:
                    continue
                axes[row_index, column_index].set_title(title)
    figure.suptitle("0 dB 下不同潜特征通道数的重建对比（3 个 Kodak 测试样例）")
    figure.tight_layout()
    figure.savefig(figure_dir / "model_reconstruction_comparison.png", dpi=220)
    plt.close(figure)


def export_tables(results, figure_dir):
    metric_rows = []
    complexity_rows = []
    for name, item in results.items():
        for row in item["metrics"]:
            metric_rows.append(
                {
                    "experiment": name,
                    "label": item["config"]["label"],
                    "latent_channels": item["summary"]["latent_channels"],
                    "snr_db": row["snr_db"],
                    "mse": row["mse"],
                    "psnr_db": row["psnr_db"],
                    "ssim": row["ssim"],
                    "lpips": row["lpips"],
                }
            )
        training_config = item["summary"].get("training_config", {})
        complexity_rows.append(
            {
                "experiment": name,
                "latent_channels": item["summary"]["latent_channels"],
                "feature_dimension_ratio": item["summary"]["feature_dimension_ratio"],
                "parameter_count": item["summary"]["parameter_count"],
                "checkpoint_bytes": item["summary"]["checkpoint_bytes"],
                "training_seconds": training_config.get("training_seconds", ""),
                "cpu_inference_ms": item["summary"]["cpu_inference_ms"],
            }
        )
    write_csv(
        figure_dir / "all_metrics.csv",
        metric_rows,
        [
            "experiment",
            "label",
            "latent_channels",
            "snr_db",
            "mse",
            "psnr_db",
            "ssim",
            "lpips",
        ],
    )
    write_csv(
        figure_dir / "complexity_summary.csv",
        complexity_rows,
        [
            "experiment",
            "latent_channels",
            "feature_dimension_ratio",
            "parameter_count",
            "checkpoint_bytes",
            "training_seconds",
            "cpu_inference_ms",
        ],
    )


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(output_root)
    if not results:
        raise RuntimeError("未找到可绘制的实验结果")
    plot_required_curves(results, figure_dir)
    plot_latent_tradeoff(results, figure_dir)
    plot_training_curves(results, figure_dir)
    plot_reconstruction_grid(results, figure_dir)
    plot_model_reconstruction_comparison(results, figure_dir)
    export_tables(results, figure_dir)
    print("图表已保存: {}".format(figure_dir.resolve()))


if __name__ == "__main__":
    main()
