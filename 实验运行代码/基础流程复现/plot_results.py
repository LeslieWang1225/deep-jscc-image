"""绘制任务一原始基础代码实验结果的报告图表。"""

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "基础流程复现"
DEFAULT_DATA_DIR = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "kodak"
TEST_SNRS = [-10, -5, 0, 5, 10]
DEFAULT_VISUAL_SAMPLES = 3

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def read_csv(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_args():
    parser = argparse.ArgumentParser(description="绘制任务一基础复现图表")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=DEFAULT_VISUAL_SAMPLES,
        help="对比图中显示的 Kodak 样例数（默认：3）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "结果图表"
    figure_dir.mkdir(parents=True, exist_ok=True)

    history = read_csv(output_dir / "training_history.csv")
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(epochs, losses, color="#1f77b4", linewidth=1.8)
    plt.xlabel("训练轮数（Epoch）")
    plt.ylabel("MSE损失")
    plt.title("基础模型训练损失曲线（MSE）")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "task1_training_loss.png", dpi=180)
    plt.close()

    metrics = read_csv(output_dir / "metrics.csv")
    snrs = [float(row["snr_db"]) for row in metrics]
    psnrs = [float(row["psnr"]) for row in metrics]
    ssims = [float(row["ssim"]) for row in metrics]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    left.plot(snrs, psnrs, "o-", color="#1f77b4", linewidth=2)
    left.set_title("（a）PSNR 随信噪比变化", pad=8)
    left.set_xlabel("SNR（dB）")
    left.set_ylabel("PSNR（dB）")
    left.grid(alpha=0.3)

    right.plot(snrs, ssims, "s-", color="#d62728", linewidth=2)
    right.set_title("（b）SSIM 随信噪比变化", pad=8)
    right.set_xlabel("SNR（dB）")
    right.set_ylabel("SSIM")
    right.grid(alpha=0.3)

    fig.suptitle("基础模型在不同信噪比下的重建质量指标", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figure_dir / "task1_snr_quality.png", dpi=180)
    plt.close(fig)

    sources = sorted(Path(args.data_dir).glob("*.png"))[: args.visual_samples]
    if not sources:
        raise FileNotFoundError("未找到可用于重建对比的 Kodak 图像")

    # 每行对应一张 Kodak 图像；原图与五档重建结果均统一为 64×64。
    fig, axes = plt.subplots(
        len(sources), 1 + len(TEST_SNRS), figsize=(14.4, 2.8 * len(sources)), squeeze=False
    )
    for row_index, source in enumerate(sources):
        image_name = source.stem
        panels = [
            (
                "原图（64×64）",
                Image.open(source).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC),
            )
        ]
        for snr in TEST_SNRS:
            reconstruction_path = output_dir / "重建样例" / f"snr_{snr:+d}_{row_index + 1}.png"
            if not reconstruction_path.exists():
                raise FileNotFoundError(
                    f"缺少 {image_name} 在 {snr:+d} dB 下的重建图：{reconstruction_path}"
                )
            panels.append((f"SNR={snr:+d} dB", Image.open(reconstruction_path).convert("RGB")))
        for column_index, (title, image) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image)
            if row_index == 0:
                axis.set_title(
                    "原图：{}（64×64）".format(image_name)
                    if column_index == 0
                    else title
                )
            elif column_index == 0:
                axis.set_title("原图：{}（64×64）".format(image_name))
            axis.axis("off")
    fig.suptitle("原始基础代码在五档 SNR 下的重建结果（3 个 Kodak 样例）")
    fig.tight_layout()
    fig.savefig(figure_dir / "task1_reconstruction_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
