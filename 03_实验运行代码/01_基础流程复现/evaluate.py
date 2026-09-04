"""任务一：在全部 Kodak 图像上评估基础复现模型。"""

import argparse
import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from train import KodakDataset, SemanticCommSystem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "kodak"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "01_基础流程复现"
TEST_SNRS = [-10, -5, 0, 5, 10]
DEFAULT_VISUAL_SAMPLES = 3


def calculate_ssim(reference, reconstruction, window_size=7):
    """计算彩色图像SSIM，避免依赖未列入项目要求的scikit-image。"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    covariance_scale = window_size ** 2 / (window_size ** 2 - 1)
    scores = []
    for channel in range(reference.shape[-1]):
        original = reference[..., channel].astype(np.float64)
        restored = reconstruction[..., channel].astype(np.float64)
        original_mean = uniform_filter(original, size=window_size)
        restored_mean = uniform_filter(restored, size=window_size)
        original_variance = covariance_scale * (uniform_filter(original * original, size=window_size) - original_mean ** 2)
        restored_variance = covariance_scale * (uniform_filter(restored * restored, size=window_size) - restored_mean ** 2)
        covariance = covariance_scale * (uniform_filter(original * restored, size=window_size) - original_mean * restored_mean)
        score_map = ((2 * original_mean * restored_mean + c1) * (2 * covariance + c2)) / ((original_mean ** 2 + restored_mean ** 2 + c1) * (original_variance + restored_variance + c2))
        pad = (window_size - 1) // 2
        scores.append(float(score_map[pad:-pad, pad:-pad].mean()))
    return float(np.mean(scores))


def calculate_metrics(reference, reconstruction):
    mse_value = float(np.mean((reference - reconstruction) ** 2))
    psnr_value = float("inf") if mse_value == 0 else 10.0 * math.log10(1.0 / mse_value)
    return mse_value, psnr_value, calculate_ssim(reference, reconstruction)


def parse_args():
    parser = argparse.ArgumentParser(description="任务一：基础模型评估")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=DEFAULT_VISUAL_SAMPLES,
        help="保存用于重建对比图的前几个 Kodak 样例（默认：3）",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "semantic_model.pth"
    transform = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor()]
    )
    dataset = KodakDataset(args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = SemanticCommSystem(checkpoint.get("latent_channels", 16)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    image_dir = output_dir / "重建样例"
    image_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for snr in TEST_SNRS:
            values = []
            for index, image in enumerate(loader):
                image = image.to(device)
                reconstructed = model(image, snr)
                original_np = image[0].cpu().numpy().transpose(1, 2, 0)
                reconstructed_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
                values.append(
                    (
                        calculate_metrics(original_np, reconstructed_np)
                    )
                )
                if index < args.visual_samples:
                    save_image(reconstructed, image_dir / f"snr_{snr:+d}_{index + 1}.png")
            means = np.mean(values, axis=0)
            rows.append({"snr_db": snr, "mse": means[0], "psnr": means[1], "ssim": means[2]})

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["snr_db", "mse", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"评估完成：{output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
