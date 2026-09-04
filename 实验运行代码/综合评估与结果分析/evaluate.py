"""评估语义通信模型并保存结果。"""

import argparse
import csv
import math
import time
import tempfile
from pathlib import Path

import numpy as np
import lpips
import torch
from PIL import Image
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader
from torchvision import transforms

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TASK2_CODE = PROJECT_ROOT / "实验运行代码" / "数据划分与基线模型训练"
sys.path.insert(0, str(TASK2_CODE))

from 公共模块.experiment_config import TEST_SNRS
from 公共模块.experiment_utils import load_or_create_split, resolve_device, save_json, set_seed
from train import KodakDataset, SemanticCommSystem


DEFAULT_DATA_DIR = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "kodak"
TASK2_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "数据划分与基线模型训练"


def parse_args():
    parser = argparse.ArgumentParser(description="评估图像语义通信模型")
    parser.add_argument(
        "--checkpoint", default=str(TASK2_OUTPUT_DIR / "baseline" / "model.pth")
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--split-file", default=str(TASK2_OUTPUT_DIR / "data_split.json")
    )
    parser.add_argument("--test-snrs", type=float, nargs="+", default=TEST_SNRS)
    parser.add_argument("--noise-repeats", type=int, default=5)
    parser.add_argument("--save-image-count", type=int, default=3)
    parser.add_argument("--train-count", type=int, default=18)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument(
        "--lpips-backbone",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS特征主干，默认使用轻量AlexNet",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device, fallback_latent, fallback_resolution):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}
    latent_channels = int(config.get("latent_channels", fallback_latent))
    resolution = int(config.get("resolution", fallback_resolution))
    model = SemanticCommSystem(latent_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict, config, latent_channels, resolution


def tensor_to_image(tensor):
    return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()


def save_image(array, path):
    output = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(output).save(path)


def snr_tag(snr):
    prefix = "m" if snr < 0 else "p"
    value = str(abs(float(snr))).replace(".", "_")
    return "{}{}".format(prefix, value)


def calculate_metrics(reference, reconstruction):
    mse_value = float(np.mean((reference - reconstruction) ** 2))
    psnr_value = float("inf") if mse_value == 0 else 10.0 * math.log10(1.0 / mse_value)
    ssim_value = calculate_ssim(reference, reconstruction)
    return mse_value, psnr_value, ssim_value


def calculate_lpips(perceptual_metric, reference, reconstruction):
    """计算LPIPS感知距离，输入张量范围为[0, 1]。"""
    reference_normalized = reference * 2.0 - 1.0
    reconstruction_normalized = reconstruction * 2.0 - 1.0
    return float(
        perceptual_metric(reference_normalized, reconstruction_normalized)
        .mean()
        .item()
    )


def calculate_ssim(reference, reconstruction, window_size=7):
    """计算彩色图像SSIM。"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    covariance_scale = window_size ** 2 / (window_size ** 2 - 1)
    channel_scores = []
    for channel in range(reference.shape[-1]):
        original = reference[..., channel].astype(np.float64)
        restored = reconstruction[..., channel].astype(np.float64)
        original_mean = uniform_filter(original, size=window_size)
        restored_mean = uniform_filter(restored, size=window_size)
        original_variance = covariance_scale * (
            uniform_filter(original * original, size=window_size)
            - original_mean * original_mean
        )
        restored_variance = covariance_scale * (
            uniform_filter(restored * restored, size=window_size)
            - restored_mean * restored_mean
        )
        covariance = covariance_scale * (
            uniform_filter(original * restored, size=window_size)
            - original_mean * restored_mean
        )
        numerator = (2 * original_mean * restored_mean + c1) * (
            2 * covariance + c2
        )
        denominator = (
            original_mean * original_mean + restored_mean * restored_mean + c1
        ) * (original_variance + restored_variance + c2)
        score_map = numerator / denominator
        pad = (window_size - 1) // 2
        channel_scores.append(float(score_map[pad:-pad, pad:-pad].mean()))
    return float(np.mean(channel_scores))


def benchmark_cpu(state_dict, latent_channels, resolution, runs):
    model = SemanticCommSystem(latent_channels).cpu().eval()
    model.load_state_dict(state_dict)
    sample = torch.zeros(1, 3, resolution, resolution)
    with torch.no_grad():
        for _ in range(10):
            model(sample, 0.0)
        start = time.perf_counter()
        for _ in range(runs):
            model(sample, 0.0)
        elapsed = time.perf_counter() - start
        latent_shape = list(model.encoder(sample).shape[1:])
    return elapsed * 1000.0 / runs, latent_shape


def write_metrics(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "snr_db",
                "mse",
                "psnr_db",
                "ssim",
                "lpips",
                "sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(args):
    if args.noise_repeats <= 0:
        raise ValueError("noise_repeats必须大于0")
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError("未找到模型权重: {}".format(checkpoint_path))

    output_dir = (
        Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "评价结果"
    )
    reconstruction_dir = output_dir / "重建样例"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    model, state_dict, train_config, latent_channels, resolution = load_checkpoint(
        checkpoint_path, device, args.latent_channels, args.resolution
    )
    perceptual_metric = lpips.LPIPS(
        net=args.lpips_backbone, verbose=False
    ).to(device).eval()
    for parameter in perceptual_metric.parameters():
        parameter.requires_grad_(False)
    split = load_or_create_split(
        args.data_dir, args.split_file, args.train_count, args.seed
    )
    transform = transforms.Compose(
        [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
    )
    dataset = KodakDataset(args.data_dir, transform, split["test"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for snr_index, snr in enumerate(args.test_snrs):
            totals = np.zeros(4, dtype=np.float64)
            sample_count = 0
            for repeat in range(args.noise_repeats):
                set_seed(args.seed + snr_index * 1000 + repeat)
                for image_index, images in enumerate(dataloader):
                    images = images.to(device)
                    reconstructed = model(images, snr)
                    original_np = tensor_to_image(images)
                    reconstructed_np = tensor_to_image(reconstructed)
                    lpips_value = calculate_lpips(
                        perceptual_metric, images, reconstructed
                    )
                    totals += np.asarray(
                        calculate_metrics(original_np, reconstructed_np)
                        + (lpips_value,),
                        dtype=np.float64,
                    )
                    sample_count += 1

                    if repeat == 0 and image_index < args.save_image_count:
                        image_name = Path(dataset.img_names[image_index]).stem
                        save_image(
                            original_np,
                            reconstruction_dir / "original_{}.png".format(image_name),
                        )
                        save_image(
                            reconstructed_np,
                            reconstruction_dir
                            / "reconstructed_{}_snr_{}.png".format(
                                image_name, snr_tag(snr)
                            ),
                        )

            averages = totals / sample_count
            row = {
                "snr_db": float(snr),
                "mse": float(averages[0]),
                "psnr_db": float(averages[1]),
                "ssim": float(averages[2]),
                "lpips": float(averages[3]),
                "sample_count": sample_count,
            }
            rows.append(row)
            print(
                "SNR {:+.1f} dB | MSE {:.6f} | PSNR {:.3f} dB | "
                "SSIM {:.4f} | LPIPS {:.4f}".format(
                    snr,
                    row["mse"],
                    row["psnr_db"],
                    row["ssim"],
                    row["lpips"],
                )
            )

    inference_ms, latent_shape = benchmark_cpu(
        state_dict, latent_channels, resolution, args.benchmark_runs
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    latent_values = int(np.prod(latent_shape))
    source_values = 3 * resolution * resolution
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "latent_channels": latent_channels,
        "latent_shape": latent_shape,
        "latent_values": latent_values,
        "source_values": source_values,
        "feature_dimension_ratio": latent_values / source_values,
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "cpu_inference_ms": inference_ms,
        "benchmark_runs": args.benchmark_runs,
        "noise_repeats": args.noise_repeats,
        "test_image_count": len(dataset),
        "test_snrs": list(args.test_snrs),
        "lpips_backbone": args.lpips_backbone,
        "training_config": train_config,
    }
    write_metrics(rows, output_dir / "metrics.csv")
    save_json(summary, output_dir / "summary.json")
    print("评估完成，结果目录: {}".format(output_dir.resolve()))
    return output_dir


def main():
    evaluate_model(parse_args())


if __name__ == "__main__":
    main()
