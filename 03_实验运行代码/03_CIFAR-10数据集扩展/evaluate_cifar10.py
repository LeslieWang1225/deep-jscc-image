"""任务三：在 CIFAR-10 官方测试集上评估 Deep JSCC 模型。"""

import argparse
import csv
import tempfile
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
TASK2_CODE = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练"
TASK6_CODE = PROJECT_ROOT / "03_实验运行代码" / "06_综合评估与结果分析"
sys.path.insert(0, str(TASK2_CODE))
sys.path.insert(0, str(TASK6_CODE))

from train import SemanticCommSystem
from 公共模块.experiment_config import TEST_SNRS
from 公共模块.experiment_utils import resolve_device, save_json, set_seed
from evaluate import calculate_lpips, calculate_metrics, snr_tag, tensor_to_image


DEFAULT_DATA_ROOT = PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "cifar10"
DEFAULT_EXPERIMENT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "03_CIFAR-10数据集扩展" / "cifar10_random_snr"


def parse_args():
    parser = argparse.ArgumentParser(description="评估 CIFAR-10 图像语义通信模型")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_EXPERIMENT_DIR / "model.pth"))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPERIMENT_DIR / "评价结果"))
    parser.add_argument("--test-snrs", type=float, nargs="+", default=TEST_SNRS)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--save-image-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lpips-backbone", choices=["alex", "vgg", "squeeze"], default="alex")
    return parser.parse_args()


def select_samples(dataset, maximum, seed):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum <= 0:
        raise ValueError("max_test_samples必须为正数")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:maximum].tolist()
    return Subset(dataset, indices)


def save_image(array, path):
    Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)).save(path)


def write_rows(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["snr_db", "mse", "psnr_db", "ssim", "lpips", "sample_count"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(args):
    if args.noise_repeats <= 0:
        raise ValueError("noise_repeats必须大于0")
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = SemanticCommSystem(int(config.get("latent_channels", 16))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    resolution = int(config.get("resolution", 32))
    transform = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])
    dataset = select_samples(datasets.CIFAR10(args.data_root, train=False, transform=transform, download=True), args.max_test_samples, args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output_dir)
    reconstruction_dir = output_dir / "重建样例"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    perceptual = lpips.LPIPS(net=args.lpips_backbone, verbose=False).to(device).eval()
    rows = []
    with torch.no_grad():
        for snr_index, snr in enumerate(args.test_snrs):
            totals = np.zeros(4, dtype=np.float64)
            samples = 0
            for repeat in range(args.noise_repeats):
                set_seed(args.seed + snr_index * 1000 + repeat)
                for index, (images, _) in enumerate(loader):
                    images = images.to(device)
                    reconstructed = model(images, snr)
                    reference = tensor_to_image(images)
                    restored = tensor_to_image(reconstructed)
                    totals += np.asarray(calculate_metrics(reference, restored) + (calculate_lpips(perceptual, images, reconstructed),))
                    samples += 1
                    if repeat == 0 and index < args.save_image_count:
                        name = "cifar10_{:05d}".format(index)
                        save_image(reference, reconstruction_dir / "original_{}.png".format(name))
                        save_image(restored, reconstruction_dir / "reconstructed_{}_snr_{}.png".format(name, snr_tag(snr)))
            values = totals / samples
            rows.append({"snr_db": float(snr), "mse": float(values[0]), "psnr_db": float(values[1]), "ssim": float(values[2]), "lpips": float(values[3]), "sample_count": samples})
    write_rows(rows, output_dir / "metrics.csv")
    save_json({"dataset": "CIFAR-10 official test split", "checkpoint": str(Path(args.checkpoint)), "test_count": len(dataset), "test_snrs": list(args.test_snrs), "noise_repeats": args.noise_repeats, "lpips_backbone": args.lpips_backbone}, output_dir / "summary.json")
    print("评估完成，结果目录: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    evaluate_model(parse_args())
