"""任务三：在 CIFAR-10 官方训练集上训练 Deep JSCC 模型。"""

import argparse
import csv
import random
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
TASK2_CODE = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练"
sys.path.insert(0, str(TASK2_CODE))

from train import SemanticCommSystem
from 公共模块.experiment_config import RANDOM_TRAIN_SNRS
from 公共模块.experiment_utils import resolve_device, save_json, set_seed


DEFAULT_DATA_ROOT = PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "cifar10"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "03_CIFAR-10数据集扩展"


def parse_args():
    parser = argparse.ArgumentParser(description="训练 CIFAR-10 图像语义通信模型")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment-name", default="cifar10_random_snr")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--train-snrs", type=float, nargs="+", default=RANDOM_TRAIN_SNRS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def select_samples(dataset, maximum, seed):
    if maximum is None or maximum >= len(dataset):
        return dataset, list(range(len(dataset)))
    if maximum <= 0:
        raise ValueError("max_train_samples必须为正数")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:maximum].tolist()
    return Subset(dataset, indices), indices


def write_history(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "train_snr", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def train_model(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor()])
    full_dataset = datasets.CIFAR10(args.data_root, train=True, transform=transform, download=True)
    dataset, indices = select_samples(full_dataset, args.max_train_samples, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed))
    model = SemanticCommSystem(args.latent_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        used_snrs = []
        for images, _ in loader:
            images = images.to(device)
            snr = random.choice(args.train_snrs)
            optimizer.zero_grad()
            loss = criterion(model(images, snr), images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            used_snrs.append(float(snr))
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "loss": total_loss / len(loader), "train_snr": sum(used_snrs) / len(used_snrs), "elapsed_seconds": elapsed})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print("Epoch [{}/{}] Loss: {:.6f} Time: {:.1f}s".format(epoch, args.epochs, history[-1]["loss"], elapsed))
    config = {"dataset": "CIFAR-10 official training split", "data_root": str(Path(args.data_root)), "experiment_name": args.experiment_name, "latent_channels": args.latent_channels, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "resolution": args.resolution, "train_count": len(dataset), "train_snrs": list(args.train_snrs), "seed": args.seed, "device": str(device), "training_seconds": time.perf_counter() - start, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "max_train_samples": args.max_train_samples}
    torch.save({"model_state_dict": model.state_dict(), "config": config, "cifar10_train_indices": indices}, experiment_dir / "model.pth")
    write_history(history, experiment_dir / "training_history.csv")
    save_json(config, experiment_dir / "training_config.json")
    print("训练完成，结果目录: {}".format(experiment_dir.resolve()))


if __name__ == "__main__":
    train_model(parse_args())
