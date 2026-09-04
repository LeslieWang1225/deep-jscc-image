"""训练轻量图像语义通信模型。"""

import argparse
import csv
import random
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys

TASK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_ROOT))

from 公共模块.experiment_config import RANDOM_TRAIN_SNRS
from 公共模块.experiment_utils import (
    load_or_create_split,
    resolve_device,
    save_json,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "kodak"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "数据划分与基线模型训练"


class KodakDataset(Dataset):
    """读取Kodak图像。"""

    def __init__(self, img_dir, transform=None, image_names=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        if image_names is None:
            image_names = sorted(path.name for path in self.img_dir.glob("*.png"))
        self.img_names = list(image_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path = self.img_dir / self.img_names[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class SemanticCommSystem(nn.Module):
    """卷积编码器、AWGN信道和卷积解码器。"""

    def __init__(self, latent_channels=16):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def channel_forward(self, features, snr_db):
        """模拟AWGN信道。"""
        if snr_db is None:
            return features
        signal_power = torch.mean(
            features.pow(2), dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-12)
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_std = torch.sqrt(signal_power / snr_linear)
        return features + torch.randn_like(features) * noise_std

    def forward(self, images, snr_db):
        encoded = self.encoder(images)
        received = self.channel_forward(encoded, snr_db)
        return self.decoder(received)


def parse_args():
    parser = argparse.ArgumentParser(description="训练图像语义通信模型")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument(
        "--split-file", default=str(DEFAULT_OUTPUT_DIR / "data_split.json")
    )
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=18)
    parser.add_argument("--train-snrs", type=float, nargs="+", default=RANDOM_TRAIN_SNRS)
    parser.add_argument("--fixed-train-snr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def write_history(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "loss", "train_snr", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)


def train_model(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    split = load_or_create_split(
        args.data_dir, args.split_file, args.train_count, args.seed
    )
    transform = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor()]
    )
    dataset = KodakDataset(args.data_dir, transform, split["train"])
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )

    model = SemanticCommSystem(args.latent_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    start_time = time.perf_counter()

    print("设备: {}".format(device))
    print("实验: {}，训练图像: {}".format(args.experiment_name, len(dataset)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        used_snrs = []
        for images in dataloader:
            images = images.to(device)
            current_snr = (
                args.fixed_train_snr
                if args.fixed_train_snr is not None
                else random.choice(args.train_snrs)
            )
            optimizer.zero_grad()
            reconstructed = model(images, current_snr)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            used_snrs.append(float(current_snr))

        elapsed = time.perf_counter() - start_time
        average_loss = total_loss / len(dataloader)
        history.append(
            {
                "epoch": epoch,
                "loss": average_loss,
                "train_snr": sum(used_snrs) / len(used_snrs),
                "elapsed_seconds": elapsed,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                "Epoch [{}/{}] Loss: {:.6f} Time: {:.1f}s".format(
                    epoch, args.epochs, average_loss, elapsed
                )
            )

    total_seconds = time.perf_counter() - start_time
    config = {
        "experiment_name": args.experiment_name,
        "data_dir": str(Path(args.data_dir)),
        "split_file": str(Path(args.split_file)),
        "latent_channels": args.latent_channels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "train_count": len(split["train"]),
        "test_count": len(split["test"]),
        "train_snrs": list(args.train_snrs),
        "fixed_train_snr": args.fixed_train_snr,
        "seed": args.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": total_seconds,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "split": split,
    }
    torch.save(checkpoint, experiment_dir / "model.pth")
    write_history(history, experiment_dir / "training_history.csv")
    save_json(config, experiment_dir / "training_config.json")
    print("训练完成，结果目录: {}".format(experiment_dir.resolve()))
    return experiment_dir


def main():
    train_model(parse_args())


if __name__ == "__main__":
    main()
