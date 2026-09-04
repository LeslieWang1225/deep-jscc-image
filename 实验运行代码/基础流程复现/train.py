"""任务一：按教师基础代码复现 Deep JSCC（不划分训练集与测试集）。"""

import argparse
import csv
import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "kodak"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "基础流程复现"
TRAIN_SNRS = [-10, -5, 0, 5, 10]


class KodakDataset(Dataset):
    """读取全部 Kodak PNG 图像；本任务不做训练/测试划分。"""

    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_names = sorted(path.name for path in self.img_dir.glob("*.png"))
        if not self.img_names:
            raise FileNotFoundError(f"未找到 PNG 图像：{self.img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image = Image.open(self.img_dir / self.img_names[index]).convert("RGB")
        return self.transform(image) if self.transform else image


class SemanticCommSystem(nn.Module):
    """教师给定的卷积编码器—AWGN 信道—卷积解码器。"""

    def __init__(self, latent_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def channel_forward(self, features, snr_db):
        signal_power = torch.mean(features.pow(2), dim=(1, 2, 3), keepdim=True)
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_std = torch.sqrt(signal_power.clamp_min(1e-12) / snr_linear)
        return features + torch.randn_like(features) * noise_std

    def forward(self, images, snr_db):
        return self.decoder(self.channel_forward(self.encoder(images), snr_db))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="任务一：不划分数据集的基础复现")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")

    transform = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor()]
    )
    dataset = KodakDataset(args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SemanticCommSystem().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    history = []

    print(f"任务一：使用全部 {len(dataset)} 张 Kodak 图像训练（不划分训练/测试集）")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = model(images, random.choice(TRAIN_SNRS))
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        history.append({"epoch": epoch, "loss": average_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] Loss: {average_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "latent_channels": 16, "split": None},
        output_dir / "semantic_model.pth",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"模型和训练记录已保存到：{output_dir}")


if __name__ == "__main__":
    main()
