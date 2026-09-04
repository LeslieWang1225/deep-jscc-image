"""任务四入口：运行潜特征通道数消融实验。"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练" / "train.py"


def parse_args():
    parser = argparse.ArgumentParser(description="运行任务四的潜特征通道数实验")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "kodak"))
    parser.add_argument("--output-dir", default=str(Path(tempfile.gettempdir()) / "deep-jscc-image" / "04_潜特征通道数消融"))
    parser.add_argument("--split-file", default=str(Path(tempfile.gettempdir()) / "deep-jscc-image" / "02_数据划分与基线模型训练" / "data_split.json"))
    parser.add_argument(
        "--latent-channels",
        choices=[4, 8, 32],
        type=int,
        nargs="+",
        default=[4, 8, 32],
        help="要运行的潜特征通道数；省略时依次运行 4、8、32 三组。",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-snrs", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    return parser.parse_args()


def main():
    args = parse_args()
    for latent_channels in args.latent_channels:
        command = [
            sys.executable, str(TRAIN_SCRIPT), "--data-dir", args.data_dir,
            "--output-dir", args.output_dir,
            "--experiment-name", f"latent_{latent_channels}",
            "--split-file", args.split_file,
            "--latent-channels", str(latent_channels),
            "--epochs", str(args.epochs), "--device", args.device, "--train-snrs",
        ] + [str(value) for value in args.train_snrs]
        print("运行: {}".format(" ".join(command)), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
