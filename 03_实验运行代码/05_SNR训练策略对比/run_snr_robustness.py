"""任务五入口：运行固定 SNR 训练，并用于与随机 SNR 基线比较。"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练" / "train.py"


def parse_args():
    parser = argparse.ArgumentParser(description="运行任务五的固定 SNR 鲁棒性实验")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "kodak"))
    parser.add_argument("--output-dir", default=str(Path(tempfile.gettempdir()) / "deep-jscc-image" / "05_SNR训练策略对比"))
    parser.add_argument("--split-file", default=str(Path(tempfile.gettempdir()) / "deep-jscc-image" / "02_数据划分与基线模型训练" / "data_split.json"))
    parser.add_argument("--experiment-name", default="fixed_snr_5", choices=["fixed_snr_5"])
    parser.add_argument("--latent-channels", type=int, default=16, choices=[16])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-snrs", type=float, nargs="+", default=[5.0])
    parser.add_argument("--fixed-train-snr", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    command = [
        sys.executable, str(TRAIN_SCRIPT), "--data-dir", args.data_dir,
        "--output-dir", args.output_dir, "--experiment-name", args.experiment_name,
        "--split-file", args.split_file, "--latent-channels", str(args.latent_channels),
        "--epochs", str(args.epochs), "--device", args.device, "--train-snrs",
        str(args.fixed_train_snr), "--fixed-train-snr", str(args.fixed_train_snr),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
