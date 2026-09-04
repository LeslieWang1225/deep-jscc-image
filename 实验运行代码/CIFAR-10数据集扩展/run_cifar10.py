"""任务三总入口：训练并评估 CIFAR-10 扩展实验。"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "cifar10"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "deep-jscc-image" / "CIFAR-10数据集扩展"


def parse_args():
    parser = argparse.ArgumentParser(description="运行任务三 CIFAR-10 数据集扩展实验")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def run(command):
    print("运行: {}".format(" ".join(command)), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    experiment_dir = Path(args.output_dir) / "cifar10_random_snr"
    checkpoint = experiment_dir / "model.pth"
    if args.force_train or not checkpoint.exists():
        command = [sys.executable, str(TASK_DIR / "train_cifar10.py"), "--data-root", args.data_root, "--output-dir", args.output_dir, "--epochs", str(args.epochs), "--device", args.device]
        if args.max_train_samples is not None:
            command += ["--max-train-samples", str(args.max_train_samples)]
        run(command)
    command = [sys.executable, str(TASK_DIR / "evaluate_cifar10.py"), "--data-root", args.data_root, "--checkpoint", str(checkpoint), "--output-dir", str(experiment_dir / "评价结果"), "--device", args.device, "--noise-repeats", str(args.noise_repeats)]
    if args.max_test_samples is not None:
        command += ["--max-test-samples", str(args.max_test_samples)]
    run(command)
    run([sys.executable, str(TASK_DIR / "plot_cifar10.py"), "--experiment-dir", str(experiment_dir)])


if __name__ == "__main__":
    main()
