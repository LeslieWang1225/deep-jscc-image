"""任务六入口：评估已有模型的扩展指标并绘制汇总图表。"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[1]
TASK2_CODE = PROJECT_ROOT / "03_实验运行代码" / "02_数据划分与基线模型训练"
sys.path.insert(0, str(TASK2_CODE))

from 公共模块.experiment_config import EXPERIMENTS, TEST_SNRS


def parse_args():
    parser = argparse.ArgumentParser(description="运行任务六的指标扩展与结果分析")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "03_实验运行代码" / "实验数据集" / "kodak"))
    parser.add_argument("--output-root", default=str(Path(tempfile.gettempdir()) / "deep-jscc-image"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise-repeats", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument("--lpips-backbone", choices=["alex", "vgg", "squeeze"], default="alex")
    return parser.parse_args()


def run(command):
    print("运行: {}".format(" ".join(command)), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    split_file = output_root / "02_数据划分与基线模型训练" / "data_split.json"

    for experiment in EXPERIMENTS:
        experiment_dir = output_root / experiment["result_task"] / experiment["name"]
        checkpoint = experiment_dir / "model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                "缺少模型权重：{}。请先运行对应的任务训练模型。".format(checkpoint)
            )
        run(
            [
                sys.executable, str(TASK_DIR / "evaluate.py"),
                "--checkpoint", str(checkpoint),
                "--data-dir", args.data_dir,
                "--output-dir", str(experiment_dir / "评价结果"),
                "--split-file", str(split_file),
                "--test-snrs",
            ]
            + [str(value) for value in TEST_SNRS]
            + [
                "--noise-repeats", str(args.noise_repeats),
                "--benchmark-runs", str(args.benchmark_runs),
                "--lpips-backbone", args.lpips_backbone,
                "--device", args.device,
            ]
        )

    run(
        [
            sys.executable, str(TASK_DIR / "plot_results.py"),
            "--output-root", str(output_root),
            "--figure-dir", str(output_root / "06_综合评估与结果分析" / "结果图表"),
        ]
    )


if __name__ == "__main__":
    main()
