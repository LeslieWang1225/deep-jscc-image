"""创建或校验任务二固定训练/测试划分。"""

import sys
import tempfile
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parents[1]
sys.path.insert(0, str(TASK_ROOT))

from 公共模块.experiment_utils import load_or_create_split


def main():
    data_dir = PROJECT_ROOT / "实验运行代码" / "实验数据集" / "kodak"
    split_file = Path(tempfile.gettempdir()) / "deep-jscc-image" / "数据划分与基线模型训练" / "data_split.json"
    split = load_or_create_split(data_dir, split_file, train_count=18, seed=42)
    print(f"训练集 {len(split['train'])} 张，测试集 {len(split['test'])} 张：{split_file}")


if __name__ == "__main__":
    main()
