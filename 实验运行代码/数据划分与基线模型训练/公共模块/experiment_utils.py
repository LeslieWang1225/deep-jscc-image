"""任务二：数据划分、随机性与设备公共工具。"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    """解析计算设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用CUDA设备")
    return torch.device(device_name)


def save_json(data, path):
    """保存UTF-8 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_or_create_split(data_dir, split_file, train_count, seed):
    """读取或创建固定数据划分。"""
    data_path = Path(data_dir)
    image_names = sorted(path.name for path in data_path.glob("*.png"))
    if not image_names:
        raise FileNotFoundError("数据目录中未找到PNG图像: {}".format(data_path))

    split_path = Path(split_file)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as file:
            split = json.load(file)
        recorded = sorted(split["train"] + split["test"])
        if recorded != image_names:
            raise ValueError("数据集内容与已有划分文件不一致")
        return split

    if train_count <= 0 or train_count >= len(image_names):
        raise ValueError("train_count必须在1到数据集总数减1之间")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(image_names), generator=generator).tolist()
    train_names = sorted(image_names[index] for index in order[:train_count])
    test_names = sorted(image_names[index] for index in order[train_count:])
    split = {
        "seed": seed,
        "train_count": len(train_names),
        "test_count": len(test_names),
        "train": train_names,
        "test": test_names,
    }
    save_json(split, split_path)
    return split
