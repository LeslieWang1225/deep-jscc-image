"""任务二：实验矩阵与结果目录映射。"""

TEST_SNRS = [-10, -5, 0, 5, 10]
RANDOM_TRAIN_SNRS = [-10, -5, 0, 5, 10]

EXPERIMENTS = [
    {
        "name": "baseline",
        "result_task": "数据划分与基线模型训练",
        "label": "基线模型 C=16（随机SNR）",
        "latent_channels": 16,
        "train_snrs": RANDOM_TRAIN_SNRS,
        "fixed_train_snr": None,
    },
    {
        "name": "latent_4",
        "result_task": "潜特征通道数消融",
        "label": "潜特征 C=4（随机SNR）",
        "latent_channels": 4,
        "train_snrs": RANDOM_TRAIN_SNRS,
        "fixed_train_snr": None,
    },
    {
        "name": "latent_8",
        "result_task": "潜特征通道数消融",
        "label": "潜特征 C=8（随机SNR）",
        "latent_channels": 8,
        "train_snrs": RANDOM_TRAIN_SNRS,
        "fixed_train_snr": None,
    },
    {
        "name": "latent_32",
        "result_task": "潜特征通道数消融",
        "label": "潜特征 C=32（随机SNR）",
        "latent_channels": 32,
        "train_snrs": RANDOM_TRAIN_SNRS,
        "fixed_train_snr": None,
    },
    {
        "name": "fixed_snr_5",
        "result_task": "SNR训练策略对比",
        "label": "C=16（固定5 dB）",
        "latent_channels": 16,
        "train_snrs": [5],
        "fixed_train_snr": 5.0,
    },
]
