"""Препроцессинг image-датасетов (CIFAR-10, Fashion-MNIST).

Train/val/test split с фиксированным seed. Нормализация — стандартная для
CV: ToTensor + per-channel standardization (mean/std считаются по train).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ..config import CONFIG


class TransformedDataset(Dataset):
    """Обёртка над torchvision Dataset, применяющая нормализацию."""
    def __init__(self, base_ds, mean, std):
        self.base = base_ds
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # base_ds уже отдаёт тензор в [0, 1] благодаря transform=ToTensor
        x = (x - self.mean) / self.std
        return x, y


def prepare_image_splits(image_loader_fn, seed, split=None):
    """image_loader_fn — callable, возвращает (train_full_ds, test_ds, info_dict).

    info_dict должен содержать:
      - in_channels (int)
      - image_size (int)
      - num_classes (int)
    """
    split = split or CONFIG["split"]
    train_full, test_ds, info = image_loader_fn()

    n_total = len(train_full)
    val_frac_of_train = split["val"] / (split["train"] + split["val"])
    n_val = int(round(n_total * val_frac_of_train))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)

    # Per-channel statistics на train
    n_sample = min(2000, len(train_ds))
    sample_idx = np.random.RandomState(seed).choice(len(train_ds), n_sample, replace=False)
    Xs = torch.stack([train_ds[int(i)][0] for i in sample_idx])  # (N, C, H, W)
    mean = Xs.mean(dim=(0, 2, 3)).tolist()
    std = Xs.std(dim=(0, 2, 3)).tolist()

    train_ds = TransformedDataset(train_ds, mean, std)
    val_ds = TransformedDataset(val_ds, mean, std)
    test_ds = TransformedDataset(test_ds, mean, std)

    meta = {
        "task_type": "classification",
        "model_type": "cnn",
        "in_channels": info["in_channels"],
        "image_size": info["image_size"],
        "num_classes": info["num_classes"],
        "out_dim": info["num_classes"],   # для совместимости с tabular API
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_ds),
        "channel_mean": mean,
        "channel_std": std,
    }
    return train_ds, val_ds, test_ds, meta
