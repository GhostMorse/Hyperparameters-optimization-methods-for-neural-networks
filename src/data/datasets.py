"""Реестр датасетов: имя -> dict с описанием.

Структура записи:
  {
    "kind": "tabular" | "image",
    "task": "classification" | "regression",
    "loader": callable -> raw data object (для tabular) или
                          (train_full_ds, test_ds, info) (для image).
  }

Tabular-датасеты обрабатываются через prepare_splits.
Image-датасеты — через prepare_image_splits.
"""

import os
import sklearn.datasets
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo


def _load_cifar10():
    import torchvision
    import torchvision.transforms as T
    transform = T.ToTensor()
    root = os.path.expanduser("~/torch_data")
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    info = {"in_channels": 3, "image_size": 32, "num_classes": 10}
    return train, test, info


def _load_fashion_mnist():
    """Fashion-MNIST (Zalando, AWS S3 / GitHub mirrors). 60k train, 28×28 grayscale."""
    import torchvision
    import torchvision.transforms as T
    transform = T.ToTensor()
    root = os.path.expanduser("~/torch_data")
    train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    info = {"in_channels": 1, "image_size": 28, "num_classes": 10}
    return train, test, info


def _load_higgs(subsample=1_000_000, seed=42):
    """Higgs из OpenML, с опциональной подвыборкой.

    Возвращает объект с .data и .target в формате, совместимом с prepare_splits.
    """
    import numpy as np
    import pandas as pd

    class _HiggsBundle:
        pass

    bundle = _HiggsBundle()
    raw = fetch_openml("higgs", version=2, as_frame=True, parser='auto')
    X = raw.data
    y = raw.target

    if subsample is not None and len(X) > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), subsample, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True) if isinstance(y, pd.Series) else y[idx]

    bundle.data = X
    bundle.target = y
    return bundle


# === Реестр ===

DATASETS = {
    # Tabular small (из преддиплома)
    "wine":    {"kind": "tabular", "task": "classification",
                "loader": lambda: sklearn.datasets.load_wine()},
    "wdbc":    {"kind": "tabular", "task": "classification",
                "loader": lambda: fetch_ucirepo(id=17)},
    "digits":  {"kind": "tabular", "task": "classification",
                "loader": lambda: sklearn.datasets.load_digits()},
    "abalone": {"kind": "tabular", "task": "regression",
                "loader": lambda: fetch_ucirepo(id=1)},
    "kin8nm":  {"kind": "tabular", "task": "regression",
                "loader": lambda: fetch_openml("kin8nm", version=1, as_frame=True)},

    # Большой tabular
    "higgs":   {"kind": "tabular", "task": "classification",
                "loader": lambda: _load_higgs(subsample=1_000_000, seed=42)},

    # Image
    "cifar10":       {"kind": "image", "task": "classification",
                      "loader": _load_cifar10},
    "fashion_mnist": {"kind": "image", "task": "classification",
                      "loader": _load_fashion_mnist},
}


def load_datasets():
    """Совместимость со старым API: возвращает dict в формате
    {name: (task, loader_fn)}, где loader_fn — это lambda для tabular.
    Image-датасеты тут не возвращаются — их обрабатывает prepare_image_splits.
    """
    return {name: (info["task"], info["loader"])
            for name, info in DATASETS.items()
            if info["kind"] == "tabular"}
