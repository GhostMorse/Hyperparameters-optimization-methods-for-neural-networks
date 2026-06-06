import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CONFIG


def set_seed(seed: int):
    """Фиксирует все источники случайности для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_weights_from_vec(model: nn.Module, vector: np.ndarray):
    """Записывает плоский numpy-вектор в параметры модели (для DE)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        chunk = torch.from_numpy(vector[offset:offset + n].astype(np.float32)).view_as(p.data)
        p.data.copy_(chunk)
        offset += n


def get_weights_vector(model: nn.Module) -> np.ndarray:
    """Извлекает плоский numpy-вектор из параметров модели."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_loader(ds, batch_size, shuffle, seed=None):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=CONFIG["num_workers"],
    )
