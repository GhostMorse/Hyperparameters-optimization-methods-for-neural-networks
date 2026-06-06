"""Фабрика моделей: создаёт MLP или CNN в зависимости от meta['model_type'].

Это позволяет всем методам обучения быть model-agnostic: они вызывают
build_model(meta, hp), а конкретный класс выбирается по meta.
"""

from .mlp import MLP
from .cnn import SimpleCNN


def build_model(meta, hp):
    """Создаёт модель на основе meta + hp.

    hp — словарь с ключами 'h' (hidden_d) и 'dropout' (опционально).
    Для tabular: meta['model_type'] == 'mlp', нужны 'in_dim' и 'out_dim'.
    Для image:   meta['model_type'] == 'cnn', нужны 'in_channels', 'image_size', 'num_classes'.
    """
    h = hp["h"]
    dropout = hp.get("dropout", 0.2)
    mtype = meta.get("model_type", "mlp")

    if mtype == "mlp":
        return MLP(meta["in_dim"], h, meta["out_dim"], dropout)
    elif mtype == "cnn":
        return SimpleCNN(meta["in_channels"], meta["image_size"],
                         h, meta["num_classes"], dropout)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")


def n_params_for_meta(meta, h_default=256):
    """Утилита: число параметров модели с дефолтным h для baseline-DE-методов.

    DE-методы оптимизируют веса напрямую — на CNN они не работают
    (~миллионы весов), но эта функция нужна для backward-compat.
    """
    from ..utils import n_params
    hp = {"h": h_default, "dropout": 0.2}
    model = build_model(meta, hp)
    return n_params(model)
