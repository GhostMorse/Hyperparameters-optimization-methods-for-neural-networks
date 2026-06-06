"""Чистый baseline: MLP + Adam с фиксированными гиперпараметрами, без подбора."""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..config import CONFIG, device
from ..models import build_model
from ..utils import set_seed
from ..evaluation import eval_model
from ..tracking import BudgetTracker


def _train_one_epoch(model, loader, opt, crit, task):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        t = y if task == 'classification' else y.view_as(out)
        loss = crit(out, t)
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def train_adam_baseline(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_baseline"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    hp = {"h": cfg["hidden_dim"], "dropout": cfg["dropout"]}
    model = build_model(meta, hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    tracker = BudgetTracker()
    for _ in tqdm(range(cfg["epochs"]), desc="Adam baseline", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        # Для baseline 1 эпоха = 1 "partial" eval, в сумме всё обучение = 1 full eval
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="epoch")
    # В конце засчитаем целиком как 1 full eval (типичная единица бюджета HPO)
    tracker._n_full = 1  # одно полное обучение
    method_cfg = {"h": cfg["hidden_dim"], "lr": cfg["lr"], "dropout": cfg["dropout"]}
    return model, tracker.trajectory, method_cfg
