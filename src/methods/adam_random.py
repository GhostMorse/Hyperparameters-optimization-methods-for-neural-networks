"""Adam + Random Search по гиперпараметрам.

Простой baseline для всех HPO-методов: семплируем случайные конфиги, обучаем
search_epochs эпох, выбираем лучший по val, доучиваем final_epochs.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..config import CONFIG, device
from ..models import MLP, build_model
from ..utils import set_seed
from ..evaluation import eval_model
from ..tracking import BudgetTracker
from .adam_baseline import _train_one_epoch


def _sample_config(rng):
    space = CONFIG["hpo_space"]
    h = int(rng.integers(*space["h_range"]))
    lr = float(np.exp(rng.uniform(np.log(space["lr_range"][0]),
                                   np.log(space["lr_range"][1]))))
    dropout = float(rng.uniform(*space["dropout_range"]))
    return {"h": h, "lr": lr, "dropout": dropout}


def _train_eval(cfg_hpo, search_epochs, loaders, meta, crit, task):
    model = build_model(meta, {"h": cfg_hpo["h"], "dropout": cfg_hpo["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_hpo["lr"])
    last_train = 0.0
    for _ in range(search_epochs):
        last_train = _train_one_epoch(model, loaders["train"], opt, crit, task)
    val = eval_model(model, loaders["val"], task,
                     meta.get("y_mean"), meta.get("y_std"))
    return model, last_train, val["loss"]


def train_adam_random(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_random"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    rng = np.random.default_rng(seed)
    tracker = BudgetTracker()

    best = None
    best_loss = float('inf')
    for trial_idx in range(cfg["n_trials"]):
        hp = _sample_config(rng)
        _, train_loss, val_loss = _train_eval(hp, cfg["search_epochs"], loaders, meta, crit, task)
        tracker.add_partial_eval(epochs=cfg["search_epochs"])
        tracker.log(train_fitness=train_loss, val_loss=val_loss,
                    note=f"trial={trial_idx} h={hp['h']} lr={hp['lr']:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best = hp

    # Финальное обучение лучшего конфига
    model = build_model(meta, {"h": best["h"], "dropout": best["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best["lr"])
    for _ in tqdm(range(cfg["final_epochs"]), desc="RS final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1
    return model, tracker.trajectory, best
