"""Adam + подбор гиперпараметров (h, lr, dropout) с помощью Optuna/TPE."""

import torch
import torch.nn as nn
import optuna
from tqdm.auto import tqdm

from ..config import CONFIG, device
from ..models import MLP, build_model
from ..utils import set_seed
from ..evaluation import eval_model
from ..tracking import BudgetTracker
from .adam_baseline import _train_one_epoch


def train_adam_optuna(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_optuna"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    tracker = BudgetTracker()

    def objective(trial):
        h = trial.suggest_int('h', *cfg["h_range"])
        lr = trial.suggest_float('lr', *cfg["lr_range"], log=True)
        dropout = trial.suggest_float('dropout', *cfg["dropout_range"])
        model = build_model(meta, {"h": h, "dropout": dropout}).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        last_train = 0.0
        for _ in range(cfg["search_epochs"]):
            last_train = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=cfg["search_epochs"])
        tracker.log(train_fitness=last_train, val_loss=val["loss"],
                    note=f"trial={trial.number}")
        return val["loss"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)

    best = study.best_params
    model = build_model(meta, {"h": best['h'], "dropout": best['dropout']}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best['lr'])
    for _ in tqdm(range(cfg["final_epochs"]), desc="Adam+TPE final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1  # финальное обучение = 1 full eval
    return model, tracker.trajectory, best
