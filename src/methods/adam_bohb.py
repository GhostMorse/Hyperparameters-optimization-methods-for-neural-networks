"""Adam + BOHB.

BOHB = Hyperband + TPE: семплер выбирает перспективные конфиги (не случайно),
pruner отсеивает плохие на малом бюджете. Идиоматично — TPE sampler + Hyperband
pruner в Optuna.
"""

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


def train_adam_bohb(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_bohb"]
    space = CONFIG["hpo_space"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    tracker = BudgetTracker()

    def objective(trial):
        h = trial.suggest_int('h', *space["h_range"])
        lr = trial.suggest_float('lr', *space["lr_range"], log=True)
        dropout = trial.suggest_float('dropout', *space["dropout_range"])
        model = build_model(meta, {"h": h, "dropout": dropout}).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        last_train = 0.0
        for epoch in range(cfg["max_resource"]):
            last_train = _train_one_epoch(model, loaders["train"], opt, crit, task)
            val = eval_model(model, loaders["val"], task,
                             meta.get("y_mean"), meta.get("y_std"))
            tracker.add_partial_eval(epochs=1.0)
            tracker.log(train_fitness=last_train, val_loss=val["loss"],
                        note=f"trial={trial.number} epoch={epoch + 1}")
            trial.report(val["loss"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val["loss"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=cfg["min_resource"],
        max_resource=cfg["max_resource"],
        reduction_factor=cfg["reduction_factor"],
    )
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)

    best = study.best_params
    model = build_model(meta, {"h": best['h'], "dropout": best['dropout']}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best['lr'])
    for _ in tqdm(range(cfg["final_epochs"]), desc="BOHB final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1
    return model, tracker.trajectory, best
