"""Adam + Gaussian Process Bayesian Optimization (BoTorch).

Используем SingleTaskGP + qExpectedImprovement. Поиск идёт в нормализованном
пространстве [0,1]^3 (h, lr, dropout); затем разнормализуется обратно.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..config import CONFIG, device
from ..models import MLP, build_model
from ..utils import set_seed
from ..evaluation import eval_model
from ..tracking import BudgetTracker
from .adam_baseline import _train_one_epoch


def _denorm(x_unit, space):
    """[0,1]^3 -> (h, lr, dropout)."""
    h_lo, h_hi = space["h_range"]
    lr_lo, lr_hi = space["lr_range"]
    do_lo, do_hi = space["dropout_range"]
    h = int(round(h_lo + x_unit[0] * (h_hi - h_lo)))
    lr = float(np.exp(np.log(lr_lo) + x_unit[1] * (np.log(lr_hi) - np.log(lr_lo))))
    dropout = float(do_lo + x_unit[2] * (do_hi - do_lo))
    return {"h": h, "lr": lr, "dropout": dropout}


def _evaluate_config(cfg_hpo, search_epochs, loaders, meta, crit, task):
    model = build_model(meta, {"h": cfg_hpo["h"], "dropout": cfg_hpo["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_hpo["lr"])
    last_train = 0.0
    for _ in range(search_epochs):
        last_train = _train_one_epoch(model, loaders["train"], opt, crit, task)
    val = eval_model(model, loaders["val"], task,
                     meta.get("y_mean"), meta.get("y_std"))
    return last_train, val["loss"]


def train_adam_gp_bo(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_gp_bo"]
    space = CONFIG["hpo_space"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    rng = np.random.default_rng(seed)
    tracker = BudgetTracker()

    bounds = torch.stack([torch.zeros(3, dtype=torch.double),
                          torch.ones(3, dtype=torch.double)])

    # 1. Initial random points
    X = []
    Y = []
    for trial_idx in range(cfg["n_init"]):
        x_unit = rng.uniform(0, 1, size=3)
        hp = _denorm(x_unit, space)
        train_loss, val_loss = _evaluate_config(hp, cfg["search_epochs"], loaders, meta, crit, task)
        X.append(x_unit)
        Y.append(-val_loss)  # BoTorch максимизирует, мы минимизируем
        tracker.add_partial_eval(epochs=cfg["search_epochs"])
        tracker.log(train_fitness=train_loss, val_loss=val_loss,
                    note=f"init={trial_idx} h={hp['h']}")

    X_t = torch.tensor(np.array(X), dtype=torch.double)
    Y_t = torch.tensor(np.array(Y), dtype=torch.double).unsqueeze(-1)

    # 2. BO loop
    for trial_idx in range(cfg["n_init"], cfg["n_trials"]):
        try:
            gp = SingleTaskGP(X_t, Y_t)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acq = qLogExpectedImprovement(gp, best_f=Y_t.max())
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=5, raw_samples=64)
            x_unit = cand.squeeze().cpu().numpy()
        except Exception as e:
            # Fallback на случайную точку, если GP/acq упал
            x_unit = rng.uniform(0, 1, size=3)
            tracker.log(train_fitness=None, val_loss=float('nan'),
                        note=f"acq_fail: {type(e).__name__}")

        hp = _denorm(x_unit, space)
        train_loss, val_loss = _evaluate_config(hp, cfg["search_epochs"], loaders, meta, crit, task)
        tracker.add_partial_eval(epochs=cfg["search_epochs"])
        tracker.log(train_fitness=train_loss, val_loss=val_loss,
                    note=f"bo={trial_idx} h={hp['h']}")

        X_t = torch.cat([X_t, torch.tensor(x_unit, dtype=torch.double).unsqueeze(0)])
        Y_t = torch.cat([Y_t, torch.tensor([[-val_loss]], dtype=torch.double)])

    # 3. Лучший конфиг и финальное обучение
    best_idx = int(Y_t.argmax().item())
    best_hp = _denorm(X_t[best_idx].cpu().numpy(), space)

    model = build_model(meta, {"h": best_hp["h"], "dropout": best_hp["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_hp["lr"])
    for _ in tqdm(range(cfg["final_epochs"]), desc="GP-BO final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1
    return model, tracker.trajectory, best_hp
