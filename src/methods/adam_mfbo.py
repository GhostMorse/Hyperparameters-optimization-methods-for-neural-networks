"""Adam + Multi-Fidelity Bayesian Optimization (BoTorch).

Fidelity = число эпох обучения. Низкая fidelity (1 эпоха) — дёшево, шумно;
высокая (max_epochs) — дорого, точно.

Используем SingleTaskMultiFidelityGP + qMultiFidelityKnowledgeGradient.
Это ключевой метод диплома. Если основная схема падает (BoTorch иногда капризный),
есть fallback на упрощённую multi-fidelity через optuna pruners.
"""

import warnings
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


def _denorm_hp(x_unit, space):
    h_lo, h_hi = space["h_range"]
    lr_lo, lr_hi = space["lr_range"]
    do_lo, do_hi = space["dropout_range"]
    h = int(round(h_lo + x_unit[0] * (h_hi - h_lo)))
    lr = float(np.exp(np.log(lr_lo) + x_unit[1] * (np.log(lr_hi) - np.log(lr_lo))))
    dropout = float(do_lo + x_unit[2] * (do_hi - do_lo))
    return {"h": h, "lr": lr, "dropout": dropout}


def _evaluate_at_fidelity(hp, n_epochs, loaders, meta, crit, task):
    model = build_model(meta, {"h": hp["h"], "dropout": hp["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    last_train = 0.0
    for _ in range(n_epochs):
        last_train = _train_one_epoch(model, loaders["train"], opt, crit, task)
    val = eval_model(model, loaders["val"], task,
                     meta.get("y_mean"), meta.get("y_std"))
    return last_train, val["loss"]


def train_adam_mfbo(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["adam_mfbo"]
    space = CONFIG["hpo_space"]
    task = meta["task_type"]
    crit = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    rng = np.random.default_rng(seed)
    tracker = BudgetTracker()

    fidelities = cfg["fidelities"]
    max_fid = max(fidelities)

    # Пытаемся импортировать BoTorch MFBO. Если не получится — fallback.
    try:
        from botorch.models import SingleTaskMultiFidelityGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
        from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        from botorch.models.cost import AffineFidelityCostModel
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        botorch_available = True
    except Exception:
        botorch_available = False

    if not botorch_available:
        return _fallback_mfbo(loaders, meta, seed, tracker, cfg, space, crit, task, rng)

    # === BoTorch MFBO ===
    # Точки в R^4: (h_unit, lr_unit, dropout_unit, fidelity_unit)
    # fidelity_unit = epochs / max_fid
    bounds = torch.tensor([[0., 0., 0., min(fidelities) / max_fid],
                           [1., 1., 1., 1.]], dtype=torch.double)
    target_fidelities = {3: 1.0}  # размерность 3 — fidelity, target = 1.0 (max)

    X = []
    Y = []
    # Initial: по одной точке на каждой fidelity, со случайными hp
    for trial_idx in range(cfg["n_init"]):
        x_unit = rng.uniform(0, 1, size=3)
        fid_idx = trial_idx % len(fidelities)
        n_ep = fidelities[fid_idx]
        hp = _denorm_hp(x_unit, space)
        train_loss, val_loss = _evaluate_at_fidelity(hp, n_ep, loaders, meta, crit, task)
        if n_ep == max_fid:
            tracker.add_full_eval(epochs=n_ep)
        else:
            tracker.add_partial_eval(epochs=n_ep)
        tracker.log(train_fitness=train_loss, val_loss=val_loss,
                    note=f"mfbo_init={trial_idx} fid={n_ep} h={hp['h']}")
        X.append(np.append(x_unit, n_ep / max_fid))
        Y.append(-val_loss)

    X_t = torch.tensor(np.array(X), dtype=torch.double)
    Y_t = torch.tensor(np.array(Y), dtype=torch.double).unsqueeze(-1)

    for trial_idx in range(cfg["n_init"], cfg["n_trials"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = SingleTaskMultiFidelityGP(X_t, Y_t, data_fidelities=[3])
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                cost_model = AffineFidelityCostModel(fidelity_weights={3: 1.0}, fixed_cost=0.1)
                cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

                from botorch.acquisition.utils import project_to_target_fidelity

                def project(X):
                    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

                acq = qMultiFidelityKnowledgeGradient(
                    model=gp, num_fantasies=8,
                    current_value=Y_t.max().item(),
                    cost_aware_utility=cost_aware_utility,
                    project=project,
                )
                cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=3, raw_samples=32)
                x_full = cand.squeeze().cpu().numpy()
                x_unit = x_full[:3]
                # Дискретизируем fidelity к ближайшему уровню
                fid_target = x_full[3] * max_fid
                n_ep = min(fidelities, key=lambda v: abs(v - fid_target))
        except Exception as e:
            # Fallback: случайная точка на средней fidelity
            x_unit = rng.uniform(0, 1, size=3)
            n_ep = fidelities[len(fidelities) // 2]
            tracker.log(train_fitness=None, val_loss=float('nan'),
                        note=f"mfbo_acq_fail: {type(e).__name__}")

        hp = _denorm_hp(x_unit, space)
        train_loss, val_loss = _evaluate_at_fidelity(hp, n_ep, loaders, meta, crit, task)
        if n_ep == max_fid:
            tracker.add_full_eval(epochs=n_ep)
        else:
            tracker.add_partial_eval(epochs=n_ep)
        tracker.log(train_fitness=train_loss, val_loss=val_loss,
                    note=f"mfbo={trial_idx} fid={n_ep} h={hp['h']}")
        X_t = torch.cat([X_t, torch.tensor(np.append(x_unit, n_ep / max_fid),
                                            dtype=torch.double).unsqueeze(0)])
        Y_t = torch.cat([Y_t, torch.tensor([[-val_loss]], dtype=torch.double)])

    # Лучший конфиг (предпочитаем оценённые на max_fid)
    high_fid_mask = X_t[:, 3] >= 0.99
    if high_fid_mask.any():
        sub_idx = high_fid_mask.nonzero(as_tuple=True)[0]
        best_idx = sub_idx[Y_t[sub_idx].argmax()].item()
    else:
        best_idx = int(Y_t.argmax().item())
    best_hp = _denorm_hp(X_t[best_idx, :3].cpu().numpy(), space)

    # Финальное обучение
    model = build_model(meta, {"h": best_hp["h"], "dropout": best_hp["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_hp["lr"])
    for _ in tqdm(range(cfg["final_epochs"]), desc="MFBO final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1
    return model, tracker.trajectory, best_hp


def _fallback_mfbo(loaders, meta, seed, tracker, cfg, space, crit, task, rng):
    """Упрощённая multi-fidelity на случай, если BoTorch MFBO падает."""
    fidelities = cfg["fidelities"]
    max_fid = max(fidelities)

    best = None
    best_loss = float('inf')
    # Простая стратегия: все hp оценить на min fidelity, top-k продвинуть на средний и т.д.
    # Имитирует successive halving в BO-стиле через random-sampling hp.
    n_per_round = max(cfg["n_trials"] // len(fidelities), 1)
    candidates = [{"h": int(rng.integers(*space["h_range"])),
                   "lr": float(np.exp(rng.uniform(np.log(space["lr_range"][0]),
                                                  np.log(space["lr_range"][1])))),
                   "dropout": float(rng.uniform(*space["dropout_range"]))}
                  for _ in range(cfg["n_trials"])]
    survivors = candidates
    for fid in fidelities:
        scores = []
        for hp in survivors:
            train_loss, val_loss = _evaluate_at_fidelity(hp, fid, loaders, meta, crit, task)
            if fid == max_fid:
                tracker.add_full_eval(epochs=fid)
            else:
                tracker.add_partial_eval(epochs=fid)
            tracker.log(train_fitness=train_loss, val_loss=val_loss,
                        note=f"fallback_mfbo fid={fid} h={hp['h']}")
            scores.append((val_loss, hp))
            if val_loss < best_loss:
                best_loss = val_loss
                best = hp
        scores.sort(key=lambda s: s[0])
        survivors = [s[1] for s in scores[: max(n_per_round, 1)]]

    model = build_model(meta, {"h": best["h"], "dropout": best["dropout"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best["lr"])
    for _ in tqdm(range(cfg["final_epochs"]), desc="MFBO-fallback final", leave=False):
        train_loss = _train_one_epoch(model, loaders["train"], opt, crit, task)
        val = eval_model(model, loaders["val"], task,
                         meta.get("y_mean"), meta.get("y_std"))
        tracker.add_partial_eval(epochs=1.0)
        tracker.log(train_fitness=train_loss, val_loss=val["loss"], note="final_epoch")
    tracker._n_full += 1
    return model, tracker.trajectory, best
