"""Optuna (TPE) подбирает h, DE обучает веса. Black-box постановка."""

import time
import numpy as np
import optuna

from ..config import CONFIG, device
from ..models import MLP
from ..utils import set_seed, set_weights_from_vec, n_params
from ..evaluation import eval_fitness_on_val
from ..tracking import BudgetTracker
from .de import differential_evolution


def train_optuna_de(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["optuna_de"]
    task = meta["task_type"]
    tracker = BudgetTracker()
    deadline = time.time() + cfg.get("max_seconds_per_run", 1200)

    def objective(trial):
        if time.time() > deadline:
            raise optuna.TrialPruned()
        h = trial.suggest_int('h', *cfg["h_range"])
        model = MLP(meta["in_dim"], h, meta["out_dim"]).to(device)
        dim_w = n_params(model)
        rng = np.random.default_rng(seed + trial.number)
        low, high = cfg["bounds"]

        def fitness(vec):
            return eval_fitness_on_val(vec, model, loaders["val"], task,
                                       meta.get("y_mean"), meta.get("y_std"))

        _, best_f, _ = differential_evolution(
            fitness, [low] * dim_w, [high] * dim_w,
            n_pop=cfg["inner_pop"], n_gen=cfg["inner_gen"],
            F=cfg["F"], Cr=cfg["Cr"], rng=rng,
            desc=f"Optuna-DE trial {trial.number}",
            tracker=tracker, val_eval_fn=fitness, log_every=5,
            deadline=deadline,
        )
        return best_f

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    # n_trials с timeout по wall-clock — Optuna умеет
    study.optimize(objective, n_trials=cfg["n_trials"],
                   timeout=max(60, deadline - time.time()),
                   show_progress_bar=False)

    # Финальный DE с лучшим h, если есть время
    if not study.trials or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        # Никаких удачных trial — берём дефолт h
        best_h = (cfg["h_range"][0] + cfg["h_range"][1]) // 2
    else:
        best_h = study.best_params['h']

    model = MLP(meta["in_dim"], best_h, meta["out_dim"]).to(device)
    dim_w = n_params(model)
    rng = np.random.default_rng(seed + 10_000)
    low, high = cfg["bounds"]

    def fitness(vec):
        return eval_fitness_on_val(vec, model, loaders["val"], task,
                                   meta.get("y_mean"), meta.get("y_std"))

    # Финальный run — считаем оставшееся время
    remaining = max(60.0, deadline - time.time())
    final_deadline = time.time() + remaining
    best_w, best_f, _ = differential_evolution(
        fitness, [low] * dim_w, [high] * dim_w,
        n_pop=cfg["final_pop"], n_gen=cfg["final_gen"],
        F=cfg["F"], Cr=cfg["Cr"], rng=rng, desc="Optuna-DE final",
        tracker=tracker, val_eval_fn=fitness, log_every=1,
        deadline=final_deadline,
    )
    set_weights_from_vec(model, best_w)
    method_cfg = {"h": best_h, "F": cfg["F"], "Cr": cfg["Cr"]}
    return model, tracker.trajectory, method_cfg
