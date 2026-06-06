"""Baseline DE: эволюционная оптимизация весов MLP с фиксированной архитектурой.

Постановка: black-box. Fitness = val-loss (val играет роль внешнего оценщика).
Это методологическое допущение, явно проговариваемое в тексте диплома.
"""

import time
import numpy as np

from ..config import CONFIG, device
from ..models import MLP
from ..utils import set_seed, set_weights_from_vec, n_params
from ..evaluation import eval_fitness_on_val
from ..tracking import BudgetTracker
from .de import differential_evolution


def train_baseline_de(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["baseline_de"]
    task = meta["task_type"]
    rng = np.random.default_rng(seed)

    model = MLP(meta["in_dim"], cfg["hidden_dim"], meta["out_dim"]).to(device)
    dim_w = n_params(model)
    tracker = BudgetTracker()
    deadline = time.time() + cfg.get("max_seconds_per_run", 600)

    def fitness(vec):
        return eval_fitness_on_val(vec, model, loaders["val"], task,
                                   meta.get("y_mean"), meta.get("y_std"))

    val_eval_fn = fitness

    low, high = cfg["bounds"]
    best_w, best_f, _ = differential_evolution(
        fitness, [low] * dim_w, [high] * dim_w,
        n_pop=cfg["n_pop"], n_gen=cfg["n_gen"],
        F=cfg["F"], Cr=cfg["Cr"], rng=rng, desc="Baseline DE",
        tracker=tracker, val_eval_fn=val_eval_fn, log_every=1,
        deadline=deadline,
    )
    set_weights_from_vec(model, best_w)
    method_cfg = {"h": cfg["hidden_dim"], "F": cfg["F"], "Cr": cfg["Cr"]}
    return model, tracker.trajectory, method_cfg
