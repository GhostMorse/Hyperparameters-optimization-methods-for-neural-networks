"""DE+DE Hybrid: внешний DE ищет (h, F, Cr), внутренний — веса. Black-box постановка."""

import time
import numpy as np

from ..config import CONFIG, device
from ..models import MLP
from ..utils import set_seed, set_weights_from_vec, n_params
from ..evaluation import eval_fitness_on_val
from ..tracking import BudgetTracker
from .de import differential_evolution


def train_de_de_hybrid(loaders, meta, seed):
    set_seed(seed)
    cfg = CONFIG["de_de_hybrid"]
    task = meta["task_type"]
    tracker = BudgetTracker()
    deadline = time.time() + cfg.get("max_seconds_per_run", 1800)

    def outer_fitness(outer_vec):
        if time.time() > deadline:
            return 1e10
        h_val = int(outer_vec[0])
        F_val = float(outer_vec[1])
        Cr_val = float(outer_vec[2])
        model = MLP(meta["in_dim"], h_val, meta["out_dim"]).to(device)
        dim_w = sum(param.numel() for param in model.parameters())
        low, high = cfg["bounds"]
        rng_inner = np.random.default_rng(seed + h_val)

        def fitness(vec):
            return eval_fitness_on_val(vec, model, loaders["val"], task,
                                       meta.get("y_mean"), meta.get("y_std"))

        _, best_f, _ = differential_evolution(
            fitness, [low] * dim_w, [high] * dim_w,
            n_pop=cfg["inner_pop"], n_gen=cfg["inner_gen"],
            F=F_val, Cr=Cr_val, rng=rng_inner, desc="Inner DE",
            tracker=tracker, val_eval_fn=fitness, log_every=5,
            deadline=deadline,
        )
        return best_f

    rng_outer = np.random.default_rng(seed)
    outer_low, outer_high = cfg["outer_bounds"]
    best_outer, _, _ = differential_evolution(
        outer_fitness, outer_low, outer_high,
        n_pop=cfg["outer_pop"], n_gen=cfg["outer_gen"],
        F=0.6, Cr=0.9, rng=rng_outer, desc="Outer DE (meta)",
        deadline=deadline,
    )

    h_best = int(best_outer[0])
    F_best = float(best_outer[1])
    Cr_best = float(best_outer[2])
    model = MLP(meta["in_dim"], h_best, meta["out_dim"]).to(device)
    dim_w = n_params(model)
    low, high = cfg["bounds"]

    def fitness(vec):
        return eval_fitness_on_val(vec, model, loaders["val"], task,
                                   meta.get("y_mean"), meta.get("y_std"))

    rng_final = np.random.default_rng(seed + 99_999)
    final_deadline = time.time() + max(60.0, deadline - time.time())
    best_w, _, _ = differential_evolution(
        fitness, [low] * dim_w, [high] * dim_w,
        n_pop=cfg["inner_pop"], n_gen=cfg["inner_gen"],
        F=F_best, Cr=Cr_best, rng=rng_final, desc="Final inner DE",
        deadline=final_deadline,
    )
    set_weights_from_vec(model, best_w)
    method_cfg = {"h": h_best, "F": F_best, "Cr": Cr_best}
    return model, tracker.trajectory, method_cfg
