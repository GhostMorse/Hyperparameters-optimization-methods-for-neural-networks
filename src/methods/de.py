"""Дифференциальная эволюция (DE/best/1/bin) с поддержкой логирования траектории и таймаута."""

import time
import numpy as np
from tqdm.auto import tqdm


def differential_evolution(func, bounds_low, bounds_high, n_pop, n_gen,
                           F=0.6, Cr=0.9, rng=None, desc="DE",
                           tracker=None, val_eval_fn=None, log_every=1,
                           deadline=None):
    """Стандартный DE/best/1/bin.

    Если переданы `tracker` и `val_eval_fn`, после каждых `log_every` поколений
    логируется точка траектории.

    Если задан `deadline` (UNIX-время в секундах), DE прерывается, как только
    время превышено — возвращает текущий best. Это защита от зависших циклов.
    """
    rng = rng or np.random.default_rng()
    bounds_low = np.asarray(bounds_low, dtype=np.float64)
    bounds_high = np.asarray(bounds_high, dtype=np.float64)
    dim = len(bounds_low)

    pop = rng.uniform(bounds_low, bounds_high, size=(n_pop, dim))
    fits = np.array([func(ind) for ind in pop])
    best_idx = int(np.argmin(fits))
    best_ind = pop[best_idx].copy()
    best_fit = float(fits[best_idx])

    history = []
    timed_out = False
    pbar = tqdm(range(n_gen), desc=desc, leave=False)
    for gen_idx in pbar:
        if deadline is not None and time.time() > deadline:
            timed_out = True
            pbar.set_postfix({"best": f"{best_fit:.4f}", "TIMEOUT": True})
            break
        for i in range(n_pop):
            idxs = [j for j in range(n_pop) if j != i]
            r1, r2 = rng.choice(idxs, 2, replace=False)
            mutant = np.clip(best_ind + F * (pop[r1] - pop[r2]), bounds_low, bounds_high)
            cross_mask = rng.random(dim) < Cr
            if not cross_mask.any():
                cross_mask[rng.integers(dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            f_trial = func(trial)
            if f_trial < fits[i]:
                fits[i] = f_trial
                pop[i] = trial
                if f_trial < best_fit:
                    best_fit = float(f_trial)
                    best_ind = trial.copy()
        history.append(best_fit)
        pbar.set_postfix({"best": f"{best_fit:.4f}"})

        if tracker is not None and ((gen_idx + 1) % log_every == 0 or gen_idx == n_gen - 1):
            tracker.add_full_eval(epochs=1.0)
            val_loss = val_eval_fn(best_ind) if val_eval_fn is not None else best_fit
            note = f"gen={gen_idx + 1}" + (" [TIMEOUT]" if timed_out else "")
            tracker.log(train_fitness=best_fit, val_loss=val_loss, note=note)

    return best_ind, best_fit, history
