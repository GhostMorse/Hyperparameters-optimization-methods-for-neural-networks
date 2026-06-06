import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results_fixed"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 2024, 7, 999]

CONFIG = {
    "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    "batch_size": 64,
    "num_workers": 0,

    "adam_baseline": {
        "hidden_dim": 256,
        "lr": 1e-3,
        "dropout": 0.2,
        "epochs": 50,
    },

    "adam_optuna": {
        "n_trials": 15,
        "search_epochs": 10,
        "final_epochs": 20,
        "h_range": (128, 512),
        "lr_range": (1e-4, 5e-3),
        "dropout_range": (0.0, 0.5),
    },

    # DE-методы — это дополнительный подраздел диплома (не основной).
    # Параметры подобраны так, чтобы DE сходился, но не съедал десятки часов CPU.
    # Если хочется "честных" больших DE-параметров (как в преддипломе),
    # см. ветку de_full_params ниже.
    "baseline_de": {
        "hidden_dim": 256,
        "n_pop": 40,        # было 80
        "n_gen": 60,        # было 100
        "F": 0.6,
        "Cr": 0.9,
        "bounds": (-0.5, 0.5),
        "max_seconds_per_run": 600,    # таймаут — 10 минут на (датасет, seed)
    },

    "optuna_de": {
        "n_trials": 5,         # было 8
        "inner_pop": 25,       # было 40
        "inner_gen": 20,       # было 30
        "final_pop": 40,       # было 80
        "final_gen": 60,       # было 100
        "F": 0.6,
        "Cr": 0.9,
        "bounds": (-0.5, 0.5),
        "h_range": (128, 400),
        "max_seconds_per_run": 1200,    # 20 минут
    },

    "de_de_hybrid": {
        "outer_pop": 6,        # было 10
        "outer_gen": 6,        # было 10
        "inner_pop": 20,       # было 30
        "inner_gen": 20,       # было 30
        "bounds": (-0.5, 0.5),
        "outer_bounds": ([128, 0.4, 0.7], [400, 0.8, 0.95]),
        "max_seconds_per_run": 1800,    # 30 минут
    },

    # === Search-space для всех HPO-методов (унифицирован) ===
    # h: int [128..512], lr: log [1e-4..5e-3], dropout: float [0..0.5]
    "hpo_space": {
        "h_range": (128, 512),
        "lr_range": (1e-4, 5e-3),
        "dropout_range": (0.0, 0.5),
    },

    # Adam + Random Search
    "adam_random": {
        "n_trials": 15,
        "search_epochs": 10,
        "final_epochs": 20,
    },

    # Adam + Gaussian Process BO (BoTorch)
    "adam_gp_bo": {
        "n_init": 5,
        "n_trials": 15,
        "search_epochs": 10,
        "final_epochs": 20,
    },

    # Adam + Hyperband
    "adam_hyperband": {
        "min_resource": 1,
        "max_resource": 20,    # max epochs per config
        "reduction_factor": 3,
        "final_epochs": 20,
    },

    # Adam + BOHB (Hyperband + TPE через optuna)
    "adam_bohb": {
        "n_trials": 30,         # более агрессивно, потому что Hyperband отсеивает дешёвых
        "min_resource": 1,
        "max_resource": 20,
        "reduction_factor": 3,
        "final_epochs": 20,
    },

    # Adam + MFBO (BoTorch, multi-fidelity GP)
    "adam_mfbo": {
        "n_init": 5,
        "n_trials": 15,
        "fidelities": [1, 5, 20],   # epochs as fidelity levels
        "final_epochs": 20,
    },
}
