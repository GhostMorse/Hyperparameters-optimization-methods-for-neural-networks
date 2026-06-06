"""Агрегация результатов: загрузка сохранённых прогонов, mean ± std по seeds.

Возвращает:
  - df: DataFrame с метриками test (loss, sec) + бюджет (time, n_evals).
  - per_method_history: совместимый старый словарь best-so-far по поколениям/эпохам.
  - per_method_trajectory: новый словарь полных траекторий (если есть trajectory.json).
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate

from .config import RESULTS_DIR
from .methods import METHODS
from .data import load_datasets


def _detect_task(ds_path, methods):
    for m in methods:
        m_path = os.path.join(ds_path, m)
        if not os.path.isdir(m_path):
            continue
        for seed_dir in sorted(os.listdir(m_path)):
            meta_path = os.path.join(m_path, seed_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    mm = json.load(f)
                if "accuracy" in mm.get("test_metrics", {}):
                    return "classification"
                if "r2" in mm.get("test_metrics", {}):
                    return "regression"
    return None


def aggregate_results(results_dir=RESULTS_DIR, methods=None, datasets=None):
    methods = methods or list(METHODS.keys())
    # Включаем legacy-имена в скан, чтобы старые папки тоже подхватились
    scan_methods = list(methods) + ["Adam_Optuna"]
    datasets = datasets or list(load_datasets().keys())

    rows = []
    per_method_history = defaultdict(lambda: defaultdict(list))
    per_method_trajectory = defaultdict(lambda: defaultdict(list))

    for ds in datasets:
        ds_path = os.path.join(results_dir, ds)
        if not os.path.isdir(ds_path):
            continue
        task = _detect_task(ds_path, scan_methods)

        for m in scan_methods:
            m_path = os.path.join(ds_path, m)
            if not os.path.isdir(m_path):
                continue
            # Legacy alias: показываем папку Adam_Optuna под именем Adam_TPE
            display_name = "Adam_TPE" if m == "Adam_Optuna" else m
            losses, secondaries = [], []
            times, n_full_evals, n_epochs_eq = [], [], []

            for seed_dir in sorted(os.listdir(m_path)):
                seed_path = os.path.join(m_path, seed_dir)
                meta_path = os.path.join(seed_path, "meta.json")
                hist_path = os.path.join(seed_path, "history.npy")
                traj_path = os.path.join(seed_path, "trajectory.json")
                if not os.path.exists(meta_path):
                    continue

                with open(meta_path) as f:
                    mm = json.load(f)
                losses.append(mm["test_metrics"]["loss"])
                sec = mm["test_metrics"].get("accuracy", mm["test_metrics"].get("r2"))
                secondaries.append(sec)

                summary = mm.get("trajectory_summary", {})
                if summary:
                    times.append(summary.get("total_time"))
                    n_full_evals.append(summary.get("n_evals_full"))
                    n_epochs_eq.append(summary.get("n_epochs_eq"))

                if os.path.exists(hist_path):
                    per_method_history[ds][display_name].append(np.load(hist_path))
                if os.path.exists(traj_path):
                    with open(traj_path) as f:
                        per_method_trajectory[ds][display_name].append(json.load(f))

            if losses:
                row = {
                    "dataset": ds,
                    "method": display_name,
                    "task": task,
                    "n_seeds": len(losses),
                    "loss_mean": float(np.mean(losses)),
                    "loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                    "sec_mean": float(np.mean(secondaries)) if secondaries else None,
                    "sec_std": float(np.std(secondaries, ddof=1)) if len(secondaries) > 1 else 0.0,
                }
                if times and any(t is not None for t in times):
                    valid_times = [t for t in times if t is not None]
                    row["time_mean"] = float(np.mean(valid_times))
                    row["time_std"] = float(np.std(valid_times, ddof=1)) if len(valid_times) > 1 else 0.0
                if n_full_evals and any(n is not None for n in n_full_evals):
                    valid = [n for n in n_full_evals if n is not None]
                    row["n_full_evals_mean"] = float(np.mean(valid))
                if n_epochs_eq and any(n is not None for n in n_epochs_eq):
                    valid = [n for n in n_epochs_eq if n is not None]
                    row["n_epochs_eq_mean"] = float(np.mean(valid))
                rows.append(row)

    return pd.DataFrame(rows), per_method_history, per_method_trajectory


def print_table(df: pd.DataFrame):
    if df.empty:
        print("Нет агрегированных данных.")
        return
    methods = df["method"].unique().tolist()
    datasets = df["dataset"].unique().tolist()

    print("\n=== Test Loss (mean ± std по seeds) ===")
    table = []
    for ds in datasets:
        row = [ds.upper()]
        for m in methods:
            sub = df[(df["dataset"] == ds) & (df["method"] == m)]
            if len(sub):
                r = sub.iloc[0]
                row.append(f"{r['loss_mean']:.4f} ± {r['loss_std']:.4f}  (n={r['n_seeds']})")
            else:
                row.append("N/A")
        table.append(row)
    print(tabulate(table, headers=["Dataset"] + methods, tablefmt="fancy_grid"))

    print("\n=== Test Accuracy / R² (mean ± std по seeds) ===")
    table = []
    for ds in datasets:
        row = [ds.upper()]
        for m in methods:
            sub = df[(df["dataset"] == ds) & (df["method"] == m)]
            if len(sub) and sub.iloc[0]["sec_mean"] is not None:
                r = sub.iloc[0]
                row.append(f"{r['sec_mean']:.4f} ± {r['sec_std']:.4f}")
            else:
                row.append("N/A")
        table.append(row)
    print(tabulate(table, headers=["Dataset"] + methods, tablefmt="fancy_grid"))

    if "time_mean" in df.columns:
        print("\n=== Wall-clock seconds (mean ± std) ===")
        table = []
        for ds in datasets:
            row = [ds.upper()]
            for m in methods:
                sub = df[(df["dataset"] == ds) & (df["method"] == m)]
                if len(sub) and pd.notna(sub.iloc[0].get("time_mean")):
                    r = sub.iloc[0]
                    row.append(f"{r['time_mean']:.1f} ± {r.get('time_std', 0):.1f}")
                else:
                    row.append("N/A")
            table.append(row)
        print(tabulate(table, headers=["Dataset"] + methods, tablefmt="fancy_grid"))
