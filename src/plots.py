"""Визуализация результатов из реальных сохранённых данных.

Кривые сходимости рисуются по трём осям X:
  - 'time'         — wall-clock секунды;
  - 'n_evals_full' — число full-fidelity оценок;
  - 'n_epochs_eq'  — эпохи-эквиваленты.

По оси Y — best-so-far val-loss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import RESULTS_DIR


X_AXES = ["time", "n_evals_full", "n_epochs_eq"]
X_LABELS = {
    "time": "Wall-clock seconds",
    "n_evals_full": "Number of full-fidelity evaluations",
    "n_epochs_eq": "Epochs-equivalent",
}


def _interpolate_to_grid(trajectories, x_key, n_points=100):
    """Для набора траекторий метода: интерполируем best-so-far на общую сетку.

    Возвращает (x_grid, mean, std) или (None, None, None), если нет данных.
    """
    valid = []
    for traj in trajectories:
        xs = [pt.get(x_key) for pt in traj]
        ys = [pt["best_val_so_far"] for pt in traj]
        if not xs or any(v is None for v in xs) or any(v is None for v in ys):
            continue
        valid.append((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    if not valid:
        return None, None, None

    x_min = max(v[0][0] for v in valid)
    x_max = min(v[0][-1] for v in valid)
    if x_max <= x_min:
        return None, None, None

    x_grid = np.linspace(x_min, x_max, n_points)
    interp = np.array([np.interp(x_grid, xs, ys) for xs, ys in valid])
    return x_grid, interp.mean(axis=0), (interp.std(axis=0, ddof=1) if interp.shape[0] > 1 else np.zeros(n_points))


def plot_convergence_unified(per_method_trajectory, x_key="time", save=True, save_name=None):
    """Кривые сходимости для всех датасетов на одной фигуре, по выбранной оси X."""
    if not per_method_trajectory:
        print("Нет траекторий для графика.")
        return

    datasets = list(per_method_trajectory.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, (method, trajectories) in enumerate(per_method_trajectory[ds].items()):
            if not trajectories:
                continue
            x, mean, std = _interpolate_to_grid(trajectories, x_key)
            if x is None:
                continue
            ax.plot(x, mean, label=method, color=colors[j % 10], linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[j % 10])
        ax.set_title(ds.upper(), fontweight='bold')
        ax.set_xlabel(X_LABELS[x_key])
        ax.set_ylabel("Best-so-far val-loss")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save:
        name = save_name or f"convergence_{x_key}.png"
        plt.savefig(os.path.join(RESULTS_DIR, name), dpi=200)
    plt.show()


def plot_convergence_all_axes(per_method_trajectory, save=True):
    """Все три оси X на отдельных фигурах."""
    for x_key in X_AXES:
        plot_convergence_unified(per_method_trajectory, x_key=x_key, save=save)


def plot_bar_comparison(df, save=True):
    if df.empty:
        print("Нет данных для barplot.")
        return
    plt.figure(figsize=(max(12, 2 * df['dataset'].nunique()), 6))
    ax = sns.barplot(data=df, x="dataset", y="loss_mean", hue="method", palette="viridis")

    datasets = df['dataset'].unique().tolist()
    methods = df['method'].unique().tolist()
    for cont, m in zip(ax.containers, methods):
        # Берём error для тех баров, которые реально есть в этом контейнере
        n_bars = len(cont)
        errs = []
        for p in cont:
            # Найти датасет по позиции бара (простой подход: индекс среди datasets)
            x_center = p.get_x() + p.get_width() / 2
            ds_idx = int(round(x_center))
            if 0 <= ds_idx < len(datasets):
                ds = datasets[ds_idx]
                sub = df[(df["dataset"] == ds) & (df["method"] == m)]
                errs.append(sub.iloc[0]["loss_std"] if len(sub) else 0)
            else:
                errs.append(0)
        if errs:
            ax.errorbar(
                x=[p.get_x() + p.get_width() / 2 for p in cont],
                y=[p.get_height() for p in cont],
                yerr=errs, fmt='none', c='black', capsize=3, linewidth=1,
            )

    plt.yscale('symlog', linthresh=0.01)
    plt.title("Сравнение методов: test-loss (mean ± std по seeds)")
    plt.ylabel("Loss (symlog)")
    plt.xlabel("Dataset")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, "bar_comparison.png"), dpi=200)
    plt.show()


# Backward-compat: старая plot_convergence от history.npy
def plot_convergence(per_method_history, save=True):
    """Старая функция для совместимости — рисует best-so-far по % прогресса."""
    if not per_method_history:
        print("Нет данных для графика.")
        return
    datasets = list(per_method_history.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, (method, histories) in enumerate(per_method_history[ds].items()):
            if not histories:
                continue
            max_len = max(len(h) for h in histories)
            padded = np.array([
                np.interp(np.linspace(0, 1, max_len),
                          np.linspace(0, 1, len(h)), h)
                for h in histories
            ])
            mean = padded.mean(axis=0)
            std = padded.std(axis=0, ddof=1) if padded.shape[0] > 1 else np.zeros_like(mean)
            xs = np.linspace(0, 100, max_len)
            ax.plot(xs, mean, label=method, color=colors[j % 10], linewidth=1.8)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=colors[j % 10])
        ax.set_title(ds.upper(), fontweight='bold')
        ax.set_xlabel("Relative progress (%)")
        ax.set_ylabel("Val loss")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, "convergence_legacy.png"), dpi=200)
    plt.show()
