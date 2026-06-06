"""Статистические тесты для сравнения методов.

Согласно замечанию (Demšar 2006): unit of analysis = датасет, не seed.
Для каждого датасета берём mean-результат метода (поверх seeds), получаем
матрицу [n_datasets x n_methods] и применяем Friedman test + post-hoc Nemenyi.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def build_method_dataset_matrix(df, metric="loss_mean", lower_is_better=True):
    """Превращает DataFrame из aggregate_results в матрицу [datasets × methods].

    Возвращает (matrix DataFrame, datasets list, methods list).
    """
    pivot = df.pivot(index="dataset", columns="method", values=metric)
    pivot = pivot.dropna(axis=1, how='any')  # исключаем методы, не прогнанные везде
    return pivot, pivot.index.tolist(), pivot.columns.tolist()


def friedman_test(pivot, lower_is_better=True):
    """Friedman test на матрице [datasets × methods].

    Возвращает (statistic, p_value, ranks_df), где ranks_df — ранги методов
    по каждому датасету (1 = лучший).
    """
    arr = pivot.values
    if not lower_is_better:
        arr = -arr
    # Ранжируем по строкам (по каждому датасету)
    ranks = np.array([stats.rankdata(row) for row in arr])
    ranks_df = pd.DataFrame(ranks, index=pivot.index, columns=pivot.columns)

    if len(pivot) < 3:
        return None, None, ranks_df  # Friedman требует ≥3 датасета

    stat, p = stats.friedmanchisquare(*[arr[:, j] for j in range(arr.shape[1])])
    return stat, p, ranks_df


def nemenyi_critical_difference(n_methods, n_datasets, alpha=0.05):
    """Critical Difference для Nemenyi post-hoc test (Demšar 2006).

    CD = q_alpha * sqrt(k(k+1) / (6N))
    где k — число методов, N — число датасетов, q_alpha — критическое значение
    студентизированного диапазона.
    """
    # Таблица q-значений для α=0.05, q∞ (Demšar 2006, Table 5)
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
    }
    q_alpha_10 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
        7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }
    table = q_alpha_05 if alpha == 0.05 else q_alpha_10
    if n_methods not in table:
        # Используем ближайшее значение
        n_methods = min(table.keys(), key=lambda k: abs(k - n_methods))
    q = table[n_methods]
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))
    return cd


def plot_critical_difference(ranks_df, alpha=0.05, title=None, save_path=None):
    """Critical Difference diagram (Demšar 2006).

    На оси X — средний ранг метода. Методы, чьи ранги отличаются меньше CD,
    соединены горизонтальной чертой (статистически неотличимы).
    """
    avg_ranks = ranks_df.mean(axis=0).sort_values()
    methods = avg_ranks.index.tolist()
    ranks_vals = avg_ranks.values
    n_methods = len(methods)
    n_datasets = len(ranks_df)
    cd = nemenyi_critical_difference(n_methods, n_datasets, alpha=alpha)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.3 * n_methods))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.axis('off')

    # Ось рангов
    y_axis = 0.85
    ax.plot([1, n_methods], [y_axis, y_axis], 'k-', linewidth=1.5)
    for i in range(1, n_methods + 1):
        ax.plot([i, i], [y_axis, y_axis - 0.02], 'k-', linewidth=1)
        ax.text(i, y_axis + 0.04, str(i), ha='center', fontsize=10)

    # Линии от каждого метода
    half = n_methods / 2
    for i, (m, r) in enumerate(zip(methods, ranks_vals)):
        if i < half:
            x_label = 0.5
            x_text = -0.05
            ha = 'right'
        else:
            x_label = n_methods + 0.5
            x_text = n_methods + 1.05
            ha = 'left'
        y_label = 0.55 - 0.08 * (i if i < half else (n_methods - 1 - i))
        ax.plot([r, r, x_label], [y_axis, y_label, y_label], 'k-', linewidth=0.8)
        ax.text(x_text, y_label, f"{m} ({r:.2f})", ha=ha, va='center', fontsize=9)

    # CD-bar
    ax.plot([1, 1 + cd], [0.95, 0.95], 'k-', linewidth=2)
    ax.plot([1, 1], [0.94, 0.96], 'k-', linewidth=2)
    ax.plot([1 + cd, 1 + cd], [0.94, 0.96], 'k-', linewidth=2)
    ax.text(1 + cd / 2, 0.97, f"CD = {cd:.2f}", ha='center', fontsize=10)

    # Группы статистически неразличимых методов
    sorted_idx = np.argsort(ranks_vals)
    sorted_ranks = ranks_vals[sorted_idx]
    groups = []
    i = 0
    while i < n_methods:
        j = i
        while j + 1 < n_methods and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            groups.append((sorted_ranks[i], sorted_ranks[j]))
        i = j + 1 if j > i else i + 1

    for k, (r_lo, r_hi) in enumerate(groups):
        y_g = 0.78 - 0.025 * k
        ax.plot([r_lo - 0.1, r_hi + 0.1], [y_g, y_g], 'k-', linewidth=2.5)

    if title:
        ax.set_title(title, fontsize=11, pad=20)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def run_full_statistical_analysis(df, metric="loss_mean", lower_is_better=True,
                                   alpha=0.05, save_dir=None):
    """Полный статистический анализ: Friedman + Nemenyi + CD diagram.

    Возвращает dict с ключами: pivot, ranks, friedman_stat, friedman_p, cd.
    """
    pivot, datasets, methods = build_method_dataset_matrix(df, metric=metric)
    if len(datasets) < 3 or len(methods) < 2:
        print(f"Недостаточно данных для Friedman: {len(datasets)} датасетов, "
              f"{len(methods)} методов. Нужно ≥3 датасета и ≥2 метода.")
        return {"pivot": pivot, "ranks": None, "friedman_stat": None,
                "friedman_p": None, "cd": None}

    stat, p, ranks_df = friedman_test(pivot, lower_is_better)
    cd = nemenyi_critical_difference(len(methods), len(datasets), alpha=alpha)

    print(f"=== Friedman test на матрице {len(datasets)} датасетов × {len(methods)} методов ===")
    print(f"χ² = {stat:.4f}, p-value = {p:.4f}")
    if p < alpha:
        print(f"H0 отвергается на уровне α={alpha}: между методами есть значимые различия")
    else:
        print(f"H0 не отвергается на уровне α={alpha}: значимых различий нет")
    print(f"\nСредние ранги (1 = лучший):")
    print(ranks_df.mean(axis=0).sort_values().to_string())
    print(f"\nCritical Difference (Nemenyi, α={alpha}): {cd:.4f}")

    save_path = None
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"cd_diagram_{metric}.png")

    plot_critical_difference(
        ranks_df, alpha=alpha,
        title=f"Critical Difference Diagram ({metric}, N={len(datasets)} datasets)",
        save_path=save_path,
    )

    return {
        "pivot": pivot, "ranks": ranks_df,
        "friedman_stat": stat, "friedman_p": p, "cd": cd,
    }
