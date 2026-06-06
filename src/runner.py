"""Запуск экспериментов: цикл (датасет × seed × метод), сохранение результатов.

Поддерживает tabular и image датасеты. Tabular обрабатывается через
prepare_splits, image — через prepare_image_splits.
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import torch

from .config import CONFIG, SEEDS, RESULTS_DIR
from .utils import make_loader
from .data import prepare_splits, prepare_image_splits, DATASETS
from .evaluation import eval_model
from .methods import METHODS


def _trajectory_to_history(trajectory):
    return np.array([pt["best_val_so_far"] for pt in trajectory], dtype=np.float64)


def _is_run_complete(run_dir):
    """Run считается завершённым, если все 4 файла на месте и meta.json парсится.

    Это защита от прерванных запусков: если процесс был убит между save_run шагами,
    папка может содержать только часть файлов — такие папки трактуем как незавершённые
    и пере-запускаем заново.
    """
    if not os.path.isdir(run_dir):
        return False
    required = ["checkpoint.pt", "trajectory.json", "history.npy", "meta.json"]
    for f in required:
        if not os.path.isfile(os.path.join(run_dir, f)):
            return False
    # Проверяем, что meta.json валидный
    try:
        with open(os.path.join(run_dir, "meta.json")) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return True


def save_run(ds_name, method, seed, model, trajectory, method_cfg, test_metrics):
    """Атомарное сохранение: пишем во временную папку, потом переименовываем.

    Это гарантирует, что после save_run папка либо полная, либо её нет —
    промежуточных состояний быть не может, даже если процесс убьют посередине.
    """
    final_path = os.path.join(RESULTS_DIR, ds_name, method, f"seed_{seed}")
    tmp_path = final_path + ".tmp"

    # Очищаем tmp на случай, если остался от предыдущего прерывания
    if os.path.exists(tmp_path):
        import shutil
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(tmp_path, "checkpoint.pt"))
    with open(os.path.join(tmp_path, "trajectory.json"), "w") as f:
        json.dump(trajectory, f)
    np.save(os.path.join(tmp_path, "history.npy"), _trajectory_to_history(trajectory))
    summary = {}
    if trajectory:
        last = trajectory[-1]
        summary = {
            "total_time": last["time"],
            "n_evals_full": last["n_evals_full"],
            "n_evals_partial": last["n_evals_partial"],
            "n_epochs_eq": last["n_epochs_eq"],
            "best_val": last["best_val_so_far"],
        }
    with open(os.path.join(tmp_path, "meta.json"), "w") as f:
        json.dump({
            "method_config": method_cfg,
            "test_metrics": test_metrics,
            "trajectory_summary": summary,
            "date": datetime.now().isoformat(),
        }, f, indent=2)

    # Атомарное переименование (на одной FS это атомарная операция в POSIX и Windows)
    if os.path.exists(final_path):
        import shutil
        shutil.rmtree(final_path)
    os.rename(tmp_path, final_path)


def _prepare_dataset(name, info, seed):
    """Возвращает (ds_tr, ds_val, ds_test, meta) для любого датасета."""
    if info["kind"] == "tabular":
        data_raw = info["loader"]()
        return prepare_splits(data_raw, info["task"], seed)
    elif info["kind"] == "image":
        return prepare_image_splits(info["loader"], seed)
    else:
        raise ValueError(f"Unknown kind: {info['kind']}")


def run_all(seeds=None, methods=None, datasets=None, skip_existing=False):
    seeds = seeds or SEEDS
    methods = methods or list(METHODS.keys())
    if datasets is None:
        datasets = list(DATASETS.keys())
    elif isinstance(datasets, dict):
        # backward-compat: dict {name: (task, loader_fn)} тоже принимаем
        datasets = list(datasets.keys())

    for ds_name in datasets:
        info = DATASETS[ds_name]
        task = info["task"]
        print(f"\n{'=' * 60}\nDATASET: {ds_name}  (kind={info['kind']}, task={task})\n{'=' * 60}")

        for seed in seeds:
            print(f"\n--- seed {seed} ---")
            ds_tr, ds_val, ds_test, meta = _prepare_dataset(ds_name, info, seed)
            loaders = {
                "train": make_loader(ds_tr, CONFIG["batch_size"], shuffle=True, seed=seed),
                "val":   make_loader(ds_val, CONFIG["batch_size"], shuffle=False),
                "test":  make_loader(ds_test, CONFIG["batch_size"], shuffle=False),
            }
            shape_str = (f"in={meta.get('in_dim', meta.get('image_size'))} "
                         f"out={meta.get('out_dim', meta.get('num_classes'))}")
            print(f"Shape: {shape_str} n_tr={meta['n_train']} n_val={meta['n_val']} n_test={meta['n_test']}")

            for m_name in methods:
                run_dir = os.path.join(RESULTS_DIR, ds_name, m_name, f"seed_{seed}")
                if skip_existing and _is_run_complete(run_dir):
                    print(f"  [{m_name}] skip (already exists)")
                    continue
                # Папка существует, но run не завершился — чистим, чтобы не мешать
                if os.path.exists(run_dir):
                    print(f"  [{m_name}] cleaning incomplete run from previous session")
                    import shutil
                    shutil.rmtree(run_dir)

                # DE-методы пропускаем на image: они не работают для CNN
                if info["kind"] == "image" and m_name in ("Baseline_DE", "Optuna_DE", "DE_DE_Hybrid"):
                    print(f"  [{m_name}] skip (DE not applicable to image data)")
                    continue

                print(f"  [{m_name}] ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    model, trajectory, method_cfg = METHODS[m_name](loaders, meta, seed)
                except Exception as e:
                    print(f"FAIL: {type(e).__name__}: {e}")
                    continue

                if not (trajectory and isinstance(trajectory[0], dict)):
                    raise RuntimeError(f"{m_name} returned legacy history; update method")

                test_metrics = eval_model(model, loaders["test"], task,
                                          meta.get("y_mean"), meta.get("y_std"))
                dt = time.time() - t0
                save_run(ds_name, m_name, seed, model, trajectory, method_cfg, test_metrics)
                tag = "acc" if task == 'classification' else "r2"
                sec = test_metrics.get("accuracy" if task == 'classification' else "r2")
                print(f"loss={test_metrics['loss']:.4f} {tag}={sec:.4f} [{dt:.1f}s]")
