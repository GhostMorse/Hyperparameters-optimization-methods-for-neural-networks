# Методы оптимизация гиперпараметров нейронных сетей

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
jupyter lab notebooks/VKR_main.ipynb
```

Все запуски в ноутбуке закомментированы — раскомментируйте нужные.

## Структура

```
vkr/
├── notebooks/
│   ├── VKR_main.ipynb               ← единственный ноутбук
│   └── results_fixed/               ← сохранённые результаты прогонов
├── docs/
│   ├── diploma_text.md              ← черновой текст диплома
│   ├── presentation.pptx            ← черновая презентация (20 слайдов)
│   └── build_pptx.py                ← скрипт сборки презентации
├── src/                             ← вся логика
│   ├── config.py                    — CONFIG, SEEDS, device, RESULTS_DIR
│   ├── tracking.py                  — BudgetTracker (унифицированные траектории)
│   ├── utils.py                     — set_seed, set_weights, make_loader
│   ├── evaluation.py                — eval_model, eval_fitness_on_val
│   ├── runner.py                    — run_all, save_run
│   ├── aggregation.py               — aggregate_results, print_table
│   ├── plots.py                     — plot_convergence_unified (3 оси), bar
│   ├── stats.py                     — Friedman, Nemenyi, CD diagram
│   ├── data/
│   │   ├── preprocessing.py         — TabularDataset, prepare_splits
│   │   ├── image_preprocessing.py   — prepare_image_splits для CIFAR
│   │   └── datasets.py              — DATASETS registry (7 датасетов)
│   ├── models/
│   │   ├── mlp.py                   — MLP для tabular
│   │   ├── cnn.py                   — SimpleCNN для CIFAR-10
│   │   └── model_factory.py         — build_model (выбор по meta)
│   └── methods/
│       ├── base.py                  — METHODS, HPO_METHODS, DE_METHODS registry
│       ├── de.py                    — алгоритм DE с поддержкой tracker
│       │
│       │  Класс A — поиск гиперпараметров (фиксирован Adam):
│       ├── adam_baseline.py         — чистый Adam без подбора
│       ├── adam_random.py           — Random Search
│       ├── adam_optuna.py           — TPE (Optuna)
│       ├── adam_gp_bo.py            — GP-BO (BoTorch + qLogEI)
│       ├── adam_hyperband.py        — Hyperband (RandomSampler + HBPruner)
│       ├── adam_bohb.py             — BOHB (TPE + HBPruner)
│       ├── adam_mfbo.py             — Multi-Fidelity BO (BoTorch + qMFKG)
│       │
│       │  Класс B — black-box (только tabular):
│       ├── baseline_de.py           — DE по весам MLP
│       ├── optuna_de.py             — TPE подбирает h, DE учит веса
│       └── de_de_hybrid.py          — DE+DE (внешний по h,F,Cr; внутренний по весам)
└── requirements.txt
```

## Постановка задачи

Сравниваются два класса методов:

1. **Класс A — поиск гиперпараметров** при фиксированном Adam-обучении весов:
   Adam baseline, Random Search, TPE, GP-BO, Hyperband, BOHB, MFBO.

2. **Класс B — black-box оптимизация** (отдельный подраздел):
   Baseline DE, Optuna+DE, DE+DE Hybrid. В black-box постановке fitness =
   val-loss; val играет роль внешнего ящика (явно проговорено в тексте диплома).
   DE-методы применимы только к MLP на tabular данных.

## Датасеты

| # | Датасет | Тип | N | Признаки |
|---|---|---|---|---|
| 1 | WINE | classif | 178 | 13 |
| 2 | WDBC | classif | 569 | 30 |
| 3 | DIGITS | classif | 1797 | 64 |
| 4 | ABALONE | regr | 4177 | 8 |
| 5 | KIN8NM | regr | 8192 | 8 |
| 6 | Higgs | classif | 1M (sub) | 28 |
| 7 | CIFAR-10 | classif | 50000 | 32×32×3 |

## Формат сохранённых результатов

```
results_fixed/<dataset>/<method>/seed_<i>/
  checkpoint.pt        — веса модели
  trajectory.json      — список точек траектории сходимости
  history.npy          — best-so-far val-loss (back-compat)
  meta.json            — method_config, test_metrics, trajectory_summary
```

Каждая точка траектории — словарь:
- `time` — wall-clock секунды от старта
- `n_evals_full`, `n_evals_partial` — число оценок (full/partial fidelity)
- `n_epochs_eq` — эпохи-эквиваленты
- `train_fitness` — то, что метод сам минимизирует
- `val_loss` — объективная val-метрика
- `best_val_so_far` — накопленный минимум val_loss

## Метрики

- Classification: Cross-Entropy (mean) + Accuracy
- Regression: MSE на нормализованном y + R² на исходном масштабе

mean ± std считается **по seeds** (не по объектам val).

## Статистика

Friedman + Nemenyi с post-hoc тестом, единица анализа — **датасет** (Demšar 2006).
CD diagram строится автоматически.

## Как добавить новый метод

1. Создать `src/methods/<my_method>.py` с функцией:
   ```python
   def train_my_method(loaders, meta, seed):
       tracker = BudgetTracker()
       # ... обучение, периодически tracker.add_*_eval() и tracker.log()
       return model, tracker.trajectory, method_cfg
   ```
2. Добавить в `HPO_METHODS` (или `DE_METHODS`) в `src/methods/base.py`.
3. Если нужны параметры — дописать блок в `CONFIG` в `src/config.py`.

Ноутбук менять не нужно.

## Как добавить датасет

Дописать запись в `DATASETS` в `src/data/datasets.py`. Для image-датасетов
loader должен возвращать `(train_full_ds, test_ds, info_dict)`, для tabular —
объект с `.data` и `.target` (как из sklearn/ucimlrepo/openml).
