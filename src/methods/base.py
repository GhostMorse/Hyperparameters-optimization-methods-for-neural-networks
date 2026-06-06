"""Реестр методов. Разделён на два класса:

  HPO_METHODS — оптимизация гиперпараметров при фиксированном Adam-обучении.
  DE_METHODS  — black-box оптимизация (отдельный подраздел в дипломе).

METHODS = HPO_METHODS | DE_METHODS — для совместимости и общего запуска.
"""

from .adam_baseline import train_adam_baseline
from .adam_random import train_adam_random
from .adam_optuna import train_adam_optuna
from .adam_gp_bo import train_adam_gp_bo
from .adam_hyperband import train_adam_hyperband
from .adam_bohb import train_adam_bohb
from .adam_mfbo import train_adam_mfbo

from .baseline_de import train_baseline_de
from .optuna_de import train_optuna_de
from .de_de_hybrid import train_de_de_hybrid


HPO_METHODS = {
    "Adam_baseline":   train_adam_baseline,
    "Adam_RandomSearch": train_adam_random,
    "Adam_TPE":        train_adam_optuna,
    "Adam_GP_BO":      train_adam_gp_bo,
    "Adam_Hyperband":  train_adam_hyperband,
    "Adam_BOHB":       train_adam_bohb,
    "Adam_MFBO":       train_adam_mfbo,
}

DE_METHODS = {
    "Baseline_DE":   train_baseline_de,
    "Optuna_DE":     train_optuna_de,
    "DE_DE_Hybrid":  train_de_de_hybrid,
}

# Объединённый реестр (порядок: сначала HPO, потом DE)
METHODS = {**HPO_METHODS, **DE_METHODS}

# Совместимость со старыми результатами: старое имя "Adam_Optuna" -> "Adam_TPE"
# Если в results_fixed/ остались папки с "Adam_Optuna" — оставляем рабочий доступ.
METHODS_LEGACY_ALIAS = {"Adam_Optuna": "Adam_TPE"}
