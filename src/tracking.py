"""Унифицированное логирование бюджета и траектории сходимости.

Каждый метод во время работы должен:
  1) Создать BudgetTracker в начале.
  2) После каждой осмысленной "точки" (конец эпохи / поколения / trial)
     вызывать tracker.log(...) с текущими значениями метрик.
  3) Вернуть tracker.trajectory как часть результата.

Поля в каждой точке:
  - time:           wall-clock секунды от старта
  - n_evals_full:   число "полных" обучений модели
  - n_evals_partial:число "частичных" обучений (для MFBO/Hyperband)
  - n_epochs_eq:    эпохи-эквиваленты (= sum(epochs_done) по всем оценкам)
  - train_fitness:  train-loss / fitness, видимый методом (тот, который он минимизирует)
  - val_loss:       объективная метрика на val (одинаково считается у всех)
  - best_val_so_far: накопленный минимум val_loss

Это позволяет:
  - Строить графики с осью X в любых единицах.
  - Сравнивать методы по реально сопоставимому бюджету.
  - Видеть в DE-методах разницу между "что метод оптимизирует" и "что объективно происходит на val".
"""

import time


class BudgetTracker:
    def __init__(self):
        self._t0 = time.time()
        self.trajectory = []
        self._n_full = 0
        self._n_partial = 0
        self._n_epochs = 0.0
        self._best_val = float('inf')

    def add_full_eval(self, epochs: float = 1.0):
        """Полная (full-fidelity) оценка: например, обучение модели целиком."""
        self._n_full += 1
        self._n_epochs += epochs

    def add_partial_eval(self, epochs: float):
        """Частичная (low-fidelity) оценка: обучение с урезанным числом эпох."""
        self._n_partial += 1
        self._n_epochs += epochs

    def log(self, train_fitness: float, val_loss: float, note: str = ""):
        """Записать точку в траекторию. Должна вызываться после хотя бы одной add_*."""
        if val_loss < self._best_val:
            self._best_val = val_loss
        self.trajectory.append({
            "time": time.time() - self._t0,
            "n_evals_full": self._n_full,
            "n_evals_partial": self._n_partial,
            "n_epochs_eq": self._n_epochs,
            "train_fitness": float(train_fitness) if train_fitness is not None else None,
            "val_loss": float(val_loss),
            "best_val_so_far": float(self._best_val),
            "note": note,
        })

    def summary(self) -> dict:
        return {
            "total_time": time.time() - self._t0,
            "n_evals_full": self._n_full,
            "n_evals_partial": self._n_partial,
            "n_epochs_eq": self._n_epochs,
            "best_val": self._best_val,
            "n_trajectory_points": len(self.trajectory),
        }
