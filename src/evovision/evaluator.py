"""Surrogate-assisted accuracy evaluation for the NAS search.

Error is expensive (it requires training), while MACs are cheap (analytic). This
evaluator predicts error with a coevo surrogate and computes MACs exactly,
training only the candidates in the earliest *predicted* Pareto fronts.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from coevo.core.metrics import fast_non_dominated_sort

from evovision import search_space


class NASSurrogateEvaluator:
    """Predict error with a surrogate, compute MACs exactly, train only the best."""

    def __init__(
        self,
        accuracy_fn: Callable[[np.ndarray], np.ndarray],
        surrogate_factory: Callable[[], Any],
        eval_fraction: float = 0.3,
        warmup: int = 3,
        retrain_every: int = 1,
        archive_size: int | None = 200,
    ) -> None:
        self.accuracy_fn = accuracy_fn
        self.surrogate = surrogate_factory()
        self.eval_fraction = eval_fraction
        self.warmup = warmup
        self.retrain_every = retrain_every
        self.archive_size = archive_size
        self._n_true = 0
        self._calls = 0
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []

    @property
    def n_true(self) -> int:
        return self._n_true

    def _archive(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return np.empty((0, search_space.DIM)), np.empty((0,))
        X = np.vstack(self._X)
        y = np.concatenate(self._y)
        if self.archive_size is not None and len(X) > self.archive_size:
            X, y = X[-self.archive_size :], y[-self.archive_size :]
        return X, y

    def _train(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(self.accuracy_fn(x), dtype=float).ravel()
        self._n_true += len(x)
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(y)
        return y

    def _macs(self, x: np.ndarray) -> np.ndarray:
        return np.array([search_space.flops(row) for row in x])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._calls += 1
        macs = self._macs(x)

        if self._calls <= self.warmup or not self._X:
            err = self._train(x)
            if self._calls >= self.warmup:
                X, y = self._archive()
                self.surrogate.fit(X, y)
            return np.column_stack([err, macs])

        err = np.asarray(self.surrogate.predict(x), dtype=float).ravel()

        n = len(x)
        k = max(1, int(np.ceil(n * self.eval_fraction)))
        selected: list[int] = []
        for front in fast_non_dominated_sort(np.column_stack([err, macs])):
            for i in front:
                if len(selected) >= k:
                    break
                selected.append(i)
            if len(selected) >= k:
                break

        err_true = self._train(x[selected])
        err[selected] = err_true

        if self._calls % self.retrain_every == 0:
            X, y = self._archive()
            self.surrogate.fit(X, y)

        return np.column_stack([err, macs])
