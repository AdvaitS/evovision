"""Surrogate-assisted accuracy evaluation for the NAS search.

Error is expensive (it requires training), while MACs are cheap (analytic). This
evaluator predicts error with a coevo surrogate and computes MACs exactly,
training only the candidates in the earliest *predicted* Pareto fronts.

Two things here are easy to get wrong and both distort the comparison against a
plain search:

* The surrogate learns on the *decoded architecture*
  (:func:`~evovision.search_space.encode`), not the raw genome. Genome space is
  redundant -- a wide band of values decodes to one choice -- so a surrogate
  fitted on genomes sees the same target at many coordinates and spends its
  capacity learning the decoder.
* ``n_true`` counts architectures actually trained, not calls made. When
  ``accuracy_fn`` is an :class:`~evovision.cache.ArchitectureCache`, a repeat
  request costs nothing, and counting it would overstate what the surrogate has
  spent and understate what it saves.
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
        self._fitted = False

    @property
    def n_true(self) -> int:
        return self._n_true

    def _archive(self) -> tuple[np.ndarray, np.ndarray]:
        """Distinct encoded architectures and their errors, most recent kept.

        Deduplication is required, not merely tidy. Encoding on the decoded
        architecture makes repeats *exactly* identical rows, and an interpolating
        surrogate (scipy's RBF) raises LinAlgError on a singular system rather
        than degrading. Raw genomes hid this by always being in general position
        -- which is the same redundancy that made them a poor surrogate input.
        """
        if not self._X:
            return np.empty((0, search_space.ENCODED_DIM)), np.empty((0,))
        X = np.vstack(self._X)
        y = np.concatenate(self._y)
        # Later entries win: a repeat carries the same architecture, and keeping
        # the newest matches the archive's recency trimming below.
        seen: dict[tuple, float] = {}
        for row, value in zip(X, y):
            seen[tuple(row)] = float(value)
        X = np.array(list(seen.keys()), dtype=float)
        y = np.array(list(seen.values()), dtype=float)
        if self.archive_size is not None and len(X) > self.archive_size:
            X, y = X[-self.archive_size :], y[-self.archive_size :]
        return X, y

    def _fit(self) -> bool:
        """Fit the surrogate; report whether it is usable.

        A surrogate that cannot be fitted must not take the search down with it.
        Too few distinct architectures for the model's monomial basis is the
        normal early-search state, not an error.
        """
        X, y = self._archive()
        if len(X) <= search_space.ENCODED_DIM:
            self._fitted = False
            return False
        try:
            self.surrogate.fit(X, y)
            self._fitted = True
        except Exception:
            self._fitted = False
        return self._fitted

    def _train(self, x: np.ndarray) -> np.ndarray:
        before = self._trained_so_far()
        y = np.asarray(self.accuracy_fn(x), dtype=float).ravel()
        after = self._trained_so_far()
        # Charge only for architectures genuinely trained. Without a cache
        # underneath, every request is a miss and this reduces to len(x).
        self._n_true += (after - before) if after is not None else len(x)
        self._X.append(search_space.encode_many(x))
        self._y.append(y)
        return y

    def _trained_so_far(self):
        """The cache's unique-architecture count, or None if there is no cache."""
        return getattr(self.accuracy_fn, "n_unique", None)

    def _macs(self, x: np.ndarray) -> np.ndarray:
        return np.array([search_space.flops(row) for row in x])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._calls += 1
        macs = self._macs(x)

        if self._calls <= self.warmup or not self._X:
            err = self._train(x)
            if self._calls >= self.warmup:
                self._fit()
            return np.column_stack([err, macs])

        if not self._fitted:
            # No usable model yet: evaluate honestly rather than predict from one.
            err = self._train(x)
            self._fit()
            return np.column_stack([err, macs])

        try:
            err = np.asarray(
                self.surrogate.predict(search_space.encode_many(x)), dtype=float
            ).ravel()
        except Exception:
            self._fitted = False
            err = self._train(x)
            return np.column_stack([err, macs])

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
            self._fit()

        return np.column_stack([err, macs])
