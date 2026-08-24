"""Memoize architecture evaluation by decoded architecture.

The genome is continuous but the architecture is discrete, so a large region of
genome space decodes to the same network. NSGA-II's SBX crossover and polynomial
mutation move genomes by small continuous steps, which means most offspring are
*genotypically* new and *phenotypically* identical to something already trained.

Without memoization the search spends the overwhelming majority of its budget
retraining networks it has already seen: on the default configuration
(``pop_size=20``, ``generations=15``), 320 training runs cover just 34 distinct
architectures. Caching on the decoded architecture removes that waste entirely
and is strictly lossless -- the same architecture trained with the same seed on
the same data returns the same error.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from evovision.search_space import config_key


class ArchitectureCache:
    """Wraps an expensive ``accuracy_fn`` and evaluates each architecture once.

    Exposes the statistics needed to report a search honestly: ``n_unique`` is
    the number of architectures actually trained (the real cost), while
    ``n_requested`` is how many the optimizer asked for.
    """

    def __init__(self, accuracy_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.accuracy_fn = accuracy_fn
        self._cache: dict[tuple, float] = {}
        self.n_requested = 0
        self.n_evaluated = 0

    @property
    def n_unique(self) -> int:
        """Number of distinct architectures evaluated so far."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Fraction of requests served from the cache."""
        if self.n_requested == 0:
            return 0.0
        return 1.0 - self.n_evaluated / self.n_requested

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.n_requested += len(X)

        keys = [config_key(row) for row in X]
        # Deduplicate within the batch as well as against history: a single
        # generation routinely contains several copies of one architecture.
        todo: dict[tuple, int] = {}
        for i, key in enumerate(keys):
            if key not in self._cache and key not in todo:
                todo[key] = i

        if todo:
            rows = X[list(todo.values())]
            errors = np.asarray(self.accuracy_fn(rows), dtype=float).ravel()
            if len(errors) != len(rows):
                raise ValueError(
                    f"accuracy_fn returned {len(errors)} values for {len(rows)} architectures"
                )
            self.n_evaluated += len(rows)
            for key, error in zip(todo, errors):
                self._cache[key] = float(error)

        return np.array([self._cache[key] for key in keys])


def memoize(accuracy_fn: Callable[[np.ndarray], np.ndarray]) -> ArchitectureCache:
    """Return a cached view of ``accuracy_fn``, keyed on decoded architecture."""
    return ArchitectureCache(accuracy_fn)
