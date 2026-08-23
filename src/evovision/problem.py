"""Wrap the NAS search as a coevo multi-objective problem."""

from __future__ import annotations

from typing import Callable

import numpy as np

from coevo import MultiObjectiveProblem

from evovision import search_space

# A reference point worse than any candidate: error 1.0 and 50M MACs.
REFERENCE_POINT = np.array([1.0, 5e7])


def make_problem(accuracy_fn: Callable[[np.ndarray], np.ndarray]) -> MultiObjectiveProblem:
    """Build the two-objective (error, MACs) problem.

    ``accuracy_fn`` maps a batch of genomes ``(n, dim)`` to their validation
    errors ``(n,)``; MACs are computed analytically.
    """

    def objectives(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        errors = np.asarray(accuracy_fn(X), dtype=float).ravel()
        macs = np.array([search_space.flops(x) for x in X])
        return np.column_stack([errors, macs])

    return MultiObjectiveProblem(
        name="evovision-nas",
        func=objectives,
        bounds=search_space.bounds(),
        dim=search_space.DIM,
        n_objectives=2,
        reference_point=REFERENCE_POINT.copy(),
    )
