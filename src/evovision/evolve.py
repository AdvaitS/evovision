"""Run the evolutionary search and collect the Pareto frontier."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from coevo import NSGA2, TrueMultiObjectiveEvaluator
from coevo.core.mo_result import MultiObjectiveResult

from evovision.cache import ArchitectureCache
from evovision.evaluator import NASSurrogateEvaluator
from evovision.problem import make_problem


def evolve(
    accuracy_fn: Callable[[np.ndarray], np.ndarray],
    pop_size: int = 30,
    generations: int = 30,
    surrogate_factory: Callable[[], Any] | None = None,
    eval_fraction: float = 0.3,
    seed: int = 0,
    cache: bool = True,
) -> MultiObjectiveResult:
    """Evolve efficient vision architectures.

    ``accuracy_fn`` maps genomes ``(n, dim)`` to validation errors ``(n,)``.
    Pass ``surrogate_factory`` to use surrogate-assisted accuracy prediction
    (training only a fraction of candidates); otherwise every candidate is trained.

    ``cache`` memoizes ``accuracy_fn`` on the *decoded architecture*, so a
    network is trained once however many genomes decode to it. This is lossless
    and removes the large majority of a search's cost (see :mod:`evovision.cache`).
    The cache is attached to the result as ``metadata["cache"]``, and the number
    of architectures genuinely trained as ``metadata["architectures_trained"]``,
    so a benchmark can report the real budget rather than the requested one.
    """
    cached = ArchitectureCache(accuracy_fn) if cache else None
    if cached is not None:
        accuracy_fn = cached

    problem = make_problem(accuracy_fn)
    evaluator = (
        NASSurrogateEvaluator(accuracy_fn, surrogate_factory, eval_fraction=eval_fraction)
        if surrogate_factory is not None
        else TrueMultiObjectiveEvaluator(problem)
    )
    result = NSGA2(pop_size=pop_size, generations=generations, seed=seed).optimize(
        problem, evaluator
    )
    if cached is not None:
        result.metadata["cache"] = cached
        result.metadata["architectures_trained"] = cached.n_unique
    return result
