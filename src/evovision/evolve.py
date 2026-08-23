"""Run the evolutionary search and collect the Pareto frontier."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from coevo import NSGA2, TrueMultiObjectiveEvaluator
from coevo.core.mo_result import MultiObjectiveResult

from evovision.evaluator import NASSurrogateEvaluator
from evovision.problem import make_problem


def evolve(
    accuracy_fn: Callable[[np.ndarray], np.ndarray],
    pop_size: int = 30,
    generations: int = 30,
    surrogate_factory: Callable[[], Any] | None = None,
    eval_fraction: float = 0.3,
    seed: int = 0,
) -> MultiObjectiveResult:
    """Evolve efficient vision architectures.

    ``accuracy_fn`` maps genomes ``(n, dim)`` to validation errors ``(n,)``.
    Pass ``surrogate_factory`` to use surrogate-assisted accuracy prediction
    (training only a fraction of candidates); otherwise every candidate is trained.
    """
    problem = make_problem(accuracy_fn)
    evaluator = (
        NASSurrogateEvaluator(accuracy_fn, surrogate_factory, eval_fraction=eval_fraction)
        if surrogate_factory is not None
        else TrueMultiObjectiveEvaluator(problem)
    )
    return NSGA2(pop_size=pop_size, generations=generations, seed=seed).optimize(
        problem, evaluator
    )
