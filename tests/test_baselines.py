"""Search baselines: random search and the exact enumerated front."""

import numpy as np

from evovision import evolve, search_space
from evovision.baselines import (
    evaluated_budget,
    exhaustive_front,
    front_hypervolume,
    random_search,
)
from evovision.cache import ArchitectureCache


def _proxy(X):
    X = np.atleast_2d(X)
    out = []
    for x in X:
        cfg = search_space.to_config(x)
        capacity = sum(w * d for w, d in zip(cfg["widths"], cfg["depths"]))
        out.append(0.85 - 0.30 * (1.0 - np.exp(-capacity / 60.0)))
    return np.array(out)


def test_random_search_samples_distinct_architectures():
    cache = ArchitectureCache(_proxy)
    result = random_search(cache, budget=40, seed=0)
    assert result["architectures_trained"] == 40
    assert cache.n_unique == 40, "budget must be spent on distinct architectures"
    assert len(result["objectives"]) <= 40


def test_random_search_budget_cannot_exceed_the_space():
    result = random_search(ArchitectureCache(_proxy), budget=10_000, seed=0)
    assert result["architectures_trained"] == search_space.n_architectures()


def test_exhaustive_front_bounds_every_search():
    """The enumerated front is ground truth: no search can beat it."""
    exact = exhaustive_front(_proxy)
    assert exact["architectures_trained"] == search_space.n_architectures()

    evolved = evolve(_proxy, pop_size=20, generations=10, seed=0)
    assert front_hypervolume(evolved.objectives) <= exact["hypervolume"] + 1e-6

    sampled = random_search(ArchitectureCache(_proxy), budget=60, seed=0)
    assert sampled["hypervolume"] <= exact["hypervolume"] + 1e-6


def test_exhaustive_front_is_non_dominated():
    exact = exhaustive_front(_proxy)
    objectives = exact["objectives"]
    for i, a in enumerate(objectives):
        for j, b in enumerate(objectives):
            if i == j:
                continue
            assert not (np.all(b <= a) and np.any(b < a)), "front contains a dominated point"


def test_evaluated_budget_reports_trained_not_requested():
    """true_evaluations counts requests; the honest cost is architectures trained."""
    result = evolve(_proxy, pop_size=20, generations=15, seed=0)
    assert evaluated_budget(result) == result.metadata["architectures_trained"]
    assert evaluated_budget(result) < result.true_evaluations
