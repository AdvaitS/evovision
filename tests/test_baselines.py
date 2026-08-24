"""Search baselines: random search and the exact enumerated front."""

import numpy as np

from evovision import evolve, search_space
from evovision.baselines import (
    evaluated_budget,
    front_hypervolume,
    random_search,
    reference_front,
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
    result = random_search(ArchitectureCache(_proxy), budget=80, seed=0)
    assert result["architectures_trained"] == min(80, search_space.n_architectures())


def test_reference_front_falls_back_to_sampling_when_too_large():
    """The space is no longer enumerable; the reference must say so, not pretend."""
    ref = reference_front(_proxy, n_samples=1500, seed=0)
    assert ref["exact"] is False
    assert ref["architectures_trained"] == 1500
    assert ref["hypervolume"] > 0


def test_reference_front_dominates_a_small_search():
    """A 30,000-architecture reference should still beat a 60-evaluation search."""
    ref = reference_front(_proxy, n_samples=3000, seed=0)
    sampled = random_search(ArchitectureCache(_proxy), budget=60, seed=0)
    assert sampled["hypervolume"] <= ref["hypervolume"] + 1e-6


def test_reference_front_is_non_dominated():
    ref = reference_front(_proxy, n_samples=1200, seed=0)
    objectives = ref["objectives"]
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


def test_evolution_beats_random_search_given_enough_budget():
    """The claim the repo exists to make, at a budget where it can be made.

    At the old default (~58 architectures trained) evolution and random search
    are indistinguishable; from ~156 the difference is significant. This asserts
    the direction at a budget in between, kept small so the test stays fast.
    """
    evo, rand = [], []
    for seed in range(7):
        result = evolve(_proxy, pop_size=30, generations=30, seed=seed)
        budget = evaluated_budget(result)
        evo.append(front_hypervolume(result.objectives))
        rand.append(random_search(ArchitectureCache(_proxy), budget, seed=seed)["hypervolume"])
    assert np.median(evo) > np.median(rand)
    assert sum(e > r for e, r in zip(evo, rand)) >= 4, f"evolution won {sum(e > r for e, r in zip(evo, rand))}/7"
