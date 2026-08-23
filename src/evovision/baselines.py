"""Search baselines that any NAS result must be measured against.

Random search is the baseline the NAS literature settled on after a run of
published methods turned out not to beat it (Li & Talwalkar, *Random Search and
Reproducibility for NAS*, UAI 2019; Yang, Esperança & Carlucci, *NAS Evaluation
is Frustratingly Hard*, ICLR 2020). A search that cannot beat uniform sampling
at a matched budget has not been shown to search.

This space is small enough to enumerate (4,096 architectures), so two stronger
statements are available than usual:

* random search can sample *distinct* architectures without replacement, which
  is a considerably harder baseline than sampling genomes;
* the exact Pareto front is computable by brute force whenever the accuracy
  function is cheap enough to run on every architecture, giving searches a
  ground truth rather than a relative comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from coevo.core.metrics import hypervolume, nondominated_mask

from evovision import search_space
from evovision.cache import ArchitectureCache
from evovision.problem import REFERENCE_POINT


def _front(genomes: np.ndarray, objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = nondominated_mask(objectives)
    return genomes[mask], objectives[mask]


def random_search(
    accuracy_fn: Callable[[np.ndarray], np.ndarray],
    budget: int,
    seed: int = 0,
) -> dict:
    """Evaluate ``budget`` distinct architectures drawn uniformly at random.

    Sampling without replacement from the enumerated space, so the budget is
    spent entirely on architectures the baseline has not already seen -- the
    strongest form of the baseline, and the fair comparison for a search that
    also caches.
    """
    genomes = search_space.enumerate_genomes()
    rng = np.random.default_rng(seed)
    budget = min(budget, len(genomes))
    chosen = genomes[rng.choice(len(genomes), size=budget, replace=False)]

    errors = np.asarray(accuracy_fn(chosen), dtype=float).ravel()
    macs = np.array([search_space.flops(g) for g in chosen])
    objectives = np.column_stack([errors, macs])

    front_g, front_o = _front(chosen, objectives)
    return {
        "solutions": front_g,
        "objectives": front_o,
        "architectures_trained": budget,
        "hypervolume": hypervolume(front_o, REFERENCE_POINT),
    }


def exhaustive_front(accuracy_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
    """The exact Pareto front, by evaluating every architecture in the space.

    Only tractable when ``accuracy_fn`` is cheap (a proxy, a lookup table, or a
    very short training schedule), but when it is, this is ground truth: it puts
    a hard ceiling on the hypervolume any search can reach, so "how good is this
    search" becomes a percentage rather than a comparison.
    """
    genomes = search_space.enumerate_genomes()
    errors = np.asarray(accuracy_fn(genomes), dtype=float).ravel()
    macs = np.array([search_space.flops(g) for g in genomes])
    objectives = np.column_stack([errors, macs])

    front_g, front_o = _front(genomes, objectives)
    return {
        "solutions": front_g,
        "objectives": front_o,
        "architectures_trained": len(genomes),
        "hypervolume": hypervolume(front_o, REFERENCE_POINT),
    }


def front_hypervolume(objectives: np.ndarray) -> float:
    """Hypervolume of ``objectives`` against the shared reference point."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.size == 0:
        return 0.0
    return hypervolume(objectives[nondominated_mask(objectives)], REFERENCE_POINT)


def evaluated_budget(result) -> int:
    """Architectures genuinely trained by a search, preferring the cache count.

    ``true_evaluations`` counts what the optimizer asked for, which overstates
    the cost by roughly an order of magnitude once duplicate architectures are
    cached. Benchmarks should quote this instead.
    """
    trained = result.metadata.get("architectures_trained")
    if trained is not None:
        return int(trained)
    return int(result.true_evaluations)


def matched_budget_comparison(
    make_accuracy_fn: Callable[[], Callable[[np.ndarray], np.ndarray]],
    evolve_fn: Callable[[Callable], object],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    """Run evolution and random search at the *same* number of trained models.

    The budget is not fixed in advance: it is whatever the evolutionary search
    ends up spending, which random search is then given. That removes the usual
    objection that the baseline was quietly handicapped.
    """
    evo_hv, rand_hv, budgets = [], [], []
    for seed in seeds:
        evo = evolve_fn(make_accuracy_fn())
        budget = evaluated_budget(evo)
        budgets.append(budget)
        evo_hv.append(front_hypervolume(evo.objectives))

        cache = ArchitectureCache(make_accuracy_fn())
        rand_hv.append(random_search(cache, budget, seed=seed)["hypervolume"])

    return {
        "seeds": list(seeds),
        "budgets": budgets,
        "evolution_hv": evo_hv,
        "random_hv": rand_hv,
        "evolution_median": float(np.median(evo_hv)),
        "random_median": float(np.median(rand_hv)),
    }
