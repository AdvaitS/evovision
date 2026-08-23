"""Architecture-level memoization of the expensive accuracy function."""

import numpy as np
import pytest

from evovision import evolve, search_space as ss
from evovision.cache import ArchitectureCache


def _counting_accuracy_fn(counter):
    def accuracy_fn(X):
        X = np.atleast_2d(X)
        counter.append(len(X))
        return np.array([0.9 - 0.004 * sum(ss.to_config(x)["widths"]) for x in X])

    return accuracy_fn


def test_cache_evaluates_each_architecture_once():
    calls = []
    cache = ArchitectureCache(_counting_accuracy_fn(calls))
    genome = np.array(ss.BASELINES["small"], dtype=float)
    # three genomes that all decode to the same architecture
    batch = np.vstack([genome, genome + 0.1, genome + 0.4])
    first = cache(batch)
    second = cache(batch)
    assert sum(calls) == 1, "one architecture should mean one evaluation"
    assert cache.n_unique == 1
    assert cache.n_requested == 6
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, first[0])


def test_cache_is_lossless():
    """Cached results must equal what the uncached function would return."""
    rng = np.random.default_rng(0)
    X = rng.uniform(ss.bounds()[:, 0], ss.bounds()[:, 1], size=(50, ss.DIM))
    raw = _counting_accuracy_fn([])
    cache = ArchitectureCache(_counting_accuracy_fn([]))
    np.testing.assert_allclose(cache(X), raw(X))


def test_cache_rejects_wrong_length_result():
    cache = ArchitectureCache(lambda X: np.zeros(len(X) + 1))
    with pytest.raises(ValueError, match="returned"):
        cache(np.zeros((2, ss.DIM)))


def test_evolve_caches_by_default_and_reports_real_budget():
    calls = []
    result = evolve(_counting_accuracy_fn(calls), pop_size=20, generations=15, seed=0)
    trained = result.metadata["architectures_trained"]
    assert trained == sum(calls)
    # the optimizer asks for pop_size * (generations + 1) evaluations; caching
    # must collapse that to a small number of distinct architectures
    assert result.metadata["cache"].n_requested == 20 * 16
    assert trained < 0.25 * result.metadata["cache"].n_requested


def test_evolve_cache_can_be_disabled():
    calls = []
    result = evolve(_counting_accuracy_fn(calls), pop_size=10, generations=3, seed=0, cache=False)
    assert "cache" not in result.metadata
    assert sum(calls) == 10 * 4
