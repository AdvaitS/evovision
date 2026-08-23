"""Search-space correctness."""

import numpy as np

from evovision import DIM, N_STAGES, bounds, flops, params, to_config
from evovision.search_space import baseline_genomes


def _genome(w=0, k=0, d=0):
    return np.array([w, k, d] * N_STAGES, dtype=float)


def test_bounds_shape():
    assert bounds().shape == (DIM, 2)


def test_to_config_rounds():
    cfg = to_config(_genome(w=3, k=1, d=1))
    assert cfg["widths"] == [32, 32, 32]
    assert cfg["kernels"] == [5, 5, 5]
    assert cfg["depths"] == [2, 2, 2]


def test_flops_grows_with_width():
    small = flops(_genome(w=0))
    large = flops(_genome(w=3))
    assert large > small


def test_params_positive():
    assert params(_genome(w=2)) > 0


def test_flops_deterministic():
    x = _genome(w=1, k=1, d=0)
    assert flops(x) == flops(x)


def test_baseline_genomes_are_valid_and_monotonic():
    genomes = baseline_genomes()
    assert set(genomes) == {"tiny", "small", "medium", "wide"}
    for genome in genomes.values():
        assert genome.shape == (DIM,)
        assert np.all(genome >= bounds()[:, 0]) and np.all(genome <= bounds()[:, 1])
    # wider baselines are more expensive
    assert flops(genomes["wide"]) > flops(genomes["tiny"])
