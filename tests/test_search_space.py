"""Search-space correctness."""

import numpy as np

from evovision import DIM, N_STAGES, bounds, flops, params, to_config


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
