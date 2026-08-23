"""Model building and the end-to-end evolutionary loop (with a cheap proxy)."""

import numpy as np
import pytest

from evovision import evolve, flops, to_config
from evovision.search_space import WIDTHS


def _mock_accuracy(X: np.ndarray) -> np.ndarray:
    """Cheap deterministic proxy: wider networks have lower error."""
    out = []
    for x in X:
        cfg = to_config(x)
        width_frac = np.mean([WIDTHS.index(w) / 3.0 for w in cfg["widths"]])
        out.append(0.45 - 0.30 * width_frac)
    return np.array(out)


def test_build_model_and_forward():
    torch = pytest.importorskip("torch")
    from evovision.models import build_model

    model = build_model(np.array([1.0, 1.0, 1.0] * 3))
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_evolve_finds_pareto_tradeoff():
    result = evolve(_mock_accuracy, pop_size=40, generations=40, seed=0)
    errors, macs = result.objectives[:, 0], result.objectives[:, 1]
    assert len(result.objectives) >= 2
    # a genuine accuracy-vs-cost trade-off: both objectives vary along the front
    assert errors.min() < errors.max()
    assert macs.min() < macs.max()
    # wider (lower error) architectures have higher MACs, in aggregate
    assert np.corrcoef(errors, macs)[0, 1] < 0


def test_evolve_surrogate_trains_fewer():
    from coevo import NearestNeighborSurrogate

    exact = evolve(_mock_accuracy, pop_size=30, generations=30, seed=0)
    sur = evolve(
        _mock_accuracy,
        pop_size=30,
        generations=30,
        surrogate_factory=lambda: NearestNeighborSurrogate(),
        seed=0,
    )
    assert sur.true_evaluations < exact.true_evaluations
    assert len(sur.objectives) >= 2
