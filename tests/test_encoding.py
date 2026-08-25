"""Surrogate input encoding, and the evaluator's accounting."""

import numpy as np

from evovision import search_space
from evovision.cache import ArchitectureCache
from evovision.evaluator import NASSurrogateEvaluator


def _proxy(X):
    X = np.atleast_2d(X)
    return np.array([0.9 - 1e-8 * search_space.flops(row) for row in X])


def test_genomes_that_decode_alike_encode_alike():
    """The whole point: a surrogate should see architectures, not genome noise."""
    genome = np.array(search_space.BASELINES["small"], dtype=float)
    twin = genome.copy()
    twin[4] = 2.0  # expansion gene, unused by a conv block
    assert search_space.config_key(genome) == search_space.config_key(twin)
    np.testing.assert_array_equal(search_space.encode(genome), search_space.encode(twin))


def test_distinct_architectures_encode_distinctly():
    genomes = search_space.sample_genomes(300, seed=0)
    encoded = {tuple(search_space.encode(g)) for g in genomes}
    assert len(encoded) == 300


def test_encoding_is_full_rank_with_a_constant_term():
    """One-hot block columns sum to 1 and are collinear with a polynomial
    surrogate's constant term -- scipy's RBF fails outright on that. Reference
    coding is what keeps the design matrix invertible.
    """
    encoded = search_space.encode_many(search_space.sample_genomes(400, seed=1))
    design = np.column_stack([encoded, np.ones(len(encoded))])
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_every_block_type_is_representable():
    seen = {
        tuple(search_space.encode(np.array([3, 0, 1, i, 1] * search_space.N_STAGES, dtype=float)))
        for i in range(len(search_space.BLOCKS))
    }
    assert len(seen) == len(search_space.BLOCKS)


def test_evaluator_charges_only_for_architectures_actually_trained():
    """Counting cache hits as true evaluations overstates the surrogate's cost."""
    cache = ArchitectureCache(_proxy)
    evaluator = NASSurrogateEvaluator(cache, lambda: _Stub(), warmup=1, eval_fraction=1.0)
    genome = search_space.sample_genomes(1, seed=0)
    for _ in range(5):
        evaluator(genome)          # the same architecture, five times
    assert cache.n_unique == 1
    assert evaluator.n_true == 1, f"charged {evaluator.n_true} for one architecture"


def test_evaluator_survives_a_surrogate_that_cannot_fit():
    """A surrogate that raises must not take the search down with it."""

    class Exploding:
        def fit(self, X, y):
            raise np.linalg.LinAlgError("singular matrix")

        def predict(self, X):
            raise RuntimeError("never fitted")

    cache = ArchitectureCache(_proxy)
    evaluator = NASSurrogateEvaluator(cache, Exploding, warmup=1, eval_fraction=0.5)
    genomes = search_space.sample_genomes(12, seed=2)
    for i in range(0, 12, 4):
        out = evaluator(genomes[i : i + 4])
        assert out.shape == (4, 2)
        assert np.all(np.isfinite(out))


def test_archive_deduplicates_before_fitting():
    class Recorder:
        def __init__(self):
            self.rows = None

        def fit(self, X, y):
            self.rows = len(X)

        def predict(self, X):
            return np.zeros(len(X))

    recorder = Recorder()
    cache = ArchitectureCache(_proxy)
    evaluator = NASSurrogateEvaluator(cache, lambda: recorder, warmup=1, eval_fraction=1.0)
    genome = search_space.sample_genomes(1, seed=3)
    for _ in range(6):
        evaluator(np.vstack([genome] * 3))
    assert recorder.rows in (None, 1), f"fitted on {recorder.rows} rows for one architecture"


class _Stub:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(np.atleast_2d(X)))
