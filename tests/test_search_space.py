"""Search-space correctness."""

import numpy as np

from evovision import DIM, N_STAGES, bounds, flops, params, search_space, to_config
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


# --- objective fidelity and decoding regressions ----------------------------


def test_analytic_objectives_match_the_built_model():
    """flops() and params() must describe the network build_model() constructs.

    The efficiency objective is the one the search optimizes directly, so an
    analytic shortcut that disagrees with the real module is optimizing the
    wrong thing. Checked over the whole (small) search space.
    """
    import torch

    from evovision.models import build_model

    def true_params(genome):
        return sum(p.numel() for p in build_model(genome).parameters())

    def true_macs(genome):
        model = build_model(genome).eval()
        total = [0]

        def hook(module, inputs, output):
            if isinstance(module, torch.nn.Conv2d):
                total[0] += (
                    output.numel() * module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                )
            elif isinstance(module, torch.nn.Linear):
                total[0] += module.in_features * module.out_features

        handles = [m.register_forward_hook(hook) for m in model.modules()]
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
        for h in handles:
            h.remove()
        return total[0]

    for genome in search_space.enumerate_genomes():
        assert search_space.params(genome) == true_params(genome)
        assert search_space.flops(genome) == true_macs(genome)


def test_decoding_gives_every_choice_an_equal_share():
    """Uniform genome sampling must reach every width equally often.

    Decoding with round() gives the extreme choices half-width bins, biasing the
    population away from the smallest and largest networks -- the two ends of
    the trade-off the search is meant to map.
    """
    rng = np.random.default_rng(0)
    lo, hi = search_space.bounds()[:, 0], search_space.bounds()[:, 1]
    X = rng.uniform(lo, hi, size=(20000, search_space.DIM))
    counts = {w: 0 for w in search_space.WIDTHS}
    for x in X:
        counts[search_space.to_config(x)["widths"][0]] += 1
    for width, count in counts.items():
        assert abs(count / len(X) - 0.25) < 0.02, f"width {width} sampled at {count / len(X):.1%}"


def test_enumerate_genomes_covers_the_space_exactly():
    genomes = search_space.enumerate_genomes()
    assert len(genomes) == search_space.n_architectures()
    assert len({search_space.config_key(g) for g in genomes}) == search_space.n_architectures()


def test_config_key_identifies_the_architecture_not_the_genome():
    genome = np.array(search_space.BASELINES["medium"], dtype=float)
    assert search_space.config_key(genome) == search_space.config_key(genome + 0.3)


def test_baseline_genomes_still_decode_to_their_named_configs():
    decoded = {n: search_space.to_config(g) for n, g in search_space.baseline_genomes().items()}
    assert decoded["tiny"]["widths"] == [8, 8, 8]
    assert decoded["wide"]["widths"] == [32, 32, 32]
    assert decoded["wide"]["kernels"] == [5, 5, 5]
    assert decoded["wide"]["depths"] == [2, 2, 2]
