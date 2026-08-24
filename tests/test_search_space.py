"""Search-space correctness."""

import numpy as np

from evovision import DIM, N_STAGES, bounds, flops, params, search_space, to_config
from evovision.search_space import baseline_genomes


def _genome(w=0, k=0, d=0, b=0, e=0):
    """A uniform genome: the same (width, kernel, depth, block, expansion) per stage."""
    return np.array([w, k, d, b, e] * N_STAGES, dtype=float)


def test_bounds_shape():
    assert bounds().shape == (DIM, 2)


def test_to_config_decodes_every_gene():
    cfg = to_config(_genome(w=3, k=1, d=1, b=2, e=1))
    assert cfg["widths"] == [32, 32, 32]
    assert cfg["kernels"] == [5, 5, 5]
    assert cfg["depths"] == [2, 2, 2]
    assert cfg["blocks"] == ["inverted_residual"] * 3
    assert cfg["expansions"] == [3, 3, 3]


def test_flops_grows_with_width():
    assert flops(_genome(w=len(search_space.WIDTHS) - 1)) > flops(_genome(w=0))


def test_params_positive():
    assert params(_genome(w=2)) > 0


def test_flops_deterministic():
    x = _genome(w=1, k=1, d=0)
    assert flops(x) == flops(x)


def test_baseline_genomes_are_valid_and_monotonic():
    genomes = baseline_genomes()
    assert set(genomes) == {"tiny", "small", "medium", "wide", "resnet_ish", "mobile_ish"}
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
    wrong thing. The space is far too large to enumerate now, so this checks a
    random sample spanning all three block types.
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

    def true_macs_grouped(genome):
        model = build_model(genome).eval()
        total = [0]

        def hook(module, inputs, output):
            if isinstance(module, torch.nn.Conv2d):
                total[0] += (
                    output.numel()
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
            elif isinstance(module, torch.nn.Linear):
                total[0] += module.in_features * module.out_features

        handles = [m.register_forward_hook(hook) for m in model.modules()]
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
        for h in handles:
            h.remove()
        return total[0]

    genomes = search_space.sample_genomes(120, seed=0)
    seen_blocks = set()
    for genome in genomes:
        seen_blocks.update(search_space.to_config(genome)["blocks"])
        assert search_space.params(genome) == true_params(genome)
        assert search_space.flops(genome) == true_macs_grouped(genome)
    assert seen_blocks == set(search_space.BLOCKS), "sample must span every block type"


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
    blocks = {b: 0 for b in search_space.BLOCKS}
    for x in X:
        cfg = search_space.to_config(x)
        counts[cfg["widths"][0]] += 1
        blocks[cfg["blocks"][0]] += 1
    expected_w = 1.0 / len(search_space.WIDTHS)
    for width, count in counts.items():
        assert abs(count / len(X) - expected_w) < 0.02, f"width {width} at {count / len(X):.1%}"
    expected_b = 1.0 / len(search_space.BLOCKS)
    for block, count in blocks.items():
        assert abs(count / len(X) - expected_b) < 0.02, f"block {block} at {count / len(X):.1%}"


def test_enumerate_genomes_refuses_an_intractable_space():
    """Better a clear error than an attempt to materialise 115M rows."""
    import pytest

    assert search_space.n_architectures() > search_space.MAX_ENUMERABLE
    with pytest.raises(ValueError, match="enumeration limit"):
        search_space.enumerate_genomes()


def test_sample_genomes_returns_distinct_architectures():
    genomes = search_space.sample_genomes(300, seed=0)
    assert len(genomes) == 300
    keys = {search_space.config_key(g) for g in genomes}
    assert len(keys) == 300, "sampling must deduplicate on the decoded architecture"


def test_expansion_only_identifies_inverted_residuals():
    """Two genomes differing only in an unused expansion are the same network."""
    genome = np.array(search_space.BASELINES["small"], dtype=float)  # conv blocks
    other = genome.copy()
    other[4] = 2.0  # expansion gene of stage 0
    assert search_space.to_config(genome)["blocks"][0] == "conv"
    assert search_space.config_key(genome) == search_space.config_key(other)

    mobile = np.array(search_space.BASELINES["mobile_ish"], dtype=float)
    shifted = mobile.copy()
    shifted[4] = 0.0
    assert search_space.to_config(mobile)["blocks"][0] == "inverted_residual"
    assert search_space.config_key(mobile) != search_space.config_key(shifted)


def test_block_types_have_distinct_cost_profiles():
    """The point of block types is that cost is not a function of width alone."""
    costs = {}
    for i, block in enumerate(search_space.BLOCKS):
        genome = np.array([3, 0, 1, i, 2] * search_space.N_STAGES, dtype=float)
        costs[block] = search_space.flops(genome)
    assert len(set(costs.values())) == len(costs), f"block types cost the same: {costs}"
    assert costs["inverted_residual"] != costs["conv"]


def test_config_key_identifies_the_architecture_not_the_genome():
    genome = np.array(search_space.BASELINES["medium"], dtype=float)
    assert search_space.config_key(genome) == search_space.config_key(genome + 0.3)


def test_baseline_genomes_still_decode_to_their_named_configs():
    decoded = {n: search_space.to_config(g) for n, g in search_space.baseline_genomes().items()}
    assert decoded["tiny"]["widths"] == [8, 8, 8]
    assert decoded["wide"]["widths"] == [32, 32, 32]
    assert decoded["wide"]["kernels"] == [5, 5, 5]
    assert decoded["wide"]["depths"] == [2, 2, 2]
    assert decoded["wide"]["blocks"] == ["conv"] * 3
    assert decoded["resnet_ish"]["blocks"] == ["residual"] * 3
    assert decoded["mobile_ish"]["blocks"] == ["inverted_residual"] * 3
    assert decoded["mobile_ish"]["expansions"] == [3, 3, 3]
