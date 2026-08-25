"""The macro search space: a family of convolutional networks.

Each candidate is a continuous genome decoded to three stages, each with a
width, kernel size, depth, **block type** and **expansion ratio**. FLOPs and
parameter counts are computed analytically so the efficiency objective is free
(no forward pass needed) and exact (it matches what ``models.build_model``
constructs, checked in the tests).

Why the block type matters
--------------------------
The original space was three independent, monotone axes -- wider, bigger
kernel, deeper, each strictly more accurate and more expensive. A space like
that has nothing for a search to exploit that uniform sampling does not also
find, and the benchmark showed exactly that: evolution reached 96.0% of the
exact Pareto front and random search 95.4%, indistinguishable at p=1.000.

Block type creates *interaction*. An inverted residual's cost is dominated by
its expansion ratio rather than its width, so the cheapest way to buy accuracy
depends on which block a stage uses -- and the best width for one block type is
not the best width for another. That is the structure a search can exploit and
sampling cannot.
"""

from __future__ import annotations

import numpy as np

WIDTHS = [8, 16, 24, 32, 48, 64]
KERNELS = [3, 5, 7]
DEPTHS = [1, 2, 3]
#: ``conv``: conv-bn-relu. ``residual``: the same plus a skip (with a 1x1
#: projection when the channel count changes). ``inverted_residual``: MobileNetV2's
#: MBConv -- 1x1 expand, depthwise k*k, 1x1 project, skip when shapes allow.
BLOCKS = ["conv", "residual", "inverted_residual"]
#: Expansion ratio, used only by ``inverted_residual``.
EXPANSIONS = [1, 3, 6]

N_STAGES = 3
GENES_PER_STAGE = 5  # width, kernel, depth, block, expansion
DIM = GENES_PER_STAGE * N_STAGES

#: Above this many architectures, enumerate_genomes() refuses rather than
#: silently trying to materialise the whole space.
MAX_ENUMERABLE = 2_000_000

_INPUT_CHANNELS = 3
_INPUT_SIZE = 32
_N_CLASSES = 10


def n_architectures() -> int:
    """Total number of distinct architectures in the search space."""
    per_stage = len(WIDTHS) * len(KERNELS) * len(DEPTHS) * len(BLOCKS) * len(EXPANSIONS)
    return per_stage**N_STAGES


def bounds() -> np.ndarray:
    """Per-dimension ``(dim, 2)`` bounds of the continuous genome.

    Each gene spans ``[0, n_choices]`` and decodes with ``floor``, so every
    discrete choice occupies an equal-width interval. Decoding with ``round``
    instead gives the two extreme choices half-width bins, biasing a uniformly
    sampled population against the smallest and largest widths -- precisely the
    ends of the accuracy/compute trade-off the search exists to map.
    """
    lo: list[int] = []
    hi: list[int] = []
    for _ in range(N_STAGES):
        lo += [0, 0, 0, 0, 0]
        hi += [len(WIDTHS), len(KERNELS), len(DEPTHS), len(BLOCKS), len(EXPANSIONS)]
    return np.array(list(zip(lo, hi)), dtype=float)


def _decode(value: float, choices: list[int]) -> int:
    """Decode one gene to an index into ``choices`` using equal-width bins."""
    return int(np.clip(np.floor(value), 0, len(choices) - 1))


def to_config(x: np.ndarray) -> dict:
    """Decode a continuous genome into a discrete architecture configuration."""
    x = np.asarray(x, dtype=float).ravel()
    widths, kernels, depths, blocks, expansions = [], [], [], [], []
    for s in range(N_STAGES):
        base = GENES_PER_STAGE * s
        widths.append(WIDTHS[_decode(x[base], WIDTHS)])
        kernels.append(KERNELS[_decode(x[base + 1], KERNELS)])
        depths.append(DEPTHS[_decode(x[base + 2], DEPTHS)])
        blocks.append(BLOCKS[_decode(x[base + 3], BLOCKS)])
        expansions.append(EXPANSIONS[_decode(x[base + 4], EXPANSIONS)])
    return {
        "widths": widths,
        "kernels": kernels,
        "depths": depths,
        "blocks": blocks,
        "expansions": expansions,
    }


def config_key(x: np.ndarray) -> tuple:
    """A hashable identity for the architecture a genome decodes to.

    Many genomes decode to the same network, so this is the right key both for
    caching expensive evaluations and for measuring how much of the space a
    search has actually covered.
    """
    cfg = to_config(x)
    return (
        tuple(cfg["widths"]),
        tuple(cfg["kernels"]),
        tuple(cfg["depths"]),
        tuple(cfg["blocks"]),
        # the expansion ratio only changes the network for inverted residuals,
        # so two genomes differing only there are the *same* architecture
        tuple(e if b == "inverted_residual" else 0 for e, b in zip(cfg["expansions"], cfg["blocks"])),
    )


def enumerate_genomes() -> np.ndarray:
    """Every architecture in the space, as an ``(n_architectures, DIM)`` array.

    Raises when the space is too large to materialise. The exact Pareto front is
    a wonderful thing to have and it is worth keeping this path working, but a
    space you can enumerate is a space a search cannot distinguish itself in --
    so growing past this point is the intended direction, not a regression. Use
    :func:`sample_genomes` for a random subset instead.
    """
    import itertools

    total = n_architectures()
    if total > MAX_ENUMERABLE:
        raise ValueError(
            f"the search space holds {total:,} architectures, over the "
            f"{MAX_ENUMERABLE:,} enumeration limit. Use sample_genomes(n) for a "
            "random subset, or shrink the space with a preset."
        )
    per_stage = list(
        itertools.product(
            range(len(WIDTHS)),
            range(len(KERNELS)),
            range(len(DEPTHS)),
            range(len(BLOCKS)),
            range(len(EXPANSIONS)),
        )
    )
    return np.array(
        [[g for stage in combo for g in stage] for combo in itertools.product(per_stage, repeat=N_STAGES)],
        dtype=float,
    )


def encode(x: np.ndarray) -> np.ndarray:
    """Encode a genome as its *decoded architecture*, for a surrogate to learn on.

    The genome is continuous and highly redundant: a wide band of values for a
    gene decodes to the same choice, so two genomes far apart in genome space
    routinely name the same network. A surrogate fitted on raw genomes therefore
    sees the same target at many different coordinates and has to spend capacity
    learning the decoder before it can learn anything about architectures.

    This returns the discrete choice indices instead, so genomes that decode
    identically map to identical surrogate inputs. Ordinal for width, kernel and
    depth -- they are genuinely ordered, and a surrogate should be able to use
    that -- and one-hot for block type, which is categorical and has no order to
    exploit. The unused expansion gene is zeroed for non-inverted blocks, matching
    config_key.
    """
    x = np.asarray(x, dtype=float).ravel()
    out: list[float] = []
    for s in range(N_STAGES):
        base = GENES_PER_STAGE * s
        width = _decode(x[base], WIDTHS)
        kernel = _decode(x[base + 1], KERNELS)
        depth = _decode(x[base + 2], DEPTHS)
        block = _decode(x[base + 3], BLOCKS)
        expansion = _decode(x[base + 4], EXPANSIONS)
        out += [float(width), float(kernel), float(depth)]
        out += [1.0 if block == i else 0.0 for i in range(1, len(BLOCKS))]
        out.append(float(expansion) if BLOCKS[block] == "inverted_residual" else 0.0)
    return np.array(out, dtype=float)


def encode_many(X: np.ndarray) -> np.ndarray:
    """:func:`encode` over a batch, returning ``(n, ENCODED_DIM)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.vstack([encode(row) for row in X]) if len(X) else np.empty((0, ENCODED_DIM))


#: Width of the :func:`encode` representation.
ENCODED_DIM = N_STAGES * (3 + (len(BLOCKS) - 1) + 1)


def sample_genomes(n: int, seed: int = 0) -> np.ndarray:
    """``n`` architectures drawn uniformly from the space, without duplicates.

    The replacement for :func:`enumerate_genomes` once the space is too large to
    enumerate. Deduplicates on the decoded architecture, so ``n`` distinct
    networks come back rather than ``n`` genomes that may collide.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds()[:, 0], bounds()[:, 1]
    seen: dict[tuple, np.ndarray] = {}
    attempts = 0
    while len(seen) < n and attempts < 50 * n + 1000:
        batch = rng.uniform(lo, hi, size=(max(n, 64), DIM))
        for row in batch:
            key = config_key(row)
            if key not in seen:
                seen[key] = row
                if len(seen) >= n:
                    break
        attempts += len(batch)
    return np.array(list(seen.values()), dtype=float)


def _block_cost(kind, c_in, c_out, k, expansion, h, w):
    """``(macs, params)`` for one block at resolution ``h x w``.

    Mirrors models.build_block exactly. Convolutions are bias-free, batch-norm
    contributes 2 parameters per channel and no inference MACs (it folds into
    the preceding convolution), and only conv/linear MACs are counted -- the
    convention used to report MobileNet/EfficientNet costs.
    """
    if kind == "conv":
        macs = c_out * h * w * k * k * c_in
        params = k * k * c_in * c_out + 2 * c_out
        return macs, params

    if kind == "residual":
        macs = c_out * h * w * k * k * c_in
        params = k * k * c_in * c_out + 2 * c_out
        if c_in != c_out:  # 1x1 projection on the skip path
            macs += c_out * h * w * c_in
            params += c_in * c_out + 2 * c_out
        return macs, params

    if kind == "inverted_residual":
        hidden = max(1, c_in * expansion)
        macs = params = 0
        if expansion != 1:  # 1x1 expand
            macs += hidden * h * w * c_in
            params += c_in * hidden + 2 * hidden
        macs += hidden * h * w * k * k  # depthwise: groups == hidden
        params += k * k * hidden + 2 * hidden
        macs += c_out * h * w * hidden  # 1x1 project
        params += hidden * c_out + 2 * c_out
        return macs, params

    raise ValueError(f"unknown block type {kind!r}")


def _walk(x: np.ndarray, input_channels: int, input_size: int):
    """Yield ``(macs, params)`` per block, tracking channels and resolution."""
    cfg = to_config(x)
    c_in, h, w = input_channels, input_size, input_size
    for s in range(N_STAGES):
        c_out = cfg["widths"][s]
        k = cfg["kernels"][s]
        kind = cfg["blocks"][s]
        expansion = cfg["expansions"][s]
        for _ in range(cfg["depths"][s]):
            yield _block_cost(kind, c_in, c_out, k, expansion, h, w)
            c_in = c_out
        h //= 2
        w //= 2
    yield _N_CLASSES * c_in, _N_CLASSES * c_in + _N_CLASSES  # classifier


def flops(x: np.ndarray, input_channels: int = _INPUT_CHANNELS, input_size: int = _INPUT_SIZE) -> float:
    """Total multiply-accumulate operations (MACs) for a candidate."""
    return float(sum(macs for macs, _ in _walk(x, input_channels, input_size)))


def params(x: np.ndarray, input_channels: int = _INPUT_CHANNELS) -> int:
    """Number of trainable parameters for a candidate."""
    return int(sum(p for _, p in _walk(x, input_channels, _INPUT_SIZE)))


# Hand-designed baseline architectures, as continuous genomes.
BASELINES = {
    # width, kernel, depth, block, expansion  (indices into the lists above)
    "tiny": [0, 0, 0, 0, 0] * N_STAGES,        # 8ch  k3 d1 conv
    "small": [1, 0, 0, 0, 0] * N_STAGES,       # 16ch k3 d1 conv
    "medium": [2, 0, 1, 0, 0] * N_STAGES,      # 24ch k3 d2 conv
    "wide": [3, 1, 1, 0, 0] * N_STAGES,        # 32ch k5 d2 conv
    "resnet_ish": [3, 0, 1, 1, 0] * N_STAGES,  # 32ch k3 d2 residual
    "mobile_ish": [4, 0, 1, 2, 1] * N_STAGES,  # 48ch k3 d2 inverted residual, e=3
}


def baseline_genomes() -> dict[str, np.ndarray]:
    """Return the named hand-designed baseline architectures."""
    return {name: np.array(g, dtype=float) for name, g in BASELINES.items()}
