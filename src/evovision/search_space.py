"""The macro search space: a small family of convolutional networks.

Each candidate architecture is a continuous genome ``x`` in ``[0, bounds]`` that
rounds to a discrete configuration: three stages, each with a width, kernel
size, and depth. FLOPs and parameter counts are computed analytically so the
efficiency objective is free (no forward pass needed).
"""

from __future__ import annotations

import numpy as np

WIDTHS = [8, 16, 24, 32]
KERNELS = [3, 5]
DEPTHS = [1, 2]

N_STAGES = 3
DIM = 3 * N_STAGES  # (width, kernel, depth) per stage

_INPUT_CHANNELS = 3
_INPUT_SIZE = 32
_N_CLASSES = 10


def n_architectures() -> int:
    """Total number of distinct architectures in the search space."""
    return (len(WIDTHS) * len(KERNELS) * len(DEPTHS)) ** N_STAGES


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
        lo += [0, 0, 0]
        hi += [len(WIDTHS), len(KERNELS), len(DEPTHS)]
    return np.array(list(zip(lo, hi)), dtype=float)


def _decode(value: float, choices: list[int]) -> int:
    """Decode one gene to an index into ``choices`` using equal-width bins."""
    return int(np.clip(np.floor(value), 0, len(choices) - 1))


def to_config(x: np.ndarray) -> dict:
    """Decode a continuous genome into a discrete architecture configuration."""
    x = np.asarray(x, dtype=float).ravel()
    widths, kernels, depths = [], [], []
    for s in range(N_STAGES):
        widths.append(WIDTHS[_decode(x[3 * s], WIDTHS)])
        kernels.append(KERNELS[_decode(x[3 * s + 1], KERNELS)])
        depths.append(DEPTHS[_decode(x[3 * s + 2], DEPTHS)])
    return {"widths": widths, "kernels": kernels, "depths": depths}


def config_key(x: np.ndarray) -> tuple:
    """A hashable identity for the architecture a genome decodes to.

    Many genomes decode to the same network, so this is the right key both for
    caching expensive evaluations and for measuring how much of the space a
    search has actually covered.
    """
    cfg = to_config(x)
    return (tuple(cfg["widths"]), tuple(cfg["kernels"]), tuple(cfg["depths"]))


def enumerate_genomes() -> np.ndarray:
    """Every architecture in the space, as an ``(n_architectures, DIM)`` array.

    The space is small enough to enumerate exhaustively, which makes the exact
    Pareto front computable and gives any search a ground truth to be measured
    against.
    """
    import itertools

    per_stage = list(
        itertools.product(range(len(WIDTHS)), range(len(KERNELS)), range(len(DEPTHS)))
    )
    return np.array(
        [[g for stage in combo for g in stage] for combo in itertools.product(per_stage, repeat=N_STAGES)],
        dtype=float,
    )


def flops(x: np.ndarray, input_channels: int = _INPUT_CHANNELS, input_size: int = _INPUT_SIZE) -> float:
    """Total multiply-accumulate operations (MACs) for a candidate.

    Counts conv and linear MACs only, matching the convention used to report
    MobileNet/EfficientNet costs -- and matching what :func:`models.build_model`
    actually constructs. The convolutions carry no bias term (``bias=False``),
    so none is counted; batch-norm contributes no MACs at inference, where it
    folds into the preceding convolution.
    """
    cfg = to_config(x)
    total = 0.0
    c_in, h, w = input_channels, input_size, input_size
    for s in range(N_STAGES):
        c_out = cfg["widths"][s]
        k = cfg["kernels"][s]
        for _ in range(cfg["depths"][s]):
            total += c_out * h * w * k * k * c_in
            c_in = c_out
        h //= 2
        w //= 2
    total += _N_CLASSES * c_in
    return float(total)


def params(x: np.ndarray, input_channels: int = _INPUT_CHANNELS) -> int:
    """Number of trainable parameters for a candidate.

    Matches :func:`models.build_model` exactly: bias-free convolutions, two
    learnable parameters per batch-norm channel, and a biased classifier.
    """
    cfg = to_config(x)
    total = 0
    c_in = input_channels
    for s in range(N_STAGES):
        c_out = cfg["widths"][s]
        k = cfg["kernels"][s]
        for _ in range(cfg["depths"][s]):
            total += k * k * c_in * c_out  # conv weight, bias=False
            total += 2 * c_out  # batch-norm weight and bias
            c_in = c_out
    total += _N_CLASSES * c_in + _N_CLASSES  # classifier weight and bias
    return total


# Hand-designed baseline architectures, as continuous genomes.
BASELINES = {
    "tiny": [0, 0, 0] * N_STAGES,    # width 8,  kernel 3, depth 1
    "small": [1, 0, 0] * N_STAGES,   # width 16, kernel 3, depth 1
    "medium": [2, 0, 1] * N_STAGES,  # width 24, kernel 3, depth 2
    "wide": [3, 1, 1] * N_STAGES,    # width 32, kernel 5, depth 2
}


def baseline_genomes() -> dict[str, np.ndarray]:
    """Return the named hand-designed baseline architectures."""
    return {name: np.array(g, dtype=float) for name, g in BASELINES.items()}
