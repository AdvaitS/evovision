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


def bounds() -> np.ndarray:
    """Per-dimension ``(dim, 2)`` bounds of the continuous genome."""
    lo: list[int] = []
    hi: list[int] = []
    for _ in range(N_STAGES):
        lo += [0, 0, 0]
        hi += [3, 1, 1]
    return np.array(list(zip(lo, hi)), dtype=float)


def to_config(x: np.ndarray) -> dict:
    """Round a continuous genome to a discrete architecture configuration."""
    x = np.asarray(x, dtype=float).ravel()
    widths, kernels, depths = [], [], []
    for s in range(N_STAGES):
        widths.append(WIDTHS[int(np.clip(round(x[3 * s]), 0, 3))])
        kernels.append(KERNELS[int(np.clip(round(x[3 * s + 1]), 0, 1))])
        depths.append(DEPTHS[int(np.clip(round(x[3 * s + 2]), 0, 1))])
    return {"widths": widths, "kernels": kernels, "depths": depths}


def flops(x: np.ndarray, input_channels: int = _INPUT_CHANNELS, input_size: int = _INPUT_SIZE) -> float:
    """Total multiply-accumulate operations (MACs) for a candidate."""
    cfg = to_config(x)
    total = 0.0
    c_in, h, w = input_channels, input_size, input_size
    for s in range(N_STAGES):
        c_out = cfg["widths"][s]
        k = cfg["kernels"][s]
        for _ in range(cfg["depths"][s]):
            total += c_out * h * w * (k * k * c_in + 1)
            c_in = c_out
        h //= 2
        w //= 2
    total += _N_CLASSES * c_in
    return float(total)


def params(x: np.ndarray, input_channels: int = _INPUT_CHANNELS) -> int:
    """Number of trainable parameters for a candidate."""
    cfg = to_config(x)
    total = 0
    c_in = input_channels
    for s in range(N_STAGES):
        c_out = cfg["widths"][s]
        k = cfg["kernels"][s]
        for _ in range(cfg["depths"][s]):
            total += k * k * c_in * c_out + c_out
            c_in = c_out
    total += _N_CLASSES * c_in + _N_CLASSES
    return total
