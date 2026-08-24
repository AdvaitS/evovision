"""How much of an accuracy measurement is signal, and how much is the seed?

A short training run is a noisy estimate of an architecture's quality. When that
noise is comparable to the spread between architectures, a search optimizes the
random seed rather than the design -- the central objection to published NAS
results in Yang, Esperanca & Carlucci, *NAS Evaluation is Frustratingly Hard*
(ICLR 2020).

This module measures the ratio directly, so a benchmark can state whether its
numbers mean anything before drawing conclusions from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evovision import search_space


@dataclass
class NoiseFloor:
    """The signal-to-noise ratio of an accuracy proxy.

    ``signal`` is the spread between architectures (the thing a search is meant
    to find); ``noise`` is the typical spread of one architecture across
    training seeds (the thing it must see past).
    """

    signal: float
    noise: float
    per_architecture_mean: list[float]
    per_architecture_sd: list[float]
    repeats: int

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio. Below ~1 a single-seed search is ranking noise."""
        if self.noise <= 0:
            return float("inf")
        return self.signal / self.noise

    def repeats_for_snr(self, target: float = 3.0) -> int:
        """Training repeats needed to reach ``target`` SNR, since noise falls as 1/sqrt(n)."""
        if self.snr >= target:
            return 1
        return int(np.ceil((target / max(self.snr, 1e-9)) ** 2))

    def summary(self) -> str:
        verdict = (
            "usable" if self.snr >= 3 else "marginal" if self.snr >= 1 else "dominated by noise"
        )
        return (
            f"signal={self.signal:.4f} noise={self.noise:.4f} SNR={self.snr:.2f} ({verdict}); "
            f"~{self.repeats_for_snr()} repeats needed for SNR 3"
        )


def measure_noise_floor(
    accuracy_fn: Callable[[np.ndarray, int], float],
    genomes: np.ndarray | None = None,
    repeats: int = 5,
) -> NoiseFloor:
    """Measure architecture signal against training-seed noise.

    ``accuracy_fn`` takes ``(genome, seed)`` and returns an error. ``genomes``
    defaults to the named hand-designed baselines, which span the search space's
    capacity range.
    """
    if genomes is None:
        genomes = np.array(list(search_space.baseline_genomes().values()), dtype=float)
    genomes = np.atleast_2d(np.asarray(genomes, dtype=float))

    means, sds = [], []
    for genome in genomes:
        errors = [float(accuracy_fn(genome, seed)) for seed in range(repeats)]
        means.append(float(np.mean(errors)))
        sds.append(float(np.std(errors)))

    return NoiseFloor(
        signal=float(np.std(means)),
        noise=float(np.mean(sds)),
        per_architecture_mean=means,
        per_architecture_sd=sds,
        repeats=repeats,
    )
