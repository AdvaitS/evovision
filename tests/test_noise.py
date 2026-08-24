"""Signal-to-noise measurement for the accuracy proxy."""

import numpy as np

from evovision import search_space
from evovision.noise import measure_noise_floor


def test_noise_floor_separates_signal_from_seed_noise():
    """A proxy with a big architecture effect and small seed jitter reads as usable."""

    def accuracy_fn(genome, seed):
        rng = np.random.default_rng(seed)
        capacity = sum(search_space.to_config(genome)["widths"])
        return 0.9 - 0.005 * capacity + rng.normal(0, 0.001)

    floor = measure_noise_floor(accuracy_fn, repeats=5)
    assert floor.signal > floor.noise
    assert floor.snr > 3
    assert floor.repeats_for_snr(3.0) == 1
    assert "usable" in floor.summary()


def test_noise_floor_flags_a_proxy_that_is_all_noise():
    """A proxy that ignores the architecture must be reported as dominated by noise."""

    def accuracy_fn(genome, seed):
        return float(np.random.default_rng(seed).normal(0.8, 0.05))

    floor = measure_noise_floor(accuracy_fn, repeats=6)
    assert floor.snr < 1.0
    assert floor.repeats_for_snr(3.0) > 1
    assert "dominated by noise" in floor.summary()


def test_repeats_needed_scales_as_inverse_sqrt():
    """Averaging n runs cuts noise by sqrt(n), so the requirement is quadratic."""

    def accuracy_fn(genome, seed):
        rng = np.random.default_rng(seed)
        capacity = sum(search_space.to_config(genome)["widths"])
        return 0.9 - 0.0002 * capacity + rng.normal(0, 0.02)

    floor = measure_noise_floor(accuracy_fn, repeats=6)
    assert floor.repeats_for_snr(3.0) >= int((3.0 / floor.snr) ** 2)


def test_train_and_eval_repeats_averages_over_seeds():
    from evovision import train_and_eval
    from evovision.data import synthetic

    train_loader, val_loader = synthetic(n_train=120, n_val=80, seed=0)
    genome = search_space.baseline_genomes()["tiny"]
    singles = [train_and_eval(genome, train_loader, val_loader, epochs=1, seed=s) for s in range(3)]
    averaged = train_and_eval(genome, train_loader, val_loader, epochs=1, seed=0, repeats=3)
    assert averaged == float(np.mean(singles))


def test_synthetic_task_is_learnable():
    """The smoke-test dataset must carry signal, or the search optimizes nothing.

    The previous synthetic task labelled Gaussian noise with a pixel-space
    projection, which these globally-pooled convnets cannot represent: every
    architecture scored at chance (0.90 error for ten classes).
    """
    from evovision import train_and_eval
    from evovision.data import synthetic

    train_loader, val_loader = synthetic(n_train=400, n_val=300, seed=0)
    genome = search_space.baseline_genomes()["medium"]
    error = train_and_eval(genome, train_loader, val_loader, epochs=3, seed=0)
    assert error < 0.6, f"synthetic task is not learnable (chance is 0.90, got {error:.3f})"
