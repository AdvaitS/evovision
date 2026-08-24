"""Dataset loading (CIFAR-10) with optional deterministic subsampling."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset


def cifar10(
    root: str = "./data",
    batch_size: int = 64,
    subsample: int | None = None,
    seed: int = 0,
):
    """Return (train_loader, val_loader) for CIFAR-10.

    ``subsample`` deterministically limits each split to that many examples, so
    a search can be run quickly with a small proxy dataset.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    val = datasets.CIFAR10(root, train=False, download=True, transform=transform)

    if subsample is not None:
        g = torch.Generator().manual_seed(seed)
        train = Subset(train, torch.randperm(len(train), generator=g)[:subsample].tolist())
        val = Subset(val, torch.randperm(len(val), generator=g)[:subsample].tolist())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def synthetic(
    n_train: int = 500,
    n_val: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    n_classes: int = 10,
    noise: float = 1.5,
):
    """Return (train_loader, val_loader) over a tiny deterministic toy dataset.

    Each class is a distinct oriented sinusoidal grating buried in noise. The
    task is deliberately *convolutional*: orientation and spatial frequency are
    exactly what small conv filters detect, and channel statistics survive the
    global average pool, so a network in this search space can learn it in a
    couple of epochs.

    The previous version labelled Gaussian noise by ``argmax(X @ W.T)`` for a
    fixed pixel-space projection. That is linearly separable in principle but
    unlearnable by *these* models -- global average pooling discards the
    per-pixel phase the label depends on -- so every architecture scored at
    chance (~0.90 error for ten classes) and any search over it was ranking
    noise. A smoke test should exercise the plumbing without silently
    pretending to optimize.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(seed)
    size = 32
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    # One (frequency, orientation) pair per class, well separated.
    angles = np.linspace(0.0, np.pi, n_classes, endpoint=False)
    freqs = 0.10 + 0.16 * (np.arange(n_classes) % 3)

    def make(n: int) -> TensorDataset:
        labels = rng.integers(0, n_classes, size=n)
        images = np.empty((n, 3, size, size), dtype=np.float32)
        for i, label in enumerate(labels):
            theta, freq = angles[label], freqs[label]
            phase = rng.uniform(0, 2 * np.pi)
            grating = np.sin(
                2 * np.pi * freq * (xx * np.cos(theta) + yy * np.sin(theta)) + phase
            )
            sample = grating[None, :, :] + rng.normal(0.0, noise, size=(3, size, size))
            images[i] = sample
        return TensorDataset(
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    train = DataLoader(make(n_train), batch_size=batch_size, shuffle=True)
    val = DataLoader(make(n_val), batch_size=batch_size, shuffle=False)
    return train, val
