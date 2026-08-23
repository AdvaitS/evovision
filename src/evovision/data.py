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
):
    """Return (train_loader, val_loader) over a tiny deterministic toy dataset.

    Labels are ``argmax(X @ W^T)`` for a fixed projection ``W``, so the task is
    learnable but fast — useful for smoke-testing a search without downloading
    CIFAR-10.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(seed)
    W = torch.tensor(rng.normal(size=(n_classes, 3 * 32 * 32)), dtype=torch.float32)

    def make(n: int) -> TensorDataset:
        X = torch.tensor(rng.normal(size=(n, 3, 32, 32)), dtype=torch.float32)
        y = (X.view(n, -1) @ W.t()).argmax(1)
        return TensorDataset(X, y)

    train = DataLoader(make(n_train), batch_size=batch_size, shuffle=True)
    val = DataLoader(make(n_val), batch_size=batch_size, shuffle=False)
    return train, val
