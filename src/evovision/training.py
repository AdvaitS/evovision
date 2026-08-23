"""Train a candidate network and return its validation error."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evovision.models import build_model


def train_and_eval(
    x: np.ndarray,
    train_loader,
    val_loader,
    epochs: int = 2,
    lr: float = 1e-2,
    device: str | None = None,
    seed: int = 0,
) -> float:
    """Train the architecture ``x`` briefly and return validation error in [0, 1]."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(x).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return 1.0 - correct / max(total, 1)
