"""Build PyTorch models from search-space configurations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evovision.search_space import N_STAGES, to_config


class ConvNet(nn.Module):
    """A simple conv stack: three stages of conv-bn-relu, then global-pool + FC."""

    def __init__(self, config: dict, n_classes: int = 10) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        c_in = 3
        for s in range(N_STAGES):
            c_out = config["widths"][s]
            k = config["kernels"][s]
            for _ in range(config["depths"][s]):
                stages.append(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False))
                stages.append(nn.BatchNorm2d(c_out))
                stages.append(nn.ReLU(inplace=True))
                c_in = c_out
            stages.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def build_model(x: np.ndarray, n_classes: int = 10) -> nn.Module:
    """Build a :class:`ConvNet` from a continuous genome ``x``."""
    return ConvNet(to_config(x), n_classes)
