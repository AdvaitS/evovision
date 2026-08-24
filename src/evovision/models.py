"""Build PyTorch models from search-space configurations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evovision.search_space import N_STAGES, to_config


class ConvBlock(nn.Module):
    """conv-bn-relu."""

    def __init__(self, c_in: int, c_out: int, k: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResidualBlock(nn.Module):
    """conv-bn plus a skip, with a 1x1 projection when the channel count changes."""

    def __init__(self, c_in: int, c_out: int, k: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.project = (
            None
            if c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.project is None else self.project(x)
        return self.act(self.body(x) + identity)


class InvertedResidualBlock(nn.Module):
    """MobileNetV2's MBConv: 1x1 expand, depthwise k*k, 1x1 project.

    Its cost is driven by the expansion ratio rather than the output width,
    which is what makes width and block type interact instead of being two
    independent knobs.
    """

    def __init__(self, c_in: int, c_out: int, k: int, expansion: int) -> None:
        super().__init__()
        hidden = max(1, c_in * expansion)
        layers: list[nn.Module] = []
        if expansion != 1:
            layers += [
                nn.Conv2d(c_in, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        self.body = nn.Sequential(*layers)
        self.use_skip = c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return out + x if self.use_skip else out


def build_block(kind: str, c_in: int, c_out: int, k: int, expansion: int) -> nn.Module:
    """Construct one block of the named type."""
    if kind == "conv":
        return ConvBlock(c_in, c_out, k)
    if kind == "residual":
        return ResidualBlock(c_in, c_out, k)
    if kind == "inverted_residual":
        return InvertedResidualBlock(c_in, c_out, k, expansion)
    raise ValueError(f"unknown block type {kind!r}")


class ConvNet(nn.Module):
    """Three stages of blocks with max-pooling between them, then global-pool + FC."""

    def __init__(self, config: dict, n_classes: int = 10) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        c_in = 3
        for s in range(N_STAGES):
            c_out = config["widths"][s]
            k = config["kernels"][s]
            kind = config["blocks"][s]
            expansion = config["expansions"][s]
            for _ in range(config["depths"][s]):
                stages.append(build_block(kind, c_in, c_out, k, expansion))
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
