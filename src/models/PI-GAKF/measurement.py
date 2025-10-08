"""Measurement models for PI-GAKF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .nn_blocks import MLP, MLPConfig


@dataclass
class LinearMeasurementConfig:
    state_dim: int
    obs_dim: int


class LinearMeasurement(nn.Module):
    """Linear measurement ``y = Hx``."""

    def __init__(self, config: LinearMeasurementConfig, matrix: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if matrix is None:
            matrix = torch.eye(config.obs_dim, config.state_dim)
        self.H = nn.Parameter(matrix, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.H.transpose(-1, -2)


@dataclass
class NonlinearMeasurementConfig:
    state_dim: int
    obs_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)


class NonlinearMeasurement(nn.Module):
    """MLP-based nonlinear measurement model."""

    def __init__(self, config: NonlinearMeasurementConfig) -> None:
        super().__init__()
        mlp_config = MLPConfig(
            in_features=config.state_dim,
            hidden_sizes=config.hidden_sizes,
            out_features=config.obs_dim,
        )
        self.model = MLP(mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = [
    "LinearMeasurement",
    "LinearMeasurementConfig",
    "NonlinearMeasurement",
    "NonlinearMeasurementConfig",
]
