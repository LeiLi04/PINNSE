"""Residual-physics dynamics models for PI-GAKF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from .nn_blocks import ResidualMLP

StateFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DynamicsConfig:
    """Configuration for the residual dynamics model.

    Attributes:
        state_dim: Dimension of the latent state.
        hidden_dim: Width of the residual network.
        depth: Number of residual blocks.
        dt: Discrete time-step used for physics residuals.
        gated: Whether to use gated residual blocks.
    """

    state_dim: int
    hidden_dim: int = 64
    depth: int = 2
    dt: float = 1.0
    gated: bool = False


class ResidualDynamics(nn.Module):
    """Residual dynamics ``f_mix = f_known + g_θ`` used by PI-GAKF.

    Parameters
    ----------
    config:
        Dynamics configuration.
    f_known:
        Known dynamics contribution. If ``None`` it defaults to zero.
    phys_derivative:
        Function returning the continuous-time derivative ``ẋ_phys`` used by the
        physics residual. When ``None`` it is inferred from ``f_known`` via
        ``(f_known(x) - x)/dt``.
    """

    def __init__(
        self,
        config: DynamicsConfig,
        f_known: Optional[StateFn] = None,
        phys_derivative: Optional[StateFn] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.dt = config.dt
        self.f_known = f_known or (lambda x: x)
        self.phys_derivative_fn = phys_derivative
        self.residual_net = ResidualMLP(
            in_features=self.state_dim,
            hidden_features=config.hidden_dim,
            out_features=self.state_dim,
            depth=config.depth,
            gated=config.gated,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``f_mix(x)``.

        Args:
            x: Current state `(B, D)`.
        Returns:
            Next-step deterministic state `(B, D)` before noise addition.
        """

        known = self.f_known(x)
        residual = self.residual_net(x)
        return known + residual

    def phys_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous-time derivative used in the physics residual."""

        if self.phys_derivative_fn is not None:
            return self.phys_derivative_fn(x)
        # Fallback: finite difference of known dynamics
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            known_next = self.f_known(x)
        return (known_next - x) / self.dt


def build_residual_dynamics(config: DynamicsConfig, f_known: Optional[StateFn] = None,
                             phys_derivative: Optional[StateFn] = None) -> ResidualDynamics:
    """Factory helper to instantiate :class:`ResidualDynamics`."""

    return ResidualDynamics(config=config, f_known=f_known, phys_derivative=phys_derivative)


__all__ = ["ResidualDynamics", "DynamicsConfig", "build_residual_dynamics"]
