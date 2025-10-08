"""Physics residual loss for PI-GAKF."""
from __future__ import annotations

from typing import Optional

import torch

from ..utils.masking import masked_mean


def physics_residual_loss(residual: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average squared norm of physics residuals.

    Args:
        residual: Tensor `(B, T, D)`.
        mask: Optional boolean mask `(B, T)`.
    Returns:
        Scalar loss.
    """

    squared = (residual ** 2).sum(dim=-1)
    loss = masked_mean(squared.unsqueeze(-1), mask).mean()
    return loss


__all__ = ["physics_residual_loss"]
