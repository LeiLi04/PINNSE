"""Innovation negative log-likelihood for PI-GAKF."""
from __future__ import annotations

from typing import Optional

import torch

from ..utils.masking import masked_mean


def innovation_nll(delta_y: torch.Tensor, S: torch.Tensor, logdet_S: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the innovation NLL averaged over valid time steps.

    Args:
        delta_y: Innovation `(B, T, n)`.
        S: Innovation covariance `(B, T, n, n)`.
        logdet_S: Pre-computed log-determinants `(B, T)`.
        mask: Optional boolean mask `(B, T)`.
    Returns:
        Scalar loss tensor.
    """

    chol_solve = torch.linalg.solve
    S_inv_delta = torch.linalg.solve(S, delta_y.unsqueeze(-1)).squeeze(-1)
    maha = (delta_y * S_inv_delta).sum(dim=-1)
    terms = maha + logdet_S
    loss = masked_mean(terms.unsqueeze(-1), mask).mean()
    return loss


__all__ = ["innovation_nll"]
