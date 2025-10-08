"""Consistency metrics for PI-GAKF."""
from __future__ import annotations

from typing import Optional

import torch
from statsmodels.stats.diagnostic import acorr_ljungbox

from ..utils.masking import masked_sum


def nis(delta_y: torch.Tensor, S_inv_delta: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute Normalized Innovation Squared per time-step."""

    values = (delta_y * S_inv_delta).sum(dim=-1)
    if mask is None:
        return values
    values = values.masked_fill(mask, 0.0)
    return values


def nees(error: torch.Tensor, covariance_inv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    values = (error.unsqueeze(-2) @ covariance_inv @ error.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    if mask is not None:
        values = values.masked_fill(mask, 0.0)
    return values


def ljung_box_pvalues(innovations: torch.Tensor, lag: int = 20) -> torch.Tensor:
    """Compute Ljung–Box p-values for each dimension independently."""

    batch, time, dim = innovations.shape
    pvalues = torch.zeros(batch, dim, device=innovations.device)
    for b in range(batch):
        for d in range(dim):
            series = innovations[b, :, d].detach().cpu().numpy()
            valid = ~torch.isnan(innovations[b, :, d]).cpu().numpy()
            series = series[valid]
            if len(series) <= lag + 1:
                pvalues[b, d] = 0.0
                continue
            result = acorr_ljungbox(series, lags=[lag], return_df=True)
            pvalues[b, d] = float(result["lb_pvalue"].iloc[0])
    return pvalues


def aggregate_nis(nis_values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Aggregate NIS over a window."""

    summed = masked_sum(nis_values.unsqueeze(-1), mask).squeeze(-1)
    return summed


__all__ = ["nis", "nees", "ljung_box_pvalues", "aggregate_nis"]
