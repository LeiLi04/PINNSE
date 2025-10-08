"""Positive semi-definite parameterisations for covariance matrices."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class PSDConfig:
    dim: int
    init_scale: float = 1.0
    min_diag: float = 1e-6


class PSDParameter(nn.Module):
    """Lower-triangular factor with positive diagonal using softplus."""

    def __init__(self, config: PSDConfig) -> None:
        super().__init__()
        self.config = config
        tril_indices = torch.tril_indices(config.dim, config.dim)
        self.register_buffer("tril_rows", tril_indices[0], persistent=False)
        self.register_buffer("tril_cols", tril_indices[1], persistent=False)
        num_params = config.dim * (config.dim + 1) // 2
        init = torch.zeros(num_params)
        with torch.no_grad():
            init[self.tril_rows == self.tril_cols] = torch.log(torch.exp(torch.tensor(config.init_scale)) - 1)
        self.raw = nn.Parameter(init)
        self.softplus = nn.Softplus()

    def lower_triangle(self) -> torch.Tensor:
        dim = self.config.dim
        L = torch.zeros(dim, dim, device=self.raw.device, dtype=self.raw.dtype)
        L[self.tril_rows, self.tril_cols] = self.raw
        diag = self.softplus(torch.diagonal(L)) + self.config.min_diag
        L[self.tril_rows[self.tril_rows == self.tril_cols], self.tril_cols[self.tril_rows == self.tril_cols]] = diag
        return L

    def matrix(self) -> torch.Tensor:
        L = self.lower_triangle()
        return L @ L.transpose(-1, -2)


__all__ = ["PSDParameter", "PSDConfig"]
