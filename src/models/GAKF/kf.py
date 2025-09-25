from __future__ import annotations

import torch
from torch import Tensor


def ensure_posdef(matrix: Tensor, jitter: float = 1e-5) -> Tensor:
    """Add jitter to batched SPD matrices to avoid Cholesky failure."""
    eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    return matrix + eye * jitter


def joseph_update(
    P_prior: Tensor,
    K: Tensor,
    H: Tensor,
    R: Tensor,
    identity: Tensor,
) -> Tensor:
    """
    Joseph stabilized covariance update:
        P^+ = (I - K H) P^- (I - K H)^T + K R K^T
    Shapes:
        P_prior: [B, T, n_x, n_x]
        K: [B, T, n_x, n_y]
        H: [n_y, n_x]
        R: [n_y, n_y]
        identity: [B, T, n_x, n_x]
    """
    I_minus_KH = identity - torch.matmul(K, H.unsqueeze(0).unsqueeze(0))
    left = torch.matmul(I_minus_KH, torch.matmul(P_prior, I_minus_KH.transpose(-1, -2)))
    KR = torch.matmul(K, torch.matmul(R, K.transpose(-1, -2)))
    return left + KR
