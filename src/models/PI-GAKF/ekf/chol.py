"""Cholesky utilities with jitter escalation for PI-GAKF."""
from __future__ import annotations

from typing import Tuple

import torch


def safe_cholesky(matrix: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5) -> Tuple[torch.Tensor, float]:
    """Compute a stable Cholesky factorisation with automatic jitter.

    Args:
        matrix: Positive (semi)definite matrix `(B, D, D)` or `(D, D)`.
        jitter: Initial jitter added to the diagonal when decomposition fails.
        max_tries: Maximum number of jitter escalations.
    Returns:
        A tuple ``(cholesky_factor, used_jitter)``.
    Raises:
        RuntimeError: If the matrix cannot be factorised after ``max_tries`` attempts.
    """

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    chol = None
    used_jitter = jitter
    identity = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + used_jitter * identity)
            break
        except RuntimeError:
            used_jitter *= 10
    if chol is None:
        raise RuntimeError("Cholesky decomposition failed even after jitter escalation")
    return chol, used_jitter


def chol_solve(chol: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``A x = rhs`` where ``chol`` is the lower Cholesky factor of ``A``."""

    return torch.cholesky_solve(rhs, chol)


def chol_logdet(chol: torch.Tensor) -> torch.Tensor:
    """Compute the log-determinant from a Cholesky factor."""

    diag = torch.diagonal(chol, dim1=-2, dim2=-1)
    return 2.0 * torch.log(diag).sum(dim=-1)


__all__ = ["safe_cholesky", "chol_solve", "chol_logdet"]
