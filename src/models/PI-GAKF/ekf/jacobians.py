"""Jacobian helpers for differentiable EKF."""
from __future__ import annotations

from typing import Callable

import torch

try:
    from torch.func import jacrev
except ImportError:  # pragma: no cover - fallback for older PyTorch
    jacrev = None  # type: ignore


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def batch_jacobian(fn: TensorFn, x: torch.Tensor) -> torch.Tensor:
    """Compute the Jacobian of ``fn`` for each item in the batch.

    Args:
        fn: Function mapping `(B, D)` to `(B, M)`.
        x: Input tensor `(B, D)` requiring gradients.
    Returns:
        Jacobian tensor `(B, M, D)`.
    """

    if jacrev is not None:
        def single_fn(single_x: torch.Tensor) -> torch.Tensor:
            return fn(single_x.unsqueeze(0)).squeeze(0)

        jac = jacrev(single_fn)
        return torch.stack([jac(x_i) for x_i in x], dim=0)

    # Fallback: loop with autograd.functional.jacobian
    jac_list = []
    for x_i in x:
        x_i = x_i.detach().requires_grad_(True)
        y_i = fn(x_i.unsqueeze(0)).squeeze(0)
        jac_i = torch.autograd.functional.jacobian(lambda inp: fn(inp.unsqueeze(0)).squeeze(0), x_i,
                                                   create_graph=True, vectorize=True)
        jac_list.append(jac_i)
    return torch.stack(jac_list, dim=0)


__all__ = ["batch_jacobian"]
