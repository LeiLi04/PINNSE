"""WGAN-GP losses for PI-GAKF."""
from __future__ import annotations

from typing import Optional

import torch


def critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor, gradient_penalty: torch.Tensor) -> torch.Tensor:
    return fake_scores.mean() - real_scores.mean() + gradient_penalty


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def gradient_penalty(discriminator: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     mask: Optional[torch.Tensor] = None, lambda_gp: float = 10.0) -> torch.Tensor:
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)
    scores = discriminator(interpolated, mask=mask)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * penalty


__all__ = ["critic_loss", "generator_loss", "gradient_penalty"]
