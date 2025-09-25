from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, autograd

from .utils import compute_covariance


def wgan_gp_discriminator_loss(
    D_real: Tensor,
    D_fake: Tensor,
    gp_value: Tensor,
    lambda_gp: float,
) -> Tensor:
    """WGAN-GP discriminator loss."""
    return D_fake.mean() - D_real.mean() + lambda_gp * gp_value


def wgan_gp_generator_loss(D_fake: Tensor) -> Tensor:
    """Generator adversarial loss."""
    return -D_fake.mean()


def gradient_penalty(
    discriminator: Callable[[Tensor], Tensor],
    real: Tensor,
    fake: Tensor,
    device: torch.device,
) -> Tensor:
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def reconstruction_loss(y_true: Tensor, y_hat: Tensor, R_diag: Tensor) -> Tensor:
    """
    L_rec = ||y - y_hat||_{R^{-1}}^2
    Args:
        y_true, y_hat: [B, T, n_obs]
        R_diag: [n_obs] observation variances (diagonal)
    """
    inv_R = 1.0 / R_diag.clamp_min(1e-8)
    error = y_true - y_hat
    weighted = error * inv_R.view(1, 1, -1)
    return (weighted ** 2).mean()


def multi_step_loss(
    generator_rollout_fn: Callable[[Tensor, int], Tensor],
    y_seq: Tensor,
    horizon: int,
    num_starts: int = 4,
) -> Tensor:
    """
    Random multi-step rollout consistency loss:
        L_ms = mean || y_{t:t+K-1} - ŷ_{t:t+K-1} ||_2^2
    """
    B, T, _ = y_seq.shape
    if horizon <= 0:
        return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)

    losses = []
    for _ in range(num_starts):
        start = torch.randint(0, max(T - horizon, 1), (1,), device=y_seq.device).item()
        segment = y_seq[:, start : start + horizon, :]
        preds = generator_rollout_fn(y_seq[:, : start + 1, :], horizon)
        diff = segment - preds
        losses.append((diff ** 2).mean())
    return torch.stack(losses).mean()


def innovation_whitening_loss(innovation_white: Tensor) -> Tensor:
    """
    Encourage whitened innovations ν̃ ~ N(0, I):
        mean term + Frobenius distance between covariance and identity.
    """
    B, T, n_obs = innovation_white.shape
    flatten = innovation_white.reshape(B * T, n_obs)
    mean = flatten.mean(dim=0)
    cov = compute_covariance(flatten)
    identity = torch.eye(n_obs, device=innovation_white.device, dtype=innovation_white.dtype)
    return mean.pow(2).mean() + ((cov - identity) ** 2).mean()


def riccati_consistency_loss(P_post_diag: Tensor, P_post_full: Tensor) -> Tensor:
    """
    Enforce diagonal approximation consistent with Joseph update.
        L_kf = ||diag(P_post) - P_post_full||_F^2
    """
    diag_embed = torch.diag_embed(P_post_diag)
    return ((diag_embed - P_post_full) ** 2).mean()


def apbm_regularizer(theta_vec: Tensor, theta_bar: Tensor, alpha: float = 1e-3) -> Tensor:
    """
    APBM-style regulariser towards warm-start weights:
        L_apbm = α ||θ - θ̄||_2^2
    """
    return alpha * ((theta_vec - theta_bar) ** 2).mean()
