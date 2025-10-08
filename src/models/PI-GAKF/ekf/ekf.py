"""Differentiable Extended Kalman Filter with Joseph-form update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .chol import chol_logdet, chol_solve, safe_cholesky
from .jacobians import batch_jacobian
from ..dynamics import ResidualDynamics
from ..measurement import LinearMeasurement, NonlinearMeasurement
from ..utils.psd_param import PSDParameter
from ..utils.masking import apply_mask


@dataclass
class EKFConfig:
    state_dim: int
    obs_dim: int
    dt: float
    jitter: float = 1e-6


class DifferentiableEKF(nn.Module):
    """Differentiable EKF operating on batched sequences."""

    def __init__(
        self,
        config: EKFConfig,
        dynamics: ResidualDynamics,
        measurement: nn.Module,
        q_param: PSDParameter,
        r_param: PSDParameter,
    ) -> None:
        super().__init__()
        self.config = config
        self.dynamics = dynamics
        self.measurement = measurement
        self.q_param = q_param
        self.r_param = r_param
        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim
        self.eye = torch.eye(self.state_dim)

    def forward(
        self,
        observations: torch.Tensor,
        x0: torch.Tensor,
        Sigma0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run EKF over a batch of observation sequences.

        Args:
            observations: `(B, T, n)` observation tensor.
            x0: Initial state `(B, m)` or `(m,)`.
            Sigma0: Initial covariance `(B, m, m)` or `(m, m)`.
            mask: Optional boolean mask `(B, T)` indicating padding.
        Returns:
            Dictionary with filtered states, covariances, innovations, etc.
        """

        B, T, _ = observations.shape
        device = observations.device
        eye = self.eye.to(device)

        def ensure_batch(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
            if t.dim() == len(target_shape) - 1:
                t = t.unsqueeze(0).expand(target_shape)
            return t

        x_filt = ensure_batch(x0, (B, self.state_dim))
        Sigma_filt = ensure_batch(Sigma0, (B, self.state_dim, self.state_dim))

        Q = self.q_param.matrix().to(device).unsqueeze(0).expand(B, -1, -1)
        R = self.r_param.matrix().to(device).unsqueeze(0).expand(B, -1, -1)

        x_filt_history = []
        x_pred_history = []
        Sigma_filt_history = []
        Sigma_pred_history = []
        innovations = []
        S_mats = []
        logdet_S = []
        whitened_innov = []

        for t in range(T):
            # Prediction
            x_pred = self.dynamics(x_filt)
            F = batch_jacobian(lambda inp: self.dynamics(inp), x_filt)
            Sigma_pred = F @ Sigma_filt @ F.transpose(-1, -2) + Q

            # Observation
            y_pred = self.measurement(x_pred)
            H = batch_jacobian(lambda inp: self.measurement(inp), x_pred)
            S = H @ Sigma_pred @ H.transpose(-1, -2) + R
            chol_S, _ = safe_cholesky(S, jitter=self.config.jitter)
            logdet = chol_logdet(chol_S)

            obs_t = observations[:, t]
            delta_y = obs_t - y_pred
            if mask is not None:
                delta_y = delta_y.masked_fill(mask[:, t].unsqueeze(-1), 0.0)

            Sigma_HT = Sigma_pred @ H.transpose(-1, -2)
            K = chol_solve(chol_S, Sigma_HT.transpose(-1, -2)).transpose(-1, -2)
            correction = (K @ delta_y.unsqueeze(-1)).squeeze(-1)
            x_filt = x_pred + correction

            KH = K @ H
            I_minus_KH = eye.unsqueeze(0) - KH
            Sigma_filt = (
                I_minus_KH @ Sigma_pred @ I_minus_KH.transpose(-1, -2)
                + K @ R @ K.transpose(-1, -2)
            )
            Sigma_filt = 0.5 * (Sigma_filt + Sigma_filt.transpose(-1, -2))

            whitened = torch.linalg.solve_triangular(chol_S, delta_y.unsqueeze(-1), upper=False).squeeze(-1)

            # Store
            x_pred_history.append(x_pred)
            x_filt_history.append(x_filt)
            Sigma_pred_history.append(Sigma_pred)
            Sigma_filt_history.append(Sigma_filt)
            innovations.append(delta_y)
            S_mats.append(S)
            logdet_S.append(logdet)
            whitened_innov.append(whitened)

        result = {
            "x_pred": torch.stack(x_pred_history, dim=1),
            "x_filt": torch.stack(x_filt_history, dim=1),
            "Sigma_pred": torch.stack(Sigma_pred_history, dim=1),
            "Sigma_filt": torch.stack(Sigma_filt_history, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "S": torch.stack(S_mats, dim=1),
            "logdet_S": torch.stack(logdet_S, dim=1),
            "whitened": torch.stack(whitened_innov, dim=1),
        }
        if mask is not None:
            result["mask"] = mask
        return result


__all__ = ["DifferentiableEKF", "EKFConfig"]
