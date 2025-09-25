from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from . import kf


@dataclass
class GeneratorOutput:
    """Container for generator forward pass results."""

    x_prior: Tensor  # [B, T, n_states]
    P_prior: Tensor  # [B, T, n_states]
    x_post: Tensor  # [B, T, n_states]
    P_post: Tensor  # [B, T, n_states]
    y_hat: Tensor  # [B, T, n_obs]
    aux: Dict[str, Tensor]
    hidden: Tensor  # [num_layers * num_directions, B, hidden_dim]


class GAKFGenerator(nn.Module):
    """RNN-based prior + differentiable Kalman filter posterior."""

    def __init__(
        self,
        n_obs: int,
        n_states: int,
        H: Tensor,
        R: Tensor,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "GRU",
        diag_cov: bool = True,
        lowrank: bool = False,
        cov_eps: float = 1e-5,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if lowrank:
            raise NotImplementedError("Low-rank covariance not yet supported.")
        self.n_obs = n_obs
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.diag_cov = diag_cov
        self.cov_eps = cov_eps
        self.device = device or torch.device("cpu")

        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}.get(rnn_type.upper())
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")

        # RNN processes shifted observations (teacher forcing)
        self.rnn = rnn_cls(
            input_size=n_obs,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Heads: prior mean and log-variance
        self.head_mean = nn.Linear(hidden_dim, n_states)
        self.head_logvar = nn.Linear(hidden_dim, n_states)
        # Posterior residual corrections
        self.residual_mean = nn.Linear(hidden_dim, n_states)

        # Observation projection y_hat
        self.obs_head = nn.Linear(n_states, n_obs)

        # Kalman matrices (registered buffers to follow device moves)
        self.register_buffer("H", H.clone().detach(), persistent=False)
        self.register_buffer("R", R.clone().detach(), persistent=False)
        self.register_buffer(
            "R_chol", torch.linalg.cholesky(R.clone().detach()), persistent=False
        )

        self.reset_parameters()

        # snapshot for APBM loss
        self.register_buffer("theta_bar", self._get_parameter_vector().detach().clone())

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_dim)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)

    def _get_parameter_vector(self) -> Tensor:
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def update_theta_bar(self, momentum: float = 0.0) -> None:
        """Update snapshot with simple exponential moving average."""
        with torch.no_grad():
            current = self._get_parameter_vector()
            self.theta_bar = momentum * self.theta_bar + (1.0 - momentum) * current

    def forward(
        self,
        y_seq: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> GeneratorOutput:
        """
        Args:
            y_seq: observation sequence [B, T, n_obs]
            hidden: optional initial hidden state

        Returns:
            GeneratorOutput with priors, posteriors, predictions, and auxiliaries.
        """
        B, T, _ = y_seq.shape
        device = y_seq.device

        # Shift observations to provide y_{1:t-1} as input
        zero_first = torch.zeros(B, 1, self.n_obs, device=device, dtype=y_seq.dtype)
        y_input = torch.cat([zero_first, y_seq[:, :-1, :]], dim=1)

        rnn_out, hidden_out = self.rnn(y_input, hidden)

        mu_prior = self.head_mean(rnn_out)  # [B, T, n_states]
        logvar_prior = self.head_logvar(rnn_out)
        P_prior = torch.nn.functional.softplus(logvar_prior) + self.cov_eps

        # Kalman update step
        (
            mu_post,
            P_post_diag,
            y_hat,
            kalman_gain,
            innovation,
            innovation_white,
            S_mats,
            P_post_full,
        ) = self._kalman_filter_update(mu_prior, P_prior, y_seq)

        # Optional residual correction (helps stability)
        mu_post = mu_post + self.residual_mean(rnn_out)

        aux = {
            "K": kalman_gain,
            "innovation": innovation,
            "innovation_white": innovation_white,
            "S": S_mats,
            "P_post_full": P_post_full,
            "hidden_out": hidden_out,
        }

        return GeneratorOutput(
            x_prior=mu_prior,
            P_prior=P_prior,
            x_post=mu_post,
            P_post=P_post_diag,
            y_hat=y_hat,
            aux=aux,
            hidden=hidden_out,
        )

    def _kalman_filter_update(
        self,
        mu_prior: Tensor,
        P_prior_diag: Tensor,
        y_seq: Tensor,
    ):
        """
        Perform differentiable Kalman updates (Joseph stabilized form).
        Shapes:
            mu_prior: [B, T, n_states]
            P_prior_diag: [B, T, n_states] (positive)
            y_seq: [B, T, n_obs]
        Returns:
            mu_post, P_post_diag, y_hat, K, innovation, innovation_white, S, P_post_full
        """
        B, T, _ = mu_prior.shape
        device = mu_prior.device
        dtype = mu_prior.dtype

        H = self.H.to(device=device, dtype=dtype)  # [n_obs, n_states]
        R = self.R.to(device=device, dtype=dtype)
        R_chol = self.R_chol.to(device=device, dtype=dtype)

        P_prior = torch.diag_embed(P_prior_diag)  # [B, T, n_states, n_states]
        mu_prior_vec = mu_prior.unsqueeze(-1)  # [B, T, n_states, 1]
        y_vec = y_seq.unsqueeze(-1)  # [B, T, n_obs, 1]

        # Compute innovation: _t = y_t - H _t^-
        H_mu_prior = torch.matmul(H, mu_prior_vec)  # [B, T, n_obs, 1]
        innovation = (y_vec - H_mu_prior).squeeze(-1)  # [B, T, n_obs]

        # Innovation covariance S_t = H P^- H^T + R
        HP = torch.matmul(H, P_prior)  # [B, T, n_obs, n_states]
        S = torch.matmul(HP, H.transpose(-1, -2)) + R  # broadcasting to [B, T, n_obs, n_obs]
        S = kf.ensure_posdef(S)

        # Cholesky decomposition for stability
        chol_S = torch.linalg.cholesky(S)  # [B, T, n_obs, n_obs]

        # Kalman gain K = P^- H^T S^{-1}
        PHt = torch.matmul(P_prior, H.transpose(-1, -2))  # [B, T, n_states, n_obs]
        # Solve S X = PHt^T => X = S^{-1} PHt^T
        solve = torch.cholesky_solve(
            PHt.transpose(-1, -2), chol_S
        )  # [B, T, n_obs, n_states]
        K = solve.transpose(-1, -2)  # [B, T, n_states, n_obs]

        # Whitened innovation ? = L^{-1} 
        innovation_white = torch.cholesky_solve(
            innovation.unsqueeze(-1), chol_S
        ).squeeze(-1)  # [B, T, n_obs]

        # Posterior mean ^+ = ^- + K 
        mu_post = (mu_prior_vec + torch.matmul(K, innovation.unsqueeze(-1))).squeeze(-1)

        # Posterior covariance via Joseph form: P^+ = (I - K H) P^- (I - K H)^T + K R K^T
        I = torch.eye(self.n_states, device=device, dtype=dtype).expand(B, T, -1, -1)
        P_post_full = kf.joseph_update(P_prior, K, H, R, identity=I)
        P_post_diag = torch.diagonal(P_post_full, dim1=-2, dim2=-1)

        y_hat = torch.matmul(H, mu_post.unsqueeze(-1)).squeeze(-1)

        return (
            mu_post,
            P_post_diag,
            y_hat,
            K,
            innovation,
            innovation_white,
            S,
            P_post_full,
        )

    @torch.no_grad()
    def rollout(
        self,
        y_seq: Tensor,
        horizon: int,
        hidden: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Free-run generator for K steps conditioned on observed prefix.
        Args:
            y_seq: [B, T, n_obs] initial observations.
            horizon: number of steps to roll out.
        Returns:
            y_future: [B, horizon, n_obs] predicted sequence.
        """
        self.eval()
        outs = self.forward(y_seq, hidden=hidden)
        hidden_state = outs.hidden
        last_y = outs.y_hat[:, -1:, :]  # [B, 1, n_obs]
        preds = []

        h = hidden_state
        x_prev = outs.x_post[:, -1, :]
        P_prev = outs.P_post[:, -1, :]

        for _ in range(horizon):
            rnn_input = last_y  # 当前步以上一预测作输入
            rnn_out, h = self.rnn(rnn_input, h)

            mu_prior = self.head_mean(rnn_out)
            logvar_prior = self.head_logvar(rnn_out)
            P_prior = torch.nn.functional.softplus(logvar_prior) + self.cov_eps

            pseudo_obs = last_y  # 直接使用上一时刻预测当作伪观测
            (
                mu_post,
                P_post_diag,
                y_hat,
                _,
                _,
                _,
                _,
                _,
            ) = self._kalman_filter_update(mu_prior, P_prior, pseudo_obs)

            preds.append(y_hat)
            last_y = y_hat  # 下一步继续用最新预测


            x_prev = mu_post.squeeze(1)
            P_prev = P_post_diag.squeeze(1)

        y_future = torch.cat(preds, dim=1)
        return y_future
