"""PI-GAKF training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn, optim

from ..dynamics import ResidualDynamics
from ..ekf.ekf import DifferentiableEKF
from ..losses.innovation_nll import innovation_nll
from ..losses.physics_residual import physics_residual_loss
from ..losses.wgan_gp import critic_loss, generator_loss, gradient_penalty
from ..transformer_critic import TransformerCritic
from ..utils.logging import console, log_scalars


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


class PIGAKFTrainer:
    """High-level trainer orchestrating critic/generator updates."""

    def __init__(
        self,
        ekf: DifferentiableEKF,
        dynamics: ResidualDynamics,
        critic: TransformerCritic,
        generator_opt: optim.Optimizer,
        critic_opt: optim.Optimizer,
        config,
        device: torch.device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.ekf = ekf
        self.dynamics = dynamics
        self.critic = critic
        self.generator_opt = generator_opt
        self.critic_opt = critic_opt
        self.config = config
        self.device = device
        self.scaler = scaler if config.train.amp else None
        self.state = TrainerState()

    def _initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = self.config.model.state_dim
        x0 = torch.zeros(batch_size, state_dim, device=self.device)
        Sigma0 = torch.eye(state_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        return x0, Sigma0

    def _unwrap_batch(self, batch) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            obs, mask = batch
        else:
            obs, mask = batch, None
        return obs.to(self.device), mask.to(self.device) if mask is not None else None

    def train_epoch(self, dataloader: Iterable, writer=None) -> Dict[str, float]:
        self.ekf.train()
        self.dynamics.train()
        self.critic.train()
        metrics = {"critic_loss": 0.0, "generator_loss": 0.0}
        batches = 0

        iterator = iter(dataloader)
        for batch in iterator:
            obs, mask = self._unwrap_batch(batch)
            batch_size, T, _ = obs.shape

            # Ensure enough batches for critic steps
            critic_loss_acc = 0.0
            R_chol = torch.linalg.cholesky(self.ekf.r_param.matrix().to(self.device))
            for _ in range(self.config.train.n_critic):
                x0, Sigma0 = self._initial_state(batch_size)
                outputs = self.ekf(obs, x0, Sigma0, mask=mask)
                y_pred = obs - outputs["innovations"]
                noise = torch.randn_like(y_pred @ R_chol.T) @ R_chol.T
                y_fake = y_pred + noise
                real_scores = self.critic(obs, mask=mask)
                fake_scores = self.critic(y_fake.detach(), mask=mask)
                gp = gradient_penalty(self.critic, obs, y_fake.detach(), mask=mask)
                loss_D = critic_loss(real_scores, fake_scores, gp)

                self.critic_opt.zero_grad()
                loss_D.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.train.clip_norm)
                self.critic_opt.step()
                critic_loss_acc += loss_D.item()

            # Generator step
            x0, Sigma0 = self._initial_state(batch_size)
            outputs = self.ekf(obs, x0, Sigma0, mask=mask)
            delta_y = outputs["innovations"]
            S = outputs["S"]
            logdet_S = outputs["logdet_S"]
            whitened = outputs["whitened"]

            S_inv_delta = torch.linalg.solve(S, delta_y.unsqueeze(-1)).squeeze(-1)
            nis_values = (delta_y * S_inv_delta).sum(dim=-1)

            y_pred = obs - delta_y
            R_chol = torch.linalg.cholesky(self.ekf.r_param.matrix().to(self.device))
            noise = torch.randn_like(y_pred @ R_chol.T) @ R_chol.T
            y_fake = y_pred + noise
            fake_scores = self.critic(y_fake, mask=mask)

            L_innov = innovation_nll(delta_y, S, logdet_S, mask)
            x_pred = outputs["x_pred"]
            x_filt = outputs["x_filt"]
            if x_pred.size(1) > 1:
                prev = x_filt[:, :-1]
                pred = x_pred[:, 1:]
                deriv = self.dynamics.phys_derivative(prev.reshape(-1, prev.size(-1))).reshape(prev.shape)
                physics_res = (pred - prev) / self.config.model.dt - deriv
                physics_mask = mask[:, 1:] if mask is not None else None
                L_phys = physics_residual_loss(physics_res, physics_mask)
            else:
                L_phys = torch.tensor(0.0, device=self.device)
            L_adv = generator_loss(fake_scores)
            L_noise = nis_values.mean() * 0.0  # placeholder term

            loss_G = (
                self.config.train.lambda_innov * L_innov
                + self.config.train.lambda_phys * L_phys
                + self.config.train.lambda_adv * L_adv
                + self.config.train.lambda_noise * L_noise
            )

            self.generator_opt.zero_grad()
            loss_G.backward()
            nn.utils.clip_grad_norm_(
                list(self.dynamics.parameters()) + list(self.ekf.q_param.parameters()) + list(self.ekf.r_param.parameters()),
                self.config.train.clip_norm,
            )
            self.generator_opt.step()

            batches += 1
            metrics["critic_loss"] += critic_loss_acc / self.config.train.n_critic
            metrics["generator_loss"] += loss_G.item()
            self.state.global_step += 1

        metrics = {k: v / max(batches, 1) for k, v in metrics.items()}
        log_scalars(writer, metrics, self.state.epoch)
        self.state.epoch += 1
        console.log(f"Epoch {self.state.epoch}: {metrics}")
        return metrics


__all__ = ["PIGAKFTrainer"]
