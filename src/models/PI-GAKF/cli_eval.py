from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .cli_train import MeasurementOnlyDataset
from .dynamics import DynamicsConfig, build_residual_dynamics
from .ekf.ekf import DifferentiableEKF, EKFConfig
from .losses.innovation_nll import innovation_nll
from .measurement import LinearMeasurement, LinearMeasurementConfig
from .training.metrics import ljung_box_pvalues, nis
from .transformer_critic import TransformerCritic, TransformerCriticConfig
from .utils.config import ExperimentConfig
from .utils.logging import console
from .utils.psd_param import PSDConfig, PSDParameter


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    HydraConfig.get().job.name = "pi_gakf_eval"
    exp_cfg = ExperimentConfig.from_omegaconf(cfg["experiment"])
    device = torch.device(exp_cfg.device if torch.cuda.is_available() else "cpu")

    dataset = MeasurementOnlyDataset(to_absolute_path(exp_cfg.data.path))

    def collate(batch):
        lengths = [item.size(0) for item in batch]
        max_len = max(lengths)
        obs_batch = torch.zeros(len(batch), max_len, exp_cfg.model.obs_dim)
        mask_batch = torch.ones(len(batch), max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            obs_batch[i, : item.size(0)] = item
            mask_batch[i, : item.size(0)] = False
        return obs_batch, mask_batch

    dataloader = DataLoader(dataset, batch_size=exp_cfg.data.batch_size, collate_fn=collate)

    dynamics_cfg = DynamicsConfig(
        state_dim=exp_cfg.model.state_dim,
        hidden_dim=exp_cfg.model.dynamics_hidden,
        depth=exp_cfg.model.dynamics_depth,
        dt=exp_cfg.model.dt,
    )
    dynamics = build_residual_dynamics(dynamics_cfg).to(device)

    measurement = LinearMeasurement(
        LinearMeasurementConfig(exp_cfg.model.state_dim, exp_cfg.model.obs_dim)
    ).to(device)

    ekf_cfg = EKFConfig(state_dim=exp_cfg.model.state_dim, obs_dim=exp_cfg.model.obs_dim, dt=exp_cfg.model.dt)
    q_param = PSDParameter(PSDConfig(exp_cfg.model.state_dim, init_scale=0.1))
    r_param = PSDParameter(PSDConfig(exp_cfg.model.obs_dim, init_scale=0.1))
    ekf = DifferentiableEKF(ekf_cfg, dynamics, measurement, q_param, r_param).to(device)

    checkpoint = Path(to_absolute_path("checkpoint.pt"))
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        dynamics.load_state_dict(state["dynamics"])
        ekf.q_param.load_state_dict(state["q_param"])
        ekf.r_param.load_state_dict(state["r_param"])
        console.log(f"Loaded checkpoint from {checkpoint}")

    ekf.eval()
    all_nis = []
    all_pvalues = []

    for batch in dataloader:
        obs, mask = batch
        obs = obs.to(device)
        mask = mask.to(device)
        batch_size = obs.size(0)
        x0 = torch.zeros(batch_size, exp_cfg.model.state_dim, device=device)
        Sigma0 = torch.eye(exp_cfg.model.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        outputs = ekf(obs, x0, Sigma0, mask=mask)
        delta_y = outputs["innovations"]
        S = outputs["S"]
        logdet_S = outputs["logdet_S"]
        S_inv_delta = torch.linalg.solve(S, delta_y.unsqueeze(-1)).squeeze(-1)
        nis_vals = nis(delta_y, S_inv_delta, mask)
        all_nis.append(nis_vals.mean(dim=1))
        pvalues = ljung_box_pvalues(outputs["whitened"].masked_fill(mask.unsqueeze(-1), float("nan")))
        all_pvalues.append(pvalues.mean(dim=1))

    nis_mean = torch.cat(all_nis).mean().item()
    pvalue_mean = torch.cat(all_pvalues).mean().item()
    console.log({"nis_mean": nis_mean, "ljungbox_p": pvalue_mean})


if __name__ == "__main__":
    main()
