from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .dynamics import DynamicsConfig, build_residual_dynamics
from .ekf.ekf import DifferentiableEKF, EKFConfig
from .measurement import LinearMeasurement, LinearMeasurementConfig
from .transformer_critic import TransformerCritic, TransformerCriticConfig
from .training.trainer import PIGAKFTrainer
from .utils.config import ExperimentConfig
from .utils.logging import console, tensorboard_writer
from .utils.psd_param import PSDConfig, PSDParameter


class MeasurementOnlyDataset(Dataset):
    def __init__(self, path: str) -> None:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        obs = []
        for item in data["data"]:
            obs.append(torch.tensor(item[1], dtype=torch.float32))
        self.obs = obs

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return self.obs[idx]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    exp_cfg = ExperimentConfig.from_omegaconf(cfg["experiment"])
    seed_everything(exp_cfg.seed, exp_cfg.deterministic)

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

    dataloader = DataLoader(
        dataset,
        batch_size=exp_cfg.data.batch_size,
        shuffle=exp_cfg.data.shuffle,
        collate_fn=collate,
    )

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

    ekf_cfg = EKFConfig(
        state_dim=exp_cfg.model.state_dim,
        obs_dim=exp_cfg.model.obs_dim,
        dt=exp_cfg.model.dt,
    )

    q_param = PSDParameter(PSDConfig(exp_cfg.model.state_dim, init_scale=0.1))
    r_param = PSDParameter(PSDConfig(exp_cfg.model.obs_dim, init_scale=0.1))

    ekf = DifferentiableEKF(ekf_cfg, dynamics, measurement, q_param, r_param).to(device)

    critic_cfg = TransformerCriticConfig(
        obs_dim=exp_cfg.model.obs_dim,
        d_model=exp_cfg.model.transformer_d_model,
        nhead=exp_cfg.model.transformer_nhead,
        num_layers=exp_cfg.model.transformer_layers,
    )
    critic = TransformerCritic(critic_cfg).to(device)

    generator_params = list(dynamics.parameters()) + list(ekf.q_param.parameters()) + list(ekf.r_param.parameters())
    optim_G = torch.optim.Adam(generator_params, lr=exp_cfg.train.lr_G, betas=exp_cfg.train.betas_G)
    optim_D = torch.optim.Adam(critic.parameters(), lr=exp_cfg.train.lr_D, betas=exp_cfg.train.betas_D)

    scaler = torch.cuda.amp.GradScaler() if exp_cfg.train.amp and device.type == "cuda" else None

    trainer = PIGAKFTrainer(
        ekf=ekf,
        dynamics=dynamics,
        critic=critic,
        generator_opt=optim_G,
        critic_opt=optim_D,
        config=exp_cfg,
        device=device,
        scaler=scaler,
    )

    log_dir = Path(to_absolute_path(exp_cfg.logging.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    with tensorboard_writer(log_dir, exp_cfg.logging.tensorboard) as writer:
        for _ in range(exp_cfg.train.epochs):
            trainer.train_epoch(dataloader, writer=writer)

    checkpoint = log_dir / "checkpoint.pt"
    torch.save(
        {
            "dynamics": dynamics.state_dict(),
            "q_param": ekf.q_param.state_dict(),
            "r_param": ekf.r_param.state_dict(),
        },
        checkpoint,
    )
    console.log(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
