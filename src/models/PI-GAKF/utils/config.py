"""Hydra-compatible configuration dataclasses for PI-GAKF."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    path: str = "src/data/trajectories/trajectories_m_3_n_3_LorenzSSM_data_T_200_N_500_r2_0.0dB_nu_-10.0dB.pkl"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True


@dataclass
class ModelConfig:
    state_dim: int = 3
    obs_dim: int = 3
    dt: float = 0.02
    dynamics_hidden: int = 128
    dynamics_depth: int = 3
    measurement: str = "linear"
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_layers: int = 3


@dataclass
class TrainConfig:
    epochs: int = 200
    n_critic: int = 5
    warmup_epochs: int = 20
    lambda_innov: float = 1.0
    lambda_phys: float = 0.1
    lambda_adv: float = 0.1
    lambda_noise: float = 0.0
    lr_G: float = 1e-3
    lr_D: float = 5e-4
    betas_G: Tuple[float, float] = (0.9, 0.999)
    betas_D: Tuple[float, float] = (0.9, 0.999)
    clip_norm: float = 1.0
    amp: bool = False
    ema: bool = False


@dataclass
class EarlyStopConfig:
    window: int = 100
    ljungbox_lag: int = 20
    alpha: float = 0.05
    p_min: float = 0.05
    patience: int = 5


@dataclass
class LoggingConfig:
    run_name: str = "pi_gakf"
    log_dir: str = "runs/pi-gakf"
    tensorboard: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    earlystop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "ExperimentConfig":
        raw: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        return ExperimentConfig(
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
            deterministic=raw.get("deterministic", False),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            earlystop=EarlyStopConfig(**raw.get("earlystop", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "EarlyStopConfig",
    "LoggingConfig",
    "ExperimentConfig",
]
