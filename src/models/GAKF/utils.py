from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.parameters import get_parameters, get_H_DANSE

try:
    import yaml
except ImportError as exc:
    raise ImportError("Please install pyyaml to load configuration files.") from exc


def load_config(path: Path, overrides: Optional[Dict] = None) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    overrides = overrides or {}
    return _merge_dict(config, overrides)


def _merge_dict(base: Dict, overrides: Dict) -> Dict:
    merged = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in merged:
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


class TrajectoryDataset(Dataset):
    """Wrap pickle dataset Z_XY -> [X_i, Y_i]."""

    def __init__(self, data_pairs: Sequence[Sequence[np.ndarray]]):
        self.states: List[torch.Tensor] = []
        self.observations: List[torch.Tensor] = []
        for pair in data_pairs:
            x, y = pair
            self.states.append(torch.tensor(x, dtype=torch.float32))
            self.observations.append(torch.tensor(y, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.observations[idx], self.states[idx]


def parse_dataset_file(
    dataset_dir: Path,
    dataset_type: str,
    n_states: int,
    n_obs: int,
    T: int,
) -> Path:
    """
    Attempt to find dataset file matching metadata.
    """
    candidates = []
    for file in dataset_dir.glob("*.pkl"):
        stem = file.stem
        if dataset_type not in stem:
            continue
        if f"_m_{n_states}_" not in stem or f"_n_{n_obs}_" not in stem:
            continue
        if f"_T_{T}_" not in stem:
            continue
        candidates.append(file)
    if not candidates:
        raise FileNotFoundError(
            f"No dataset files matching type={dataset_type}, m={n_states}, n={n_obs}, T={T} in {dataset_dir}"
        )
    return sorted(candidates)[0]


def load_pickle_dataset(path: Path) -> Sequence[Sequence[np.ndarray]]:
    import pickle

    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if "data" not in data:
        raise KeyError("Expected key 'data' in dataset pickle.")
    return data["data"]


def create_dataloaders(
    dataset_pairs: Sequence[Sequence[np.ndarray]],
    batch_size: int,
    N_train: int,
    N_val: int,
) -> Tuple[DataLoader, DataLoader]:
    dataset = TrajectoryDataset(dataset_pairs)
    indices = np.random.permutation(len(dataset))[: N_train + N_val]
    train_idx = indices[:N_train]
    val_idx = indices[N_train : N_train + N_val]

    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def prepare_parameters(config: Dict, device: torch.device) -> Dict:
    data_cfg = config["data"]
    ssm_params, _ = get_parameters(
        N=data_cfg.get("N_train", 1024),
        T=data_cfg["T"],
        n_states=data_cfg["n_states"],
        n_obs=data_cfg["n_obs"],
        inverse_r2_dB=data_cfg["inverse_r2_dB"],
        nu_dB=data_cfg["nu_dB"],
        device=device,
    )
    dataset_type = data_cfg["dataset_type"]
    ssm = ssm_params[dataset_type]
    n_states = ssm["n_states"]
    n_obs = ssm["n_obs"]
    H = ssm.get("H")
    if H is None:
        H = get_H_DANSE(dataset_type, n_states=n_states, n_obs=n_obs)
    R_var = 1.0 / (10 ** (ssm["inverse_r2_dB"] / 10.0))
    R = torch.eye(n_obs, device=device, dtype=torch.float32) * R_var
    return {"H": torch.tensor(H, device=device, dtype=torch.float32), "R": R}


def make_summary_writer(out_dir: Path) -> SummaryWriter:
    out_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(out_dir))


def compute_covariance(x: Tensor) -> Tensor:
    """Unbiased covariance estimator over dim 0."""
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = (x_centered.transpose(0, 1) @ x_centered) / max(x_centered.size(0) - 1, 1)
    return cov


@dataclass
class MetricAccumulator:
    """Simple running mean tracker."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def save_checkpoint(
    path: Path,
    generator_state: Dict,
    discriminator_state: Dict,
    optim_G_state: Dict,
    optim_D_state: Dict,
    step: int,
    best_metric: float,
) -> None:
    payload = {
        "generator": generator_state,
        "discriminator": discriminator_state,
        "optim_G": optim_G_state,
        "optim_D": optim_D_state,
        "step": step,
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: Optional[str | torch.device] = None) -> Dict:
    return torch.load(path, map_location=map_location)
