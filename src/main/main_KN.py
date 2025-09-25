"""KalmanNet training/testing main script.

Based on main_DANSE and main_PINN, this CLI handles dataset loading, 
KalmanNetNN instantiation, training loops, and evaluation/logging.
Adjust the configuration block near the top of main() as needed.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from parse import parse
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Runtime path setup so we can import modules under src/ when executing this
# script directly (e.g. python src/main/main_KN.py).
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.KN import KalmanNetNN, train_KalmanNetNN, test_KalmanNetNN
from parameters import get_parameters, get_H_DANSE, f_lorenz_danse, h_fn
from utils.tools import to_float


class NDArrayEncoder(json.JSONEncoder):
    """Minimal JSON encoder that serialises numpy arrays via .tolist()."""

    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SeriesDataset(Dataset):
    """Lightweight dataset wrapper around the pickle structure from generate_data."""

    def __init__(self, Z_XY_dict: Dict):
        data = Z_XY_dict.get("data")
        if data is None:
            raise KeyError("Dataset dict must contain a 'data' key")
        self._observations: List[torch.Tensor] = []
        self._states: List[torch.Tensor] = []
        for Xi, Yi in data:
            Xi_t = torch.as_tensor(Xi, dtype=torch.float32)
            Yi_t = torch.as_tensor(Yi, dtype=torch.float32)
            self._states.append(Xi_t)
            self._observations.append(Yi_t)

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._observations[idx], self._states[idx]

    def initial_state(self) -> torch.Tensor:
        if not self._states:
            raise RuntimeError("Empty dataset: cannot infer initial state")
        return self._states[0][0]


def load_saved_dataset(filename: Path) -> Dict:
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def obtain_tr_val_test_idx(
    dataset: Sequence[Iterable],
    tr_to_test_split: float = 0.9,
    tr_to_val_split: float = 0.833,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    cutoff = int(len(indices) * tr_to_test_split)
    train_val_idx = indices[:cutoff]
    test_idx = indices[cutoff:] if cutoff < len(indices) else indices[-len(indices) // 10 :]

    cutoff_tv = int(len(train_val_idx) * tr_to_val_split)
    train_idx = train_val_idx[:cutoff_tv].tolist()
    val_idx = train_val_idx[cutoff_tv:].tolist()
    return train_idx, val_idx, test_idx.tolist()


def create_splits_file_name(dataset_filename: Path, splits_filename: str) -> Path:
    base_dir = dataset_filename.parent
    return base_dir / splits_filename


def load_splits_file(splits_filename: Path) -> Dict[str, List[int]]:
    with open(splits_filename, "rb") as handle:
        return pickle.load(handle)


def get_dataloaders(
    dataset: Dataset,
    batch_size: int,
    tr_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_subset = Subset(dataset, tr_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def check_if_dir_or_file_exists(base_path: Path, file_name: str | None = None) -> Tuple[bool, bool]:
    dir_exists = base_path.exists() and base_path.is_dir()
    if file_name is None:
        return dir_exists, False
    file_exists = (base_path / file_name).exists() if dir_exists else False
    return dir_exists, file_exists


def build_dynamics(dataset_type: str, ssm_parameters: Dict) -> Tuple:
    if dataset_type == "LorenzSSM":
        return f_lorenz_danse, h_fn
    raise NotImplementedError(f"KalmanNet dynamics not defined for dataset type: {dataset_type}")


def instantiate_model(
    n_states: int,
    n_obs: int,
    n_layers: int,
    model_type: str,
    device: torch.device,
    dataset_type: str,
    ssm_parameters: Dict,
    initial_state: torch.Tensor,
) -> KalmanNetNN:
    model = KalmanNetNN(
        n_states=n_states,
        n_obs=n_obs,
        n_layers=n_layers,
        device=str(device),
        rnn_type=model_type,
    )
    f_func, h_func = build_dynamics(dataset_type, ssm_parameters)
    model.Build(f_func, h_func)
    m1x_0 = initial_state.reshape(-1, 1).to(device)
    model.ssModel = SimpleNamespace(n_states=n_states, n_obs=n_obs, m1x_0=m1x_0)
    model.init_hidden()
    return model


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    mode = "train"  # or "test"
    model_type = "gru"
    dataset_type = "LorenzSSM"
    data_filename = "trajectories_m_3_n_3_LorenzSSM_data_T_200_N_500_r2_0.0dB_nu_-10.0dB.pkl"
    splits_filename = "trajectories_m_3_n_3_LorenzSSM_data_T_200_N_500_r2_0.0dB_nu_-10.0dB_split.pkl"
    checkpoint_name = "knet_ckpt_epoch_best.pt"

    data_path = (PROJECT_ROOT / "src" / "data" / "trajectories" / data_filename).resolve()
    if not data_path.suffix:
        data_path = data_path.with_suffix(".pkl")
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return

    parsed = parse(
        "{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_r2_{}dB_nu_{}dB.pkl",
        data_path.name,
    )
    if parsed is None:
        raise ValueError(f"Cannot parse metadata from dataset filename: {data_path.name}")
    _, n_states, n_obs, _, T, N_samples, inverse_r2_token, nu_token = parsed.fixed

    inverse_r2_dB = to_float(inverse_r2_token)
    nu_dB = to_float(nu_token)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ssm_parameters_dict, est_parameters_dict = get_parameters(
        N=N_samples,
        T=T,
        n_states=n_states,
        n_obs=n_obs,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device,
    )

    estimator_options = est_parameters_dict["KNetUoffline"].copy()
    batch_size = estimator_options.get("batch_size", 100)
    num_epochs = estimator_options.get("num_epochs", 100)
    n_layers = estimator_options.get("n_layers", 1)
    unsupervised = estimator_options.get("unsupervised", True)

    dataset_dict = load_saved_dataset(data_path)
    dataset = SeriesDataset(dataset_dict)

    splits_path = create_splits_file_name(data_path, splits_filename)
    if splits_path.exists():
        splits = load_splits_file(splits_path)
        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    else:
        train_idx, val_idx, test_idx = obtain_tr_val_test_idx(dataset)
        with open(splits_path, "wb") as handle:
            pickle.dump({"train": train_idx, "val": val_idx, "test": test_idx}, handle)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        tr_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    print(
        f"Batches | train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}"
    )

    log_dir = PROJECT_ROOT / "log" / (
        f"{dataset_type}_knet_{model_type}_m_{n_states}_n_{n_obs}_T_{T}_N_{N_samples}_{inverse_r2_dB}dB_{nu_dB}dB"
    )
    model_dir = PROJECT_ROOT / "models" / log_dir.name
    ensure_directory(log_dir)
    ensure_directory(model_dir)

    training_logfile = log_dir / "training.log"
    testing_logfile = log_dir / "testing.log"

    ssm_params = ssm_parameters_dict.get(dataset_type, {})
    initial_state = dataset.initial_state()
    model = instantiate_model(
        n_states=n_states,
        n_obs=n_obs,
        n_layers=n_layers,
        model_type=model_type,
        device=device,
        dataset_type=dataset_type,
        ssm_parameters=ssm_params,
        initial_state=initial_state,
    )

    estimator_options["H"] = get_H_DANSE(dataset_type, n_states, n_obs)

    if mode.lower() == "train":
        tr_verbose = True
        save_chkpoints = "some"
        tr_losses_obs, val_losses_obs, _ = train_KalmanNetNN(
            model=model,
            options=estimator_options,
            train_loader=train_loader,
            val_loader=val_loader,
            nepochs=num_epochs,
            logfile_path=str(training_logfile),
            modelfile_path=str(model_dir),
            save_chkpoints=save_chkpoints,
            device=device,
            tr_verbose=tr_verbose,
            unsupervised=unsupervised,
        )
        losses_path = log_dir / f"knet_{model_type}_losses_eps{num_epochs}.json"
        with open(losses_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train_obs_dB": np.asarray(tr_losses_obs),
                    "val_obs_dB": np.asarray(val_losses_obs),
                },
                handle,
                cls=NDArrayEncoder,
                indent=2,
            )
        print(f"Training complete. Loss curves saved to: {losses_path}")

    elif mode.lower() == "test":
        checkpoint_path = model_dir / checkpoint_name
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        test_loss_state, test_loss_obs, _ = test_KalmanNetNN(
            model_test=model,
            test_loader=test_loader,
            options=estimator_options,
            device=device,
            model_file=str(checkpoint_path),
            test_logfile_path=str(testing_logfile),
        )
        print(
            f"Test losses (dB) | state: {test_loss_state:.3f}, observation: {test_loss_obs:.3f}"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
