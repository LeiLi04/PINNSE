from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENTS = CURRENT_DIR.parents
PROJECT_ROOT = PARENTS[3] if len(PARENTS) >= 4 else CURRENT_DIR
SRC_ROOT = PARENTS[2] if len(PARENTS) >= 3 else CURRENT_DIR
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.models.GAKF.generator import GAKFGenerator
from src.models.GAKF import utils

import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAKF on validation/test set.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint to load.")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs"))
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = utils.load_config(args.config)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    params = utils.prepare_parameters(config, device)
    H = params["H"]
    R = params["R"]
    data_cfg = config["data"]

    dataset_dir = Path(data_cfg["dataset_path"])
    dataset_file = (
        dataset_dir / data_cfg.get("dataset_file")
        if "dataset_file" in data_cfg
        else utils.parse_dataset_file(
            dataset_dir,
            data_cfg["dataset_type"],
            data_cfg["n_states"],
            data_cfg["n_obs"],
            data_cfg["T"],
        )
    )
    dataset_pairs = utils.load_pickle_dataset(dataset_file)
    _, val_loader = utils.create_dataloaders(
        dataset_pairs,
        batch_size=config["train"]["batch_size"],
        N_train=data_cfg["N_train"],
        N_val=data_cfg["N_val"],
    )

    gen_cfg = config["model"]
    generator = GAKFGenerator(
        n_obs=data_cfg["n_obs"],
        n_states=data_cfg["n_states"],
        H=H,
        R=R,
        hidden_dim=gen_cfg["hidden"],
        num_layers=gen_cfg["layers"],
        rnn_type=gen_cfg["rnn"],
        diag_cov=gen_cfg.get("diag_cov", True),
        lowrank=gen_cfg.get("lowrank", False),
        device=device,
    ).to(device)
    checkpoint = utils.load_checkpoint(args.ckpt, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    args.output.mkdir(parents=True, exist_ok=True)

    rmse_meter = utils.MetricAccumulator()
    mae_meter = utils.MetricAccumulator()

    for idx, (y_batch, x_batch) in enumerate(val_loader):
        y_batch = y_batch.to(device)
        x_batch = x_batch.to(device)
        outs = generator(y_batch)

        error = outs.x_post[:, 1:, :] - x_batch[:, 1:, :]
        rmse = torch.sqrt((error ** 2).mean(dim=(1, 2)))
        mae = error.abs().mean(dim=(1, 2))
        rmse_meter.update(rmse.mean().item(), y_batch.size(0))
        mae_meter.update(mae.mean().item(), y_batch.size(0))

        if idx == 0:
            plot_trajectories(x_batch, outs.x_post, args.output / "trajectory.png")
            plot_innovation_histograms(
                outs.aux["innovation_white"],
                args.output / "innovation_hist.png",
            )

    with open(args.output / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse_meter.avg:.6f}\n")
        f.write(f"MAE: {mae_meter.avg:.6f}\n")


def plot_trajectories(x_true: torch.Tensor, x_est: torch.Tensor, path: Path) -> None:
    """Save trajectory comparison plots."""
    x_true_np = x_true.detach().cpu().numpy()
    x_est_np = x_est.detach().cpu().numpy()
    T = x_true_np.shape[1]
    num_states = x_true_np.shape[2]

    fig, axes = plt.subplots(num_states, 1, figsize=(10, 3 * num_states))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for i in range(num_states):
        axes[i].plot(range(T), x_true_np[0, :, i], label="True")
        axes[i].plot(range(T), x_est_np[0, :, i], label="Estimate", linestyle="--")
        axes[i].set_title(f"State dimension {i}")
        axes[i].set_xlabel("t")
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_innovation_histograms(innovation_white: torch.Tensor, path: Path) -> None:
    """Plot histogram of whitened innovations."""
    data = innovation_white.cpu().numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=50, density=True, alpha=0.7, label="Whitened innovation")
    ax.set_title("Innovation whiteness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
