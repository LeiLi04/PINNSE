"""Plotting utilities for PI-GAKF diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def plot_innovation_histogram(innovations: torch.Tensor, save_path: Optional[Path] = None) -> None:
    data = innovations.detach().cpu().numpy().reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, density=True, alpha=0.7)
    plt.title("Innovation Histogram")
    plt.xlabel("Innovation")
    plt.ylabel("Density")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_acf(innovations: torch.Tensor, save_path: Optional[Path] = None) -> None:
    from statsmodels.graphics.tsaplots import plot_acf

    series = innovations.detach().cpu().numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_acf(series, ax=ax, lags=40)
    ax.set_title("Innovation ACF")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


__all__ = ["plot_innovation_histogram", "plot_acf"]
