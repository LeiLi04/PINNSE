from __future__ import annotations

import torch
from torch import Tensor, nn


def spectral_features(y: Tensor, num_bins: int = 64) -> Tensor:
    """
    Compute magnitude spectrogram / PSD-like features.
    Args:
        y: [B, T, n_obs]
    Returns:
        features: [B, n_obs, num_bins]
    """
    B, T, n_obs = y.shape
    fft = torch.fft.rfft(y, dim=1, norm="ortho")  # [B, F, n_obs]
    power = fft.abs() ** 2
    F = power.size(1)
    if F <= num_bins:
        pad = num_bins - F
        power = torch.nn.functional.pad(power, (0, 0, 0, pad))
        features = power[:, :num_bins, :].transpose(1, 2)
    else:
        step = F // num_bins
        features = power.unfold(dimension=1, size=step, step=step).mean(dim=3).transpose(1, 2)
        features = features[:, :, :num_bins]
    return features


class SpectralBranch(nn.Module):
    """Simple MLP over spectral features."""

    def __init__(
        self,
        n_obs: int,
        hidden_dim: int = 128,
        num_bins: int = 64,
    ) -> None:
        super().__init__()
        self.out_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_obs * num_bins, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.num_bins = num_bins

    def forward(self, spectral_feat: Tensor) -> Tensor:
        return self.mlp(spectral_feat)
