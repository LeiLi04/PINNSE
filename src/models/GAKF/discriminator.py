from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .spectral import SpectralBranch, spectral_features


class TemporalBackbone(nn.Module):
    """Temporal feature extractor supporting GRU/LSTM or Transformer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        backbone: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.1,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone.lower()
        self.bidirectional = bidirectional

        if self.backbone_type in {"gru", "lstm", "rnn"}:
            rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.backbone_type]
            self.model = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.out_dim = hidden_dim * (2 if bidirectional else 1)
        elif self.backbone_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            )
            self.out_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, n_obs]
        Returns:
            features flatten: [B, out_dim]
        """
        if self.backbone_type == "transformer":
            feats = self.model(x)  # [B, T, hidden]
            feats = feats.mean(dim=1)
        else:
            outputs, _ = self.model(x)
            feats = outputs.mean(dim=1)
        return feats


class GAKFDiscriminator(nn.Module):
    """WGAN-GP discriminator with optional spectral branch."""

    def __init__(
        self,
        n_obs: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        backbone: str = "gru",
        bidirectional: bool = True,
        use_spectral_branch: bool = True,
        spectral_bins: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal = TemporalBackbone(
            input_dim=n_obs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            backbone=backbone,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.use_spectral_branch = use_spectral_branch
        if use_spectral_branch:
            self.spectral_branch = SpectralBranch(
                n_obs=n_obs,
                hidden_dim=hidden_dim,
                num_bins=spectral_bins,
            )
            combined_dim = self.temporal.out_dim + self.spectral_branch.out_dim
        else:
            combined_dim = self.temporal.out_dim

        self.score_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y: Tensor) -> Tensor:
        """
        Args:
            y: [B, T, n_obs]
        Returns:
            scores: [B, 1]
        """
        temporal_feat = self.temporal(y)
        if self.use_spectral_branch:
            spec_feat = self.spectral_branch(spectral_features(y))
            feats = torch.cat([temporal_feat, spec_feat], dim=-1)
        else:
            feats = temporal_feat
        scores = self.score_net(feats)
        return scores.squeeze(-1)
