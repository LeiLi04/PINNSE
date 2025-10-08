"""Sequence-level Transformer critic for PI-GAKF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TransformerCriticConfig:
    obs_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    use_cls_token: bool = True


class TransformerCritic(nn.Module):
    """Transformer encoder that maps sequences to scalar critic scores."""

    def __init__(self, config: TransformerCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.obs_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            self.register_parameter("cls_token", None)
        self.score_head = nn.Linear(config.d_model, 1)

    def forward(self, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute critic scores for a batch of sequences.

        Args:
            y: Observation sequence `(B, T, obs_dim)`.
            mask: Boolean mask `(B, T)` where ``True`` indicates padding to ignore.
        Returns:
            Tensor `(B,)` with critic scores.
        """

        x = self.input_proj(y)
        if self.cls_token is not None:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        if self.cls_token is not None:
            pooled = encoded[:, 0]
        else:
            if mask is None:
                pooled = encoded.mean(dim=1)
            else:
                valid = (~mask).float()
                pooled = (encoded * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        scores = self.score_head(pooled)
        return scores.squeeze(-1)


__all__ = ["TransformerCritic", "TransformerCriticConfig"]
