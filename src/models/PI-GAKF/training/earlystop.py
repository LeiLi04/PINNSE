"""Early-stopping utilities based on NIS and Ljung–Box diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .metrics import ljung_box_pvalues


@dataclass
class EarlyStoppingState:
    best_metric: float = float("inf")
    patience_counter: int = 0
    triggered: bool = False


class ConsistencyEarlyStopper:
    """Monitor NIS/Ljung–Box statistics and trigger early stopping."""

    def __init__(self, window: int, p_min: float, patience: int) -> None:
        self.window = window
        self.p_min = p_min
        self.patience = patience
        self.state = EarlyStoppingState()
        self.history: List[float] = []

    def update(self, nis_window: torch.Tensor, innovations: torch.Tensor) -> bool:
        """Update the stopper with latest statistics.

        Args:
            nis_window: Tensor `(B,)` aggregated over the window.
            innovations: Whitened innovations `(B, T, n)` for Ljung–Box.
        Returns:
            ``True`` if training should stop.
        """

        avg_nis = nis_window.mean().item()
        ljung_p = ljung_box_pvalues(innovations).min().item()
        metric = avg_nis - ljung_p  # heuristic: lower is better
        self.history.append(metric)

        if metric < self.state.best_metric and ljung_p > self.p_min:
            self.state.best_metric = metric
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1
            if self.state.patience_counter >= self.patience:
                self.state.triggered = True
        return self.state.triggered


__all__ = ["ConsistencyEarlyStopper", "EarlyStoppingState"]
