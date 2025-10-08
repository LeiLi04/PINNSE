"""Logging helpers for PI-GAKF."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter


console = Console()


def log_table(title: str, data: Dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    for key, value in data.items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)


@contextmanager
def tensorboard_writer(log_dir: Path, enabled: bool = True):
    writer = SummaryWriter(log_dir=str(log_dir)) if enabled else None
    try:
        yield writer
    finally:
        if writer:
            writer.flush()
            writer.close()


def log_scalars(writer: Optional[SummaryWriter], scalars: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in scalars.items():
        writer.add_scalar(key, value, step)


__all__ = ["console", "log_table", "tensorboard_writer", "log_scalars"]
