# ==========================================================================
# 通用数值工具函数
# ==========================================================================
from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

Number = Union[int, float]
ArrayLike = Union[int, float, Sequence[Union[int, float]], np.ndarray]


def to_float(value: Any) -> float:
    """Convert arbitrary numeric-like input to ``float``.

    Strips whitespace, removes trailing ``"dB"`` tokens and normalises decimal
    separators (comma → dot). Raises ``ValueError`` if conversion fails.
    """

    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("None cannot be converted to float")
    normalised = str(value).strip().replace("dB", "").replace(",", ".")
    try:
        return float(normalised)
    except ValueError as exc:
        raise ValueError(f"Cannot convert '{value}' to float") from exc


def dB_to_lin(x: ArrayLike) -> np.ndarray:
    """Convert decibel values to linear scale as NumPy array."""

    arr = np.asarray(x, dtype=float)
    return np.power(10.0, arr / 10.0)


def lin_to_dB(x: ArrayLike) -> np.ndarray:
    """Convert linear values to decibels as NumPy array."""

    arr = np.asarray(x, dtype=float)
    if np.any(arr <= 0):  # includes zero and negative values
        raise ValueError("Linear values must be positive to compute decibels")
    return 10.0 * np.log10(arr)


def partial_corrupt(x: ArrayLike, p: float = 0.7, bias: float = 0.0) -> np.ndarray:
    """Apply proportional corruption with optional bias.

    If elements are negative, the perturbation sign is flipped to preserve
    magnitude inflation behaviour of the original helper.
    """

    arr = np.asarray(x, dtype=float)
    perturb = np.where(arr < 0, -p, p)
    return arr * (1.0 + perturb) + bias


def generate_normal(
    N: int,
    mean: ArrayLike,
    Sigma2: ArrayLike,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``N`` draws from ``N(mean, Sigma2)``.

    Args:
        N: Number of samples to draw.
        mean: Mean vector (1-D array-like).
        Sigma2: Covariance matrix matching ``mean`` dimensionality.
        rng: Optional ``numpy.random.Generator`` for deterministic sampling.
    """

    if N <= 0:
        raise ValueError("N must be positive")

    mean_arr = np.asarray(mean, dtype=float)
    cov_arr = np.asarray(Sigma2, dtype=float)
    if mean_arr.ndim != 1:
        raise ValueError("mean must be a 1-D vector")
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError("Sigma2 must be a square 2-D matrix")
    if cov_arr.shape[0] != mean_arr.size:
        raise ValueError("Mean and covariance dimension mismatch")

    generator = rng if rng is not None else np.random.default_rng()
    return generator.multivariate_normal(mean=mean_arr, cov=cov_arr, size=(N,))


__all__ = [
    "to_float",
    "dB_to_lin",
    "lin_to_dB",
    "partial_corrupt",
    "generate_normal",
]
