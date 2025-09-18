"""Visualization helpers for comparing state-space trajectories.

The utilities here provide lightweight wrappers around Matplotlib to render
2-D/3-D trajectories produced by different estimators (KF, EKF, UKF, DANSE,
KalmanNet, PINNSE, ...).  They are safe to use in headless environments and can
optionally persist figures to disk or export TikZ representations when
``tikzplotlib`` is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

_TIKZ_AVAILABLE = True
_TIKZ_ERRMSG = ""
try:  # pragma: no cover - tiny optional dependency shim
    import tikzplotlib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    _TIKZ_AVAILABLE = False
    _TIKZ_ERRMSG = str(exc)

Array2D = np.ndarray


def _ensure_2d(name: str, arr: Array2D) -> Array2D:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [T, D], got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def _check_matching_shape(reference: Array2D, candidate: Optional[Array2D], label: str) -> Optional[Array2D]:
    if candidate is None:
        return None
    candidate = _ensure_2d(label, candidate)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"{label} must match reference shape {reference.shape}, got {candidate.shape}"
        )
    return candidate


def _maybe_save(fig: plt.Figure, savefig: bool, savefig_name: Optional[Path | str]) -> None:
    if not savefig:
        return
    path = Path(savefig_name or "figure.png").expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if _TIKZ_AVAILABLE:
        try:  # pragma: no cover - exercised only when tikzplotlib is present
            tikzplotlib.save(path.with_suffix(".tex"))  # type: ignore
        except Exception as exc:  # pragma: no cover - optional export diagnostics
            print(f"[tikzplotlib] export failed: {exc}")
    elif savefig_name is not None:
        print(
            "[tikzplotlib] unavailable, skipping .tex export."
            + (f" Reason: {_TIKZ_ERRMSG}" if _TIKZ_ERRMSG else "")
        )


def _maybe_show(fig: plt.Figure, show: bool) -> None:
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def _plot_component_lines(
    ax: plt.Axes,
    t: np.ndarray,
    reference: Array2D,
    series: Mapping[str, Tuple[Array2D, str]],
    dim: int,
) -> None:
    ax.plot(t, reference[:, dim], label=f"True x{dim + 1}", linestyle="-", linewidth=1.6)
    for label, (values, style) in series.items():
        ax.plot(t, values[:, dim], style, label=f"{label} x{dim + 1}", linewidth=1.2)
    ax.set_ylabel(f"state_{dim + 1}")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_state_trajectory(
    X: Array2D,
    X_est_KF: Optional[Array2D] = None,
    X_est_EKF: Optional[Array2D] = None,
    X_est_UKF: Optional[Array2D] = None,
    X_est_DANSE: Optional[Array2D] = None,
    X_est_KNET: Optional[Array2D] = None,
    X_est_PINN: Optional[Array2D] = None,
    *,
    savefig: bool = False,
    savefig_name: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot each state component trajectory against estimator outputs.

    Returns the Matplotlib ``Figure`` for further tweaking or testing.
    """

    X = _ensure_2d("X", X)
    estimates: Dict[str, Tuple[Array2D, str]] = {}
    for label, data, style in (
        ("KF", X_est_KF, ":"),
        ("EKF", X_est_EKF, "--"),
        ("UKF", X_est_UKF, "-."),
        ("DANSE", X_est_DANSE, "r--"),
        ("KalmanNet", X_est_KNET, "c-."),
        ("PINNSE", X_est_PINN, "m^-"),
    ):
        checked = _check_matching_shape(X, data, f"X_est_{label}")
        if checked is not None:
            estimates[label] = (checked, style)

    fig, ax = plt.subplots()
    t = np.arange(X.shape[0])
    for dim in range(X.shape[1]):
        ax.plot(t, X[:, dim], linestyle="-", linewidth=1.6, label=f"True x{dim + 1}")
        for label, (values, style) in estimates.items():
            ax.plot(t, values[:, dim], style, linewidth=1.2, label=f"{label} x{dim + 1}")
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    _maybe_save(fig, savefig, savefig_name)
    _maybe_show(fig, show)
    return fig


def plot_measurement_data(
    Y: Array2D,
    *,
    savefig: bool = False,
    savefig_name: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot measured trajectories in 2-D or 3-D."""

    Y = _ensure_2d("Y", Y)
    fig = plt.figure()
    if Y.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.plot(Y[:, 0], Y[:, 1], "--", label=r"$\mathbf{y}^{measured}$")
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
    elif Y.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "--", label=r"$\mathbf{y}^{measured}$")
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
        ax.set_zlabel(r"$Y_3$")
    else:
        raise ValueError("Y must have 2 or 3 columns for plotting")
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, savefig, savefig_name)
    _maybe_show(fig, show)
    return fig


def plot_state_trajectory_axes(
    X: Array2D,
    X_est_KF: Optional[Array2D] = None,
    X_est_EKF: Optional[Array2D] = None,
    X_est_UKF: Optional[Array2D] = None,
    X_est_KNET: Optional[Array2D] = None,
    X_est_DANSE: Optional[Array2D] = None,
    X_est_PINN: Optional[Array2D] = None,
    *,
    savefig: bool = False,
    savefig_name: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot each state component as a separate subplot against estimators."""

    X = _ensure_2d("X", X)
    estimates_ordered = [
        ("KF", X_est_KF, ":"),
        ("EKF", X_est_EKF, "b.-"),
        ("UKF", X_est_UKF, "-x"),
        ("KalmanNet", X_est_KNET, "c-."),
        ("DANSE", X_est_DANSE, "r--"),
        ("PINNSE", X_est_PINN, "m^-"),
    ]
    validated: Dict[str, Tuple[Array2D, str]] = {}
    for label, data, style in estimates_ordered:
        checked = _check_matching_shape(X, data, f"X_est_{label}")
        if checked is not None:
            validated[label] = (checked, style)

    num_dims = X.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(8, 3 * num_dims), sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(X.shape[0])
    for dim, ax in enumerate(axes):
        _plot_component_lines(ax, t, X, validated, dim)
    axes[-1].set_xlabel("t")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=3, loc="upper right")
    fig.tight_layout()

    _maybe_save(fig, savefig, savefig_name)
    _maybe_show(fig, show)
    return fig


__all__ = [
    "plot_state_trajectory",
    "plot_measurement_data",
    "plot_state_trajectory_axes",
]
