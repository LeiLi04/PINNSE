"""Visualization helpers for comparing state-space trajectories.

项目背景 (Project Background):
    提供 KF/EKF/UKF/DANSE/KalmanNet/PINNSE 等估计结果的轨迹可视化能力，用于论文绘图
    与实验对比。

输入输出概览 (Input/Output Overview):
    - 接收真实状态/观测矩阵与多个估计矩阵，绘制 2D/3D 图像并可选保存或导出 TikZ。
    - 返回 Matplotlib Figure，便于进一步定制。
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
    """验证输入矩阵为二维 / Ensure input is a 2-D array.

    Args:
        name (str): 变量名称，用于错误信息。
        arr (np.ndarray): 待验证数组。

    Returns:
        np.ndarray: 原始数组，确保为二维。

    Tensor Dimensions:
        - arr: [T, D]
        - return: [T, D]

    Math Notes:
        不涉及数学变换，仅做结构校验。
    """

    # ==================================================================
    # STEP 01: 类型与维度检查 (Check type and dimensionality)
    # ------------------------------------------------------------------
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [T, D], got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def _check_matching_shape(reference: Array2D, candidate: Optional[Array2D], label: str) -> Optional[Array2D]:
    """形状校验辅助 / Ensure candidate matches reference shape.

    Args:
        reference (np.ndarray): 基准数组 [T, D]。
        candidate (np.ndarray | None): 待校验数组。
        label (str): 变量名称。

    Returns:
        np.ndarray | None: 若校验通过则返回原数组，否则 ``None``。

    Tensor Dimensions:
        - reference: [T, D]
        - candidate: [T, D]

    Math Notes:
        仅执行维度一致性检查，无数学运算。
    """

    # ==================================================================
    # STEP 01: 判空与维度一致性 (Handle None and shape check)
    # ------------------------------------------------------------------
    if candidate is None:
        return None
    candidate = _ensure_2d(label, candidate)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"{label} must match reference shape {reference.shape}, got {candidate.shape}"
        )
    return candidate


def _maybe_save(fig: plt.Figure, savefig: bool, savefig_name: Optional[Path | str]) -> None:
    """可选保存图像 / Save figure and optional TikZ export.

    Args:
        fig (plt.Figure): Matplotlib 图对象。
        savefig (bool): 是否保存图像。
        savefig_name (Path | str | None): 输出路径，若 ``None`` 使用默认名。

    Returns:
        None: 无返回值。

    Tensor Dimensions:
        - 不涉及张量操作。

    Math Notes:
        无数学计算，仅文件 IO。
    """

    # ==================================================================
    # STEP 01: 判断是否需要保存 (Check save flag)
    # ------------------------------------------------------------------
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
    """根据环境选择显示或关闭图像 / Conditionally display figure.

    Args:
        fig (plt.Figure): 图对象。
        show (bool): True 时显示，False 时关闭以释放资源。

    Returns:
        None: 无返回值。
    """

    # ==================================================================
    # STEP 01: 根据标志显示或关闭 (Show or close figure)
    # ------------------------------------------------------------------
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
    """绘制单维度时间曲线 / Plot component-wise trajectories.

    Args:
        ax (plt.Axes): 子图坐标轴。
        t (np.ndarray): 时间索引 [T]。
        reference (np.ndarray): 真实轨迹 [T, D]。
        series (Mapping[str, Tuple[np.ndarray, str]]): 估计轨迹及线型。
        dim (int): 选择绘制的维度索引。

    Returns:
        None: 无返回值。

    Tensor Dimensions:
        - reference: [T, D]
        - series[k][0]: [T, D]

    Math Notes:
        仅执行线性绘图，无数值运算。
    """

    # ==================================================================
    # STEP 01: 绘制真实与估计曲线 (Plot ground truth and estimators)
    # ------------------------------------------------------------------
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
    """绘制状态分量时间曲线 / Plot component trajectories for estimators.

    Args:
        X (np.ndarray): 真实状态轨迹 [T, D]。
        X_est_KF (np.ndarray | None): KF 估计 [T, D]。
        X_est_EKF (np.ndarray | None): EKF 估计 [T, D]。
        X_est_UKF (np.ndarray | None): UKF 估计 [T, D]。
        X_est_DANSE (np.ndarray | None): DANSE 估计 [T, D]。
        X_est_KNET (np.ndarray | None): KalmanNet 估计 [T, D]。
        X_est_PINN (np.ndarray | None): PINNSE 估计 [T, D]。
        savefig (bool): 是否保存图像。
        savefig_name (Path | str | None): 保存路径。
        show (bool): 是否显示图像。

    Returns:
        plt.Figure: 对应的 Matplotlib Figure 对象。

    Tensor Dimensions:
        - X: [T, D]
        - X_est_*: [T, D]

    Math Notes:
        仅执行绘图，无额外数学运算。
    """

    # ==================================================================
    # STEP 01: 校验输入并收集估计曲线 (Validate shapes and collect estimators)
    # ==================================================================
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

    # ==================================================================
    # STEP 02: 绘制每个状态分量 (Plot component lines)
    # ==================================================================
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
    """绘制观测轨迹 / Visualise measurement data in 2-D or 3-D.

    Args:
        Y (np.ndarray): 观测数组 [T, D]，D 取 2 或 3。
        savefig (bool): 是否保存图像。
        savefig_name (Path | str | None): 图像保存路径。
        show (bool): 是否显示图像。

    Returns:
        plt.Figure: Matplotlib Figure 对象。

    Tensor Dimensions:
        - Y: [T, D]

    Math Notes:
        绘制曲线，无额外数学运算。
    """

    # ==================================================================
    # STEP 01: 校验输入并创建画布 (Validate input and create figure)
    # ==================================================================
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
    """多子图展示状态分量 / Plot state components per subplot.

    Args:
        X (np.ndarray): 真实状态轨迹 [T, D]。
        X_est_KF (np.ndarray | None): KF 估计 [T, D]。
        X_est_EKF (np.ndarray | None): EKF 估计 [T, D]。
        X_est_UKF (np.ndarray | None): UKF 估计 [T, D]。
        X_est_KNET (np.ndarray | None): KalmanNet 估计 [T, D]。
        X_est_DANSE (np.ndarray | None): DANSE 估计 [T, D]。
        X_est_PINN (np.ndarray | None): PINNSE 估计 [T, D]。
        savefig (bool): 是否保存图像。
        savefig_name (Path | str | None): 保存路径。
        show (bool): 是否显示图像。

    Returns:
        plt.Figure: 多子图的 Figure 对象。

    Tensor Dimensions:
        - X: [T, D]
        - X_est_*: [T, D]

    Math Notes:
        纯绘图操作，无数值运算。
    """

    # ==================================================================
    # STEP 01: 校验输入并整理估计序列 (Validate input and collect estimators)
    # ==================================================================
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
