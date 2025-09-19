"""Numerical helpers for signal scaling and sampling.

项目背景 (Project Background):
    提供 dB 与线性尺度互转、噪声采样等通用工具，支撑 Kalman 滤波与数据模拟。

输入输出概览 (Input/Output Overview):
    - 接受标量/数组输入并输出 numpy.ndarray 或 float 结果。
    - ``generate_normal`` 依据均值与协方差生成样本矩阵。
"""

# ==========================================================================
# 通用数值工具函数
# ==========================================================================
from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

Number = Union[int, float]
ArrayLike = Union[int, float, Sequence[Union[int, float]], np.ndarray]


def to_float(value: Any) -> float:
    """字符串转浮点 / Convert numeric-like token into float.

    Args:
        value (Any): 输入 scalar 或字符串，可包含空格、`dB` 字样、逗号小数点。

    Returns:
        float: 清洗后的浮点值。

    Tensor Dimensions:
        - value: 标量。
        - return: 标量。

    Math Notes:
        value_float = float(cleaned_string)
    """

    # ==================================================================
    # STEP 01: 直接处理数值类型 (Handle numeric inputs early)
    # ------------------------------------------------------------------
    if isinstance(value, (int, float)):
        return float(value)

    # ==================================================================
    # STEP 02: 清洗字符串并转换 (Normalise string tokens)
    # ------------------------------------------------------------------
    if value is None:
        raise ValueError("None cannot be converted to float")
    normalised = str(value).strip().replace("dB", "").replace(",", ".")
    try:
        return float(normalised)
    except ValueError as exc:
        raise ValueError(f"Cannot convert '{value}' to float") from exc


def dB_to_lin(x: ArrayLike) -> np.ndarray:
    """dB 转线性刻度 / Convert decibels to linear scale.

    Args:
        x (ArrayLike): dB 数值，标量或数组。

    Returns:
        np.ndarray: 线性刻度数组，与输入形状一致。

    Tensor Dimensions:
        - x: [*]
        - return: [*]

    Math Notes:
        lin = 10 ** (x / 10)
    """

    arr = np.asarray(x, dtype=float)
    return np.power(10.0, arr / 10.0)


def lin_to_dB(x: ArrayLike) -> np.ndarray:
    """线性刻度转 dB / Convert linear values to decibels.

    Args:
        x (ArrayLike): 线性尺度数值，需全部为正。

    Returns:
        np.ndarray: dB 数值数组，与输入形状一致。

    Tensor Dimensions:
        - x: [*]
        - return: [*]

    Math Notes:
        dB = 10 * log10(x)
    """

    arr = np.asarray(x, dtype=float)
    if np.any(arr <= 0):  # includes zero and negative values
        raise ValueError("Linear values must be positive to compute decibels")
    return 10.0 * np.log10(arr)


def partial_corrupt(x: ArrayLike, p: float = 0.7, bias: float = 0.0) -> np.ndarray:
    """扰动信号并加偏置 / Apply proportional corruption.

    Args:
        x (ArrayLike): 待扰动数值，标量或数组。
        p (float): 扰动比例，默认 0.7。
        bias (float): 附加偏置项。

    Returns:
        np.ndarray: 扰动后的数值数组，形状与输入一致。

    Tensor Dimensions:
        - x: [*]
        - return: [*]

    Math Notes:
        perturb = p if x >= 0 else -p
        y = x * (1 + perturb) + bias
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
    """采样多元高斯 / Draw samples from N(mean, Sigma2).

    Args:
        N (int): 样本数量。
        mean (ArrayLike): 均值向量 [d]。
        Sigma2 (ArrayLike): 协方差矩阵 [d, d]。
        rng (np.random.Generator | None): 可选随机数生成器。

    Returns:
        np.ndarray: 样本矩阵，形状 [N, d]。

    Tensor Dimensions:
        - mean: [d]
        - Sigma2: [d, d]
        - return: [N, d]

    Math Notes:
        samples = rng.multivariate_normal(mean, Sigma2, size=N)
    """

    # ==================================================================
    # STEP 01: 参数校验 (Validate shapes and sizes)
    # ------------------------------------------------------------------
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

    # ==================================================================
    # STEP 02: 选择随机源并采样 (Select RNG and draw samples)
    # ------------------------------------------------------------------
    generator = rng if rng is not None else np.random.default_rng()
    return generator.multivariate_normal(mean=mean_arr, cov=cov_arr, size=(N,))


__all__ = [
    "to_float",
    "dB_to_lin",
    "lin_to_dB",
    "partial_corrupt",
    "generate_normal",
]
