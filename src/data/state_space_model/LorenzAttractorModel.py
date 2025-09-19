"""Lorenz attractor based nonlinear state-space model for data synthesis.

项目背景 (Project Background):
    本模块提供 Lorenz 吸引子近似模型，用于生成混沌系统的状态/观测序列，支撑
    滤波器和物理信息网络的基准实验。

输入输出概览 (Input/Output Overview):
    - 初始化阶段提供雅可比函数 A_fn、观测函数 h_fn 以及噪声统计。
    - ``generate_single_sequence`` 返回状态 X ∈ ℝ^[T+1, d] 与观测 Y ∈ ℝ^[T, n]。
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin


class LorenzAttractorModel:
    """Lorenz 吸引子模型 / Lorenz dynamics with Taylor approximation.

    Args:
        d (int): 状态维度。
        J (int): 泰勒展开阶数。
        delta (float): 仿真步长。
        delta_d (float): 可选下采样步长。
        A_fn (Callable): 返回雅可比矩阵 A(x) ∈ ℝ^[d, d]。
        h_fn (Callable): 观测映射函数 h(x) ∈ ℝ^[n]。
        decimate (bool): 是否启用下采样。
        mu_e (np.ndarray | None): 过程噪声均值 [d]。
        mu_w (np.ndarray | None): 观测噪声均值 [n]。
        use_Taylor (bool): 是否采用泰勒展开近似。

    Tensor Dimensions:
        - X: [T+1, d]
        - Y: [T, n]

    Math Notes:
        - F(x) = I + Σ_{j=1..J} ((A(x) * delta)^j / j!)
        - x_next = F(x) @ x + e
        - y = h(x) + v
    """

    def __init__(
        self,
        d: int,
        J: int,
        delta: float,
        delta_d: float,
        A_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        h_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        decimate: bool = False,
        mu_e: Optional[np.ndarray] = None,
        mu_w: Optional[np.ndarray] = None,
        use_Taylor: bool = True,
    ) -> None:
        self.n_states = d
        self.J = J
        self.delta = delta
        self.delta_d = delta_d
        self.A_fn = A_fn if A_fn is not None else (lambda x: np.eye(d))
        self._h_fn = h_fn if h_fn is not None else (lambda x: x)
        self.n_obs = self._h_fn(np.random.randn(d, 1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e if mu_e is not None else np.zeros(d)
        self.mu_w = mu_w if mu_w is not None else np.zeros(self.n_obs)
        self.use_Taylor = use_Taylor

        self.Q = None
        self.R = None
        self.q2 = None
        self.r2 = None

    def h_fn(self, x: np.ndarray) -> np.ndarray:
        """观测映射封装 / Observation mapping wrapper."""

        return self._h_fn(x)

    def f_linearize(self, x: np.ndarray) -> np.ndarray:
        """泰勒截断矩阵指数 / One-step prediction.

        Args:
            x (np.ndarray): 当前状态向量 [d]。

        Returns:
            np.ndarray: 预测状态向量 [d]。

        Math Notes:
            - F = I + Σ_{j=1..J} (A(x) * delta)^j / j!
            - x_next = F @ x
        """

        if not self.use_Taylor:
            return self.A_fn(x) @ x

        # ==================================================================
        # STEP 01: 初始化基矩阵 (Start from identity)
        # ------------------------------------------------------------------
        self.F = np.eye(self.n_states)
        # ==================================================================
        # STEP 02: 累加泰勒项 (Accumulate Taylor terms)
        # ------------------------------------------------------------------
        for j in range(1, self.J + 1):
            self.F += np.linalg.matrix_power(self.A_fn(x) * self.delta, j) / math.factorial(j)
        return self.F @ x

    def init_noise_covs(self) -> None:
        """根据 q2/r2 初始化对角协方差 / Init diagonal covariances."""

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float,
        nu_dB: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成 Lorenz 轨迹与观测 / Generate single trajectory.

        Args:
            T (int): 序列长度。
            inverse_r2_dB (float): 观测噪声逆功率 (dB)。
            nu_dB (float): 过程与观测噪声比 (dB)。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - x_lorenz_d: 状态序列 [T+1, d]
                - y_lorenz_d: 观测序列 [T, n]

        Tensor Dimensions:
            - e: [T+1, d]
            - v: [T, n]

        Math Notes:
            - r2 = 1 / dB_to_lin(inverse_r2_dB)
            - q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            - x_{t+1} = f_linearize(x_t) + e_t
            - y_t = h(x_t) + v_t
        """

        x = np.zeros((T + 1, self.n_states))
        y = np.zeros((T, self.n_obs))

        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()

        # ==================================================================
        # STEP 01: 采样过程/观测噪声 (Noise sampling)
        # ------------------------------------------------------------------
        e = np.random.multivariate_normal(self.mu_e, self.Q, size=(T + 1,))
        v = np.random.multivariate_normal(self.mu_w, self.R, size=(T,))

        # ==================================================================
        # STEP 02: 前向模拟 Lorenz 系统 (Forward rollout)
        # ------------------------------------------------------------------
        for t in range(T):
            x[t + 1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]

        if self.decimate:
            K = max(int(round(self.delta_d / self.delta)), 1)
            x_lorenz_d = x[0:T:K, :]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(
                self.mu_w, self.R, size=(len(x_lorenz_d),)
            )
        else:
            x_lorenz_d = x
            y_lorenz_d = y

        return x_lorenz_d, y_lorenz_d
