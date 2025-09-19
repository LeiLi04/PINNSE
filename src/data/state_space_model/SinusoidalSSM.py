"""Sinusoidal nonlinear state-space model for synthetic benchmarks.

项目背景 (Project Background):
    该模块实现正弦驱动的非线性 SSM，用于产生周期性状态序列并验证滤波/PINN 模型。

输入输出概览 (Input/Output Overview):
    - 初始化阶段配置状态维度、振幅/频率及噪声统计信息。
    - ``generate_single_sequence`` 输出状态矩阵 X ∈ ℝ^[T+1, m] 与观测矩阵 Y ∈ ℝ^[T, n]。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin


class SinusoidalSSM:
    """正弦型状态空间模型 / Sinusoidal SSM with affine observation.

    Args:
        n_states (int): 状态维度。
        alpha (float): 状态更新振幅系数。
        beta (float): 状态更新频率系数。
        phi (float): 相位偏移量。
        delta (float): 状态平移项。
        delta_d (float | None): 下采样步长；默认与 ``delta`` 相同。
        a, b, c (float): 观测仿射参数。
        decimate (bool): 是否启用下采样。
        mu_e (np.ndarray | None): 过程噪声均值 [m]。
        mu_w (np.ndarray | None): 观测噪声均值 [n]。
        use_Taylor (bool): 占位参数，确保接口兼容。

    Tensor Dimensions:
        - X: [T+1, m]
        - Y: [T, n]

    Math Notes:
        - x_next = alpha * sin(beta * x + phi) + delta + e
        - y = a * (b * x + c) + v
    """

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.9,
        beta: float = 1.1,
        phi: float = 0.1 * np.pi,
        delta: float = 0.01,
        delta_d: Optional[float] = None,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.0,
        decimate: bool = False,
        mu_e: Optional[np.ndarray] = None,
        mu_w: Optional[np.ndarray] = None,
        use_Taylor: bool = False,
    ) -> None:
        # ==================================================================
        # STEP 01: 基础属性设置 (Store dynamics and observation params)
        # ==================================================================
        self.n_states = n_states
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.a = a
        self.b = b
        self.c = c
        self.delta_d = delta_d if delta_d is not None else delta
        # ==================================================================
        # STEP 02: 观测维度与噪声统计 (Infer dims and noise means)
        # ==================================================================
        self.n_obs = self.h_fn(np.random.randn(self.n_states, 1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e if mu_e is not None else np.zeros(self.n_states)
        self.mu_w = mu_w if mu_w is not None else np.zeros(self.n_obs)
        self.use_Taylor = use_Taylor

    def init_noise_covs(self) -> None:
        """初始化噪声协方差矩阵 / Initialise diagonal covariances."""

        self.Q = self.q * np.eye(self.n_states)
        self.R = self.r * np.eye(self.n_obs)

    def h_fn(self, x: np.ndarray) -> np.ndarray:
        """观测映射函数: y = a * (b * x + c)。"""

        return self.a * (self.b * x + self.c)

    def f_fn(self, x: np.ndarray) -> np.ndarray:
        """状态更新函数: x_next = alpha * sin(beta * x + phi) + delta。"""

        return self.alpha * np.sin(self.beta * x + self.phi) + self.delta

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float,
        nu_dB: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成正弦 SSM 轨迹 / Generate sinusoidal trajectory.

        Args:
            T (int): 序列长度。
            inverse_r2_dB (float): 观测噪声逆功率 (dB)。
            nu_dB (float): 过程与观测噪声比 (dB)。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - x_d: 状态序列 [T+1, m]
                - y_d: 观测序列 [T, n]

        Tensor Dimensions:
            - e: [T+1, m]
            - v: [T, n]

        Math Notes:
            - r2 = 1 / dB_to_lin(inverse_r2_dB)
            - q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            - x_{t+1} = f_fn(x_t) + e_t
            - y_t = h_fn(x_t) + v_t
        """

        x = np.zeros((T + 1, self.n_states))
        y = np.zeros((T, self.n_obs))

        # ==================================================================
        # STEP 01: 参数换算与噪声采样 (Noise statistics and sampling)
        # ==================================================================
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r = r2
        self.q = q2
        self.init_noise_covs()
        e = np.random.multivariate_normal(self.mu_e, q2 * np.eye(self.n_states), size=(T + 1,))
        v = np.random.multivariate_normal(self.mu_w, r2 * np.eye(self.n_obs), size=(T,))

        # ==================================================================
        # STEP 02: 前向模拟状态与观测 (Roll out dynamics)
        # ==================================================================
        for t in range(T):
            x[t + 1] = self.f_fn(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]

        if self.decimate:
            K = max(int(self.delta_d // self.delta), 1)
            x_d = x[0:T:K, :]
            y_d = self.h_fn(x_d) + np.random.multivariate_normal(self.mu_e, self.R, size=(len(x_d),))
        else:
            x_d = x
            y_d = y

        return x_d, y_d
