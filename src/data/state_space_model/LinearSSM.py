"""Linear SSM utilities for synthetic trajectory generation.

项目背景 (Project Background):
    该模块实现线性状态空间模型，用于生成训练 RNN/滤波器的数据集并支持 Kalman
    系列算法验证。

输入输出概览 (Input/Output Overview):
    - 构造函数接收系统矩阵、噪声统计等，生成仿真所需的结构化对象。
    - ``generate_single_sequence`` 基于噪声驱动返回状态矩阵 X ∈ ℝ^[T+1, m] 与观测
      矩阵 Y ∈ ℝ^[T, n]。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin, generate_normal


class LinearSSM:
    """线性状态空间模型 / Linear state-space model with optional control input.

    Args:
        n_states (int): 状态维度 m。
        n_obs (int): 观测维度 n。
        F (np.ndarray | None): 状态转移矩阵 [m, m]，缺省时内部构造。
        G (np.ndarray | None): 控制矩阵 [m, u]，可为 ``None``。
        H (np.ndarray | None): 观测矩阵 [n, m]，缺省时内部构造。
        mu_e (float | np.ndarray): 过程噪声均值 [m]。
        mu_w (float | np.ndarray): 观测噪声均值 [n]。
        q2 (float): 过程噪声方差标量，若 ``Q`` 未提供则使用。
        r2 (float): 观测噪声方差标量，若 ``R`` 未提供则使用。
        Q (np.ndarray | None): 过程噪声协方差 [m, m]。
        R (np.ndarray | None): 观测噪声协方差 [n, n]。

    Tensor Dimensions:
        - x_k: [m]
        - y_k: [n]
        - X: [T+1, m]
        - Y: [T, n]

    Math Notes:
        - x_next = F @ x + G @ u + e
        - y      = H @ x + w
    """

    def __init__(
        self,
        n_states: int,
        n_obs: int,
        F: Optional[np.ndarray] = None,
        G: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        mu_e=0.0,
        mu_w=0.0,
        q2: float = 1.0,
        r2: float = 1.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> None:
        self.n_states = n_states
        self.n_obs = n_obs

        if F is None and H is None:
            self.F = self.construct_F()
            self.H = self.construct_H()
        else:
            self.F = F
            self.H = H

        self.G = G
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.q2 = q2
        self.r2 = r2
        self.Q = Q
        self.R = R

        if self.Q is None and self.R is None:
            # ==================================================================
            # STEP 01: 初始化噪声协方差 (Noise covariance bootstrapping)
            # ------------------------------------------------------------------
            self.init_noise_covs()

    def construct_F(self) -> np.ndarray:
        """构造缺省状态转移矩阵 / Build default transition matrix.

        Returns:
            np.ndarray: F ∈ ℝ^[m, m]。

        Tensor Dimensions:
            - F: [m, m]

        Math Notes:
            - F = I_m + offset，用于提供弱耦合动态。
        """

        m = self.n_states
        # ==================================================================
        # STEP 01: 构造单位阵并拼接偏移 (Baseline + shift)
        # ------------------------------------------------------------------
        F_sys = np.eye(m) + np.concatenate(
            (
                np.zeros((m, 1)),
                np.concatenate((np.ones((1, m - 1)), np.zeros((m - 1, m - 1))), axis=0),
            ),
            axis=1,
        )
        return F_sys

    def construct_H(self) -> np.ndarray:
        """构造缺省观测矩阵 / Build default observation matrix."""

        H_sys = np.rot90(np.eye(self.n_states, self.n_states)) + np.concatenate(
            (
                np.concatenate(
                    (np.ones((1, self.n_states - 1)), np.zeros((self.n_states - 1, self.n_states - 1))),
                    axis=0,
                ),
                np.zeros((self.n_states, 1)),
            ),
            axis=1,
        )
        return H_sys[: self.n_obs, : self.n_states]

    def init_noise_covs(self) -> None:
        """初始化协方差矩阵 / Set Q, R as scaled identity."""

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)

    def generate_driving_noise(self, k: int, a: float = 1.2, add_noise: bool = False) -> np.ndarray:
        """生成驱动信号 / Optional control excitation.

        Args:
            k (int): 时间索引。
            a (float): 正弦频率参数。
            add_noise (bool): 是否在相位添加高斯噪声。

        Returns:
            np.ndarray: u_k ∈ ℝ^[1, 1]。

        Math Notes:
            - u_k = cos(a * (k + 1) + noise)
        """

        if not add_noise:
            u_k = np.cos(a * (k + 1))
        else:
            u_k = np.cos(a * (k + 1) + np.random.normal(loc=0.0, scale=math.pi, size=(1, 1)))
        return np.asarray(u_k).reshape(-1, 1)

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float = 0.0,
        nu_dB: float = 0.0,
        drive_noise: bool = False,
        add_noise_flag: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成一条长度 T 的状态与观测序列 / Simulate trajectory.

        Args:
            T (int): 序列长度。
            inverse_r2_dB (float): 观测噪声逆功率 (dB)。
            nu_dB (float): 过程与观测噪声比 (dB)。
            drive_noise (bool): 是否注入驱动输入。
            add_noise_flag (bool): 驱动相位是否加入噪声。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_arr: 状态轨迹 X ∈ ℝ^[T+1, m]
                - Y_arr: 观测轨迹 Y ∈ ℝ^[T, n]

        Tensor Dimensions:
            - e_k_arr: [T, m]
            - w_k_arr: [T, n]

        Math Notes:
            - r2 = 1 / dB_to_lin(inverse_r2_dB)
            - q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            - x_{k+1} = F @ x_k + G @ u_k + e_k
            - y_k = H @ x_k + w_k
        """

        # ==================================================================
        # STEP 01: 初始化存储张量与噪声统计 (Allocate buffers + noise stats)
        # ------------------------------------------------------------------
        x_arr = np.zeros((T + 1, self.n_states))
        y_arr = np.zeros((T, self.n_obs))

        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()

        # ==================================================================
        # STEP 02: 采样过程/观测噪声 (Draw noise samples)
        # ------------------------------------------------------------------
        e_k_arr = generate_normal(N=T, mean=self.mu_e, Sigma2=self.Q)
        w_k_arr = generate_normal(N=T, mean=self.mu_w, Sigma2=self.R)

        # ==================================================================
        # STEP 03: 前向迭代生成状态与观测 (Rollout dynamics)
        # ------------------------------------------------------------------
        for k in range(T):
            if drive_noise:
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=add_noise_flag)
            else:
                u_k = np.zeros((self.G.shape[1], 1)) if self.G is not None else np.zeros((1, 1))

            e_k = e_k_arr[k]
            w_k = w_k_arr[k]

            control_term = self.G @ u_k if self.G is not None else 0.0
            x_next = self.F @ x_arr[k].reshape((-1, 1)) + control_term + e_k.reshape((-1, 1))
            x_arr[k + 1] = x_next.reshape((-1,))
            y_arr[k] = self.H @ x_arr[k] + w_k

        return x_arr, y_arr
