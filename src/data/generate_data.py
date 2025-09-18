"""
====================================================================
项目 / 算法背景 (Background)
--------------------------------------------------------------------
本文件用于自动批量生成多类状态空间模型（SSM, 如 Linear/Lorenz/Sinusoidal）的观测数据集，
支持 RNN/滤波算法等机器学习模型的训练。支持参数化仿真和文件保存，可命令行一键生成标准数据集。

输入输出概览 (Input/Output Overview)
--------------------------------------------------------------------
输入:
    - 命令行参数：n_states, n_obs, num_samples, sequence_length, inverse_r2_dB, nu_dB, dataset_type, output_path
    - 支持多种SSM结构及其参数
输出:
    - 生成并保存标准化的数据集文件（PKL），内容为状态-观测对 [T, D] × N_samples

张量维度/数据结构说明:
    - 每条数据 [X, Y]: [T, n_states], [T, n_obs]
    - 数据集内容: Z_XY["data"]: [N_samples, 2] (object数组, 每行为[X, Y])

典型应用场景:
    - 状态估计算法、深度序列模型、RNN训练及基准实验
====================================================================
"""

# ==========================================================================
# STEP 01: 导入依赖库
# ==========================================================================
import numpy as np
import scipy
import sys
import pickle
import torch
from torch import distributions
# from matplotlib import pyplot as plt
from scipy.linalg import expm
# from utils.utils import save_dataset
# from parameters import get_parameters
# from ssm_models import LinearSSM, LorenzAttractorModel, SinusoidalSSM
import argparse
from parse import parse
import os
from pathlib import Path
import math

'''
# ==========================================================================
# STEP 00: Dummy or Missing Imports from Original Code
# ==========================================================================
# 说明: 若原工程未提供 parameters 模块, 这里提供占位实现确保脚本可运行。
try:
    from parameters import get_parameters, A_fn, h_fn, J_gen, delta_t
except ImportError:
    print("Warning: 'parameters' module not found. Using dummy implementations.")

    def get_parameters(N=1000, T=200, n_states=3, n_obs=3, inverse_r2_dB=40, nu_dB=0, device='cpu'):
        """获取各模型参数占位实现。
        Args:
            N (int): number of samples.
            T (int): trajectory length.
            n_states (int): state dimension.
            n_obs (int): observation dimension.
            inverse_r2_dB (float): inverse measurement noise power in dB.
            nu_dB (float): process vs measurement noise ratio in dB.
            device (str): device name, not used in dummy.
        Returns:
            tuple(dict, dict): (ssm_params, est_params).
        Tensor Dimensions:
            - LinearSSM: F [m,m], H [n,m], Q [m,m], R [n,n].
        Math Notes:
            - r2_lin = 10^(inverse_r2_dB/10); r2 = 1/r2_lin
            - q2 = 10^((nu_dB - inverse_r2_dB)/10)
        """
        r2_lin = 10**(inverse_r2_dB/10)
        r2 = 1/r2_lin
        q2 = 10**((nu_dB - inverse_r2_dB)/10)
        # ============== 嵌套定义 Lorenz A_fn 工厂 ==================
        def make_lorenz_A_fn(sigma=10.0, rho=28.0, beta=8/3):
          def A_fn(state):
              s = np.asarray(state).reshape(-1)
              X, Y, Z = s[0], s[1], s[2]
              return np.array([
                  [-sigma,  sigma,   0.0],
                  [ rho - Z, -1.0,  -X  ],
                  [   Y,      X,   -beta]
              ], dtype=float)
          return A_fn
        def make_h_fn_identity(n_obs, n_states):
          H = np.eye(n_states)[:n_obs, :]
          def h_fn(x):
              x = np.asarray(x).reshape(-1)
              return H @ x
          return h_fn

        # 在 get_parameters 里：
        h_fn = make_h_fn_identity(n_obs, n_states)

        # 使用时：
        A_fn = make_lorenz_A_fn()

        ssm_params = {
            "LinearSSM": {
                "n_states": n_states, "n_obs": n_obs, "F": np.eye(n_states), "G": np.zeros((n_states, 1)),
                "H": np.eye(n_obs, n_states), "mu_e": np.zeros(n_states), "mu_w": np.zeros(n_obs),
                "q2": q2, "r2": r2, "Q": q2 * np.eye(n_states), "R": r2 * np.eye(n_obs)
            },
            "LorenzSSM": {
                "n_states": n_states, "n_obs": n_obs, "J": 5, "delta": 0.02, "A_fn": A_fn, "h_fn": h_fn,
                "delta_d": 0.02, "decimate": False, "mu_e": np.zeros(n_states), "mu_w": np.zeros(n_obs),
                "inverse_r2_dB": inverse_r2_dB, "nu_dB": nu_dB, "use_Taylor": True
            },
            "SinusoidalSSM": {
                "n_states": n_states, "alpha": 0.9, "beta": 1.1, "phi": 0.1*math.pi, "delta": 0.01,
                "a": 1.0, "b": 1.0, "c": 0.0, "decimate": False, "mu_e": np.zeros(n_states), "mu_w": np.zeros(n_obs),
                "inverse_r2_dB": inverse_r2_dB, "nu_dB": nu_dB, "use_Taylor": False
            }
        }
        est_params = {
            "danse": {"batch_size": 64, "rnn_type": "gru", "rnn_params_dict": {"gru": {"lr": 1e-3, "num_epochs": 2000, "n_hidden": 40, "n_layers": 2, "min_delta": 1e-2, "n_hidden_dense": 32}}},
            "KF": {}, "EKF": {}, "UKF": {}, "KNetUoffline": {}
        }
        return ssm_params, est_params


    def h_fn(z):
        """恒等观测占位实现。
        Args:
            z (array-like): state vector.
        Returns:
            np.ndarray: observation vector, same shape as input.
        """
        return z

    J_gen = 5

'''
# =====================================================
# 类实现： LinearSSM， LorenzAttractorModel， SinusoidalSSM
# =====================================================
class LinearSSM(object):
    """
    线性状态空间模型 (Linear State-Space Model).
    Attributes:
        n_states (int): state dimension m.
        n_obs (int): observation dimension n.
        F (np.ndarray): state transition [m,m].
        H (np.ndarray): observation matrix [n,m].
        G (np.ndarray or None): control matrix [m,u].
        Q (np.ndarray): process noise covariance [m,m].
        R (np.ndarray): measurement noise covariance [n,n].
        mu_e (np.ndarray or float): process noise mean [m].
        mu_w (np.ndarray or float): measurement noise mean [n].
        q2 (float): process noise variance scalar if Q not provided.
        r2 (float): measurement noise variance scalar if R not provided.
    Tensor Dimensions:
        - x_k: [m], y_k: [n]
        - X: [T+1, m], Y: [T, n]
    Math Notes:
        - State: x_{k+1} = F @ x_k + G @ u_k + e_k
        - Obs:   y_k     = H @ x_k + w_k
        - e_k ~ N(mu_e, Q), w_k ~ N(mu_w, R)
    """

    def __init__(self, n_states, n_obs, F=None, G=None, H=None, mu_e=0.0, mu_w=0.0, q2=1.0, r2=1.0, Q=None, R=None) -> None:
        """初始化 LinearSSM。
        Args:
            n_states (int): state dimension m.
            n_obs (int): observation dimension n.
            F (np.ndarray or None): state transition [m,m]. if None, constructed.
            G (np.ndarray or None): control matrix [m,u].
            H (np.ndarray or None): observation matrix [n,m]. if None, constructed.
            mu_e (float or np.ndarray): process noise mean [m] or scalar.
            mu_w (float or np.ndarray): measurement noise mean [n] or scalar.
            q2 (float): scalar variance for Q if Q is None.
            r2 (float): scalar variance for R if R is None.
            Q (np.ndarray or None): covariance [m,m].
            R (np.ndarray or None): covariance [n,n].
        Returns:
            None
        """
        self.n_states = n_states
        self.n_obs = n_obs

        # ==========================================================================
        # STEP 01: 初始化系统矩阵
        # ==========================================================================
        if F is None and H is None:
            self.F = self.construct_F()
            self.H = self.construct_H()
        else:
            self.F = F
            self.H = H

        self.G = G

        # ==========================================================================
        # STEP 02: 初始化噪声参数
        # ==========================================================================
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.q2 = q2
        self.r2 = r2
        self.Q = Q
        self.R = R

        if self.Q is None and self.R is None:
            self.init_noise_covs()

    def construct_F(self):
        """
        构造状态转移矩阵 F (示例结构)。
        Returns:
            np.ndarray: F [m,m].
        Math Notes:
            - 示例为 I 加特定偏移拼接，纯演示用途。
        """
        m = self.n_states
        F_sys = np.eye(m) + np.concatenate(
            (np.zeros((m,1)),
             np.concatenate(
                 (np.ones((1,m-1)), np.zeros((m-1,m-1))), axis=0)
            ),
            axis=1)
        return F_sys

    def construct_H(self):
        """
        构造观测矩阵 H (示例结构)。
        Returns:
            np.ndarray: H [n,m] 截取得到。
        """
        H_sys = np.rot90(np.eye(self.n_states, self.n_states)) + \
                np.concatenate(
                    (np.concatenate(
                        (np.ones((1, self.n_states-1)), np.zeros((self.n_states-1, self.n_states-1))), axis=0),
                     np.zeros((self.n_states,1))),
                    axis=1)
        return H_sys[:self.n_obs, :self.n_states]

    def init_noise_covs(self):
        """
        初始化过程噪声 Q 与观测噪声 R。
        Returns:
            None
        Tensor Dimensions:
            - Q [m,m], R [n,n].
        """
        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)
        return None

    def generate_driving_noise(k, a=1.2, add_noise=False):
        """
        生成驱动信号 u_k (可带扰动)。
        Args:
            k (int): time index.
            a (float): frequency scale.
            add_noise (bool): whether to add phase noise.
        Returns:
            np.ndarray or float: u_k as scalar array shape [1,1] or float.
        Math Notes:
            - u_k = cos(a*(k+1) + noise)
        """
        if add_noise == False:
            u_k = np.cos(a*(k+1))
        elif add_noise == True:
            u_k = np.cos(a*(k+1) + np.random.normal(loc=0, scale=np.pi, size=(1,1)))
        return u_k

    def generate_single_sequence(self, T, inverse_r2_dB=0, nu_dB=0, drive_noise=False, add_noise_flag=False):
        """
        生成一条长度 T 的状态/观测序列。
        Args:
            T (int): trajectory length.
            inverse_r2_dB (float): inverse measurement noise power in dB.
            nu_dB (float): process vs measurement noise in dB.
            drive_noise (bool): whether to use driving signal u_k.
            add_noise_flag (bool): whether to add phase noise to u_k.
        Returns:
            tuple(np.ndarray, np.ndarray): (x_arr [T+1,m], y_arr [T,n]).
        Tensor Dimensions:
            - x_arr [T+1, m], y_arr [T, n].
        Math Notes:
            - r2 = 1 / 10^(inverse_r2_dB/10)
            - q2 = 10^((nu_dB - inverse_r2_dB)/10)
            - x_{k+1} = F x_k + G u_k + e_k
            - y_k = H x_k + w_k
        """
        x_arr = np.zeros((T+1, self.n_states))
        y_arr = np.zeros((T, self.n_obs))

        # ==========================================================================
        # STEP 01: dB 参数换算为方差并初始化协方差
        # ==========================================================================
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()

        # ==========================================================================
        # STEP 02: 采样过程噪声与观测噪声
        # ==========================================================================
        e_k_arr = generate_normal(N=T, mean=self.mu_e, Sigma2=self.Q)
        w_k_arr = generate_normal(N=T, mean=self.mu_w, Sigma2=self.R)

        # ==========================================================================
        # STEP 03: 前向仿真生成 X, Y
        # ==========================================================================
        for k in range(T):
            if drive_noise == True:
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=add_noise_flag)
            else:
                u_k = np.array([0.0]).reshape((-1,1))

            e_k = e_k_arr[k]
            w_k = w_k_arr[k]

            # 状态更新与观测生成
            x_arr[k+1] = (self.F @ x_arr[k].reshape((-1,1)) + self.G @ u_k + e_k.reshape((-1,1))).reshape((-1,))
            y_arr[k] = self.H @ (x_arr[k]) + w_k

        return x_arr, y_arr

# =====================================================
class LorenzAttractorModel(object):
    """
    Lorenz 吸引子状态空间模型 (强非线性/混沌)。
    支持基于雅可比的矩阵指数泰勒截断近似进行一步预测。
    Attributes:
        n_states (int): state dimension d.
        J (int): Taylor truncation order.
        delta (float): time step.
        delta_d (float): decimation step if enabled.
        A_fn (callable): returns A(x) Jacobian-like matrix.
        h_fn (callable): observation mapping.
        n_obs (int): observation dimension inferred from h_fn.
        decimate (bool): whether to decimate outputs.
        mu_e, mu_w: noise means.
        use_Taylor (bool): keep-on switch.
    Tensor Dimensions:
        - x_t [d], y_t [n]; X [T+1,d], Y [T,n].
    Math Notes:
        - F_approx = I + sum_{j=1..J} ((A(x0)*delta)^j / j!)
        - x_{t+1} ≈ F_approx @ x_t + e_t
        - y_t = h(x_t) + v_t
    """

    def __init__(self, d, J, delta, delta_d, A_fn, h_fn, decimate=False, mu_e=None, mu_w=None, use_Taylor=True) -> None:
        """初始化 LorenzAttractorModel。
        Args:
            d (int): state dimension.
            J (int): Taylor order.
            delta (float): time step.
            delta_d (float): decimation step.
            A_fn (callable): A_fn(x)->np.ndarray [d,d].
            h_fn (callable): h_fn(x)->np.ndarray [n] or [n,1].
            decimate (bool): decimation flag.
            mu_e (np.ndarray or None): process noise mean [d].
            mu_w (np.ndarray or None): measurement noise mean [n].
            use_Taylor (bool): use Taylor switch.
        Returns:
            None
        """
        self.n_states = d
        self.J = J
        self.delta = delta
        self.delta_d = delta_d

        # self.A_fn = A_fn
        # self.h_fn = h_fn
        # 原代码： self.h_fn = h_fn
        self.A_fn = A_fn if A_fn is not None else (lambda x: np.eye(d))
        self.h_fn = h_fn if h_fn is not None else (lambda x: x)

        self.n_obs = self.h_fn(np.random.randn(d,1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def h_fn(self, x):
        """默认观测映射：恒等。
        Args:
            x (np.ndarray): state.
        Returns:
            np.ndarray: same as input.
        """
        return x

    def f_linearize(self, x):
        """
        线性化一步预测 (Taylor 截断近似矩阵指数)。
        Args:
            x (np.ndarray): current state [d] or [d,1].
        Returns:
            np.ndarray: next state approximation F @ x, shape like x.
        Math Notes:
            - F = I + sum_{j=1..J} ((A(x0)*delta)^j / j!)
        """
        # ==========================================================================
        # STEP 01: 初始化 F = I
        # ==========================================================================
        self.F = np.eye(self.n_states)
        # ==========================================================================
        # STEP 02: 累加泰勒项
        # ==========================================================================
        for j in range(1, self.J+1):
            self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / math.factorial(j)
        # ==========================================================================
        # STEP 03: 返回预测
        # ==========================================================================
        return self.F @ x

    def init_noise_covs(self):
        """
        初始化 Q/R 为对角阵。
        Returns:
            None
        """
        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)
        return None

    def generate_single_sequence(self, T, inverse_r2_dB, nu_dB):
        """
        生成一条 Lorenz 近似序列。
        Args:
            T (int): length.
            inverse_r2_dB (float): inverse measurement noise in dB.
            nu_dB (float): process vs measurement noise in dB.
        Returns:
            tuple(np.ndarray, np.ndarray): (x_lorenz_d, y_lorenz_d).
        Tensor Dimensions:
            - x [T+1,d], y [T,n].
        Math Notes:
            - r2 = 1 / 10^(inverse_r2_dB/10)
            - q2 = 10^((nu_dB - inverse_r2_dB)/10)
            - x_{t+1} ≈ F(x_t) x_t + e_t
            - y_t = h(x_t) + v_t
        """
        x = np.zeros((T+1, self.n_states))
        y = np.zeros((T, self.n_obs))
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()
        # 采样噪声
        e = np.random.multivariate_normal(self.mu_e, self.Q, size=(T+1,))
        v = np.random.multivariate_normal(self.mu_w, self.R, size=(T,))
        # 前向生成
        for t in range(0,T):
            x[t+1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        # 可选下采样: 重新生成对应时刻的观测噪声
        if self.decimate == True:
            K = int(round(self.delta_d / self.delta))
            K = max(K, 1)

            x_lorenz_d = x[0:T:K,:]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(self.mu_w, self.R, size=(len(x_lorenz_d),))
        else:
            x_lorenz_d = x
            y_lorenz_d = y
        return x_lorenz_d, y_lorenz_d

# =====================================================
class SinusoidalSSM(object):
    """
    正弦型非线性 SSM。
    Attributes:
        n_states (int): state dimension.
        alpha, beta, phi, delta (float): dynamics params.
        a, b, c (float): observation affine-sine params.
        n_obs (int): inferred via h_fn.
        decimate (bool): decimation flag.
        mu_e, mu_w (np.ndarray or None): noise means.
    Tensor Dimensions:
        - X [T+1,m], Y [T,n].
    Math Notes:
        - f(x) = alpha * sin(beta * x + phi) + delta
        - h(x) = a * (b * x + c)
    """

    def __init__(self, n_states, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01, a=1.0, b=1.0, c=0.0, decimate=False, mu_e=None, mu_w=None, use_Taylor=False):
        """初始化 SinusoidalSSM。"""
        self.n_states = n_states
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.a = a
        self.b = b
        self.c = c
        self.n_obs = self.h_fn(np.random.randn(self.n_states,1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def init_noise_covs(self):
        """初始化对角 Q/R。"""
        self.Q = self.q * np.eye(self.n_states)
        self.R = self.r * np.eye(self.n_obs)
        return None

    def h_fn(self, x):
        """观测映射: y = a * (b * x + c)。"""
        return self.a * (self.b * x + self.c)

    def f_fn(self, x):
        """状态更新: x_next = alpha * sin(beta * x + phi) + delta。"""
        return self.alpha * np.sin(self.beta * x + self.phi) + self.delta

    def generate_single_sequence(self, T, inverse_r2_dB, nu_dB):
        """
        生成一条正弦 SSM 序列。
        Args:
            T (int): length.
            inverse_r2_dB (float): inverse measurement noise in dB.
            nu_dB (float): process vs measurement noise in dB.
        Returns:
            tuple(np.ndarray, np.ndarray): (x_d, y_d).
        Tensor Dimensions:
            - x [T+1,m], y [T,n].
        """
        x = np.zeros((T+1, self.n_states))
        y = np.zeros((T, self.n_obs))
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r = r2
        self.q = q2
        self.init_noise_covs()
        e = np.random.multivariate_normal(np.zeros(self.n_states,), q2*np.eye(self.n_states),size=(T+1,))
        v = np.random.multivariate_normal(np.zeros(self.n_obs,), r2*np.eye(self.n_obs),size=(T,))
        for t in range(0,T):
            x[t+1] = self.f_fn(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_d = x[0:T:K,:]
            y_d = self.h_fn(x_d) + np.random.multivariate_normal(self.mu_e, self.R, size=(len(x_d),))
        else:
            x_d = x
            y_d = y
        return x_d, y_d


def initialize_model(type_, parameters):
    """
    根据类型及参数初始化 SSM 实例 (Factory)。
    Args:
        type_ (str): "LinearSSM" / "LorenzSSM" / "SinusoidalSSM".
        parameters (dict): required params per model type.
    Returns:
        object: model instance.
    Tensor Dimensions:
        - Varies by model; commonly uses n_states, n_obs, and matrices.
    """
    if type_ == "LinearSSM":
        model = LinearSSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            F=parameters["F"],
            G=parameters["G"],
            H=parameters["H"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"],
            q2=parameters["q2"],
            r2=parameters["r2"],
            Q=parameters["Q"],
            R=parameters["R"])
    elif type_ == "LorenzSSM":
        model = LorenzAttractorModel(
            d=parameters["n_states"],
            J=parameters["J"],
            delta=parameters["delta"],
            A_fn=parameters["A_fn"],
            h_fn=parameters["h_fn"],
            delta_d=parameters["delta_d"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"]
        )
    elif type_ == "SinusoidalSSM":
        model = SinusoidalSSM(
            n_states=parameters["n_states"],
            alpha=parameters["alpha"],
            beta=parameters["beta"],
            phi=parameters["phi"],
            delta=parameters["delta"],
            a=parameters["a"],
            b=parameters["b"],
            c=parameters["c"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"]
        )
    return model

def generate_SSM_data(model, T, parameters):
    """
    生成单条 SSM 序列 (X, Y)。
    Args:
        model (object): SSM instance.
        T (int): sequence length.
        parameters (dict): simulation hyper-parameters, including inverse_r2_dB, nu_dB.
    Returns:
        tuple(np.ndarray, np.ndarray): X_arr [T+1,m], Y_arr [T,n].
    Tensor Dimensions:
        - LinearSSM: returns from model.generate_single_sequence with driving off.
        - Others: model.generate_single_sequence(T, inverse_r2_dB, nu_dB).
    Math Notes:
        - Delegates to model.generate_single_sequence.
    """
    if type(model).__name__ == "LinearSSM":
        X_arr = np.zeros((T+1, model.n_states))
        Y_arr = np.zeros((T, model.n_obs))
        X_arr, Y_arr = model.generate_single_sequence(
            T=T,
            inverse_r2_dB=parameters["inverse_r2_dB"],
            nu_dB=parameters["nu_dB"],
            drive_noise=False,
            add_noise_flag=False
        )
    else:
        X_arr = np.zeros((T+1, model.n_states))
        Y_arr = np.zeros((T, model.n_obs))
        X_arr, Y_arr = model.generate_single_sequence(
            T=T,
            inverse_r2_dB=parameters["inverse_r2_dB"],
            nu_dB=parameters["nu_dB"]
        )
    return X_arr, Y_arr

#+++++++++++++++++修改过++++++++++++++++++++
def generate_state_observation_pairs(type_, parameters, T=200, N_samples=1000):
    """
    批量生成 N_samples 条 (X, Y) 状态-观测对。
    Args:
        type_ (str): SSM type.
        parameters (dict): SSM parameters for the given type.
        T (int): sequence length per sample.
        N_samples (int): number of sequences.
    Returns:
        dict: Z_XY dataset dict with keys:
            - 'ssm_model': the model instance used. (optional)
            - 'num_samples': N_samples.
            - 'data': np.array of shape [N, 2] with dtype=object, each row [Xi, Yi].
            - 'trajectory_lengths': np.array [N], typically all T.
    Tensor Dimensions:
        - Xi [T+1,m], Yi [T,n] (depending on model conventions).
    """
    Z_XY = {}
    Z_XY["num_samples"] = N_samples
    Z_XY_data_lengths = []
    Z_XY_data = []
    #++++++++++++++++++++在这里： 可以修改A_fn参数+++++++++++++++++++++++++++++
    ssm_model = initialize_model(type_, parameters)

    '''
    def my_A_fn(z):
      z = np.asarray(z).reshape(-1)
      X, Y, Z = z
      sigma, rho, beta = 12.0, 30.0, 2.5
      return np.array([[-sigma, sigma, 0.0],
                     [rho - Z, -1.0, -X],
                     [0.0,     X,   -beta]], dtype=float)

    model = initialize_model("LorenzSSM", ssm_parameters["LorenzSSM"])
    model.A_fn = my_A_fn
    '''
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Z_XY['ssm_model'] = ssm_model

    # ==========================================================================
    # STEP 01: 逐条生成样本
    # ==========================================================================
    for _ in range(N_samples):
        Xi, Yi = generate_SSM_data(ssm_model, T, parameters)
        Z_XY_data_lengths.append(T)
        Z_XY_data.append([Xi, Yi])

    # ==========================================================================
    # STEP 02: 打包为 object 数组, 兼容不同形状
    # ==========================================================================
    Z_XY["data"] = np.array(Z_XY_data, dtype=object)  # 每行 [Xi, Yi]
    Z_XY["trajectory_lengths"] = np.array(Z_XY_data_lengths)
    return Z_XY

def create_filename(T, N_samples, m, n, type_, inverse_r2_dB, nu_dB, dataset_basepath = './data/'):
    """
    基于核心超参数创建数据文件名 (便于溯源与区分)。
    Args:
        T (int): length per sequence.
        N_samples (int): number of sequences.
        m (int): state dimension.
        n (int): observation dimension.
        type_ (str): SSM type string.
        inverse_r2_dB (float): inverse measurement noise in dB.
        nu_dB (float): process vs measurement noise in dB.
        dataset_basepath (str): folder base path.
    Returns:
        str: dataset_fullpath, like:
             ./data/trajectories_m_3_n_3_LorenzSSM_data_T_200_N_1000_r2_40.0dB_nu_0.0dB.pkl
    """
    datafile = "trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_r2_{}dB_nu_{}dB.pkl".format(
        m, n, type_, int(T), int(N_samples), inverse_r2_dB, nu_dB)
    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def create_and_save_dataset(T, N_samples, filename, parameters, type_="LinearSSM"):
    """
    一键生成并保存 SSM 数据集到磁盘 (pickle)。
    Args:
        T (int): sequence length.
        N_samples (int): number of samples.
        filename (str): full output path to save .pkl.
        parameters (dict): SSM parameters for selected type.
        type_ (str): SSM type.
    Returns:
        None
    Notes:
        - 支持通过外部固定随机种子实现复现。
    """
    # ==========================================================================
    # STEP 01: 生成数据集
    # ==========================================================================
    Z_XY = generate_state_observation_pairs(type_=type_, parameters=parameters, T=T, N_samples=N_samples)
    # ==========================================================================
    # STEP 02: 写入磁盘 (with 语法确保异常时也会关闭文件句柄)
    # ==========================================================================
    with open(filename, 'wb') as f:
        pickle.dump(Z_XY, f)

# if __name__ == "__main__":
#     # ==========================================================================
#     # STEP 02: 命令行参数解析与准备
#     # ==========================================================================
#     usage = (
#         "Create datasets by simulating state space models \n"
#         "python generate_data.py --sequence_length T --num_samples N --dataset_type [LinearSSM/LorenzSSM] --output_path [output path name]\n"
#         "Creates the dataset at the location output_path"
#     )

#     if 'ipykernel' in sys.modules:
#         # 在 notebook 环境, 手动设定参数
#         # './eval_sets/Lorenz_Atractor/T1000_NT100/' + '/' dataFileName
#         class Args:
#             n_states = 3
#             n_obs = 3
#             num_samples = 100
#             sequence_length = 1000
#             '''
#             # | (1/r^2) [dB] | r^2 [dB] | r^2 (线性) | ν [dB] |   ν (线性)   |
#             # |-------------:|---------:|-----------:|-------:|-------------:|
#             # |          -20 |      +20 |       100  |   -30  |   0.001      |
#             # |          -10 |      +10 |        10  |   -20  |   0.01       |
#             # |           -5 |       +5 |     3.1623 |   -15  |   0.0316     |
#             # |            0 |        0 |         1  |   -10  |   0.1        |
#             # |            5 |       -5 |     0.3162 |    -5  |   0.3162     |
#             # |           10 |      -10 |       0.1  |     0  |   1          |
#             # |           20 |      -20 |      0.01  |    10  |   10         |
#             '''
#             inverse_r2_dB = -40
#             nu_dB = -50
#             dataset_type = 'LorenzSSM'
#             # output_path = './data'
#             output_path = './eval_sets/Lorenz_Atractor/T1000_NT100/' + '/'
#         args = Args()
#     else:
#         parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for training RNNs")
#         parser.add_argument("--n_states", help="denotes the number of states in the latent model", type=int, default=5)
#         parser.add_argument("--n_obs", help="denotes the number of observations", type=int, default=5)
#         parser.add_argument("--num_samples", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
#         parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
#         parser.add_argument("--inverse_r2_dB", help="denotes the inverse of measurement noise power", type=float, default=40.0)
#         parser.add_argument("--nu_dB", help="denotes the ration between process and measurement noise", type=float, default=0.0)
#         parser.add_argument("--dataset_type", help="specify mode=pfixed (all theta, except theta_3, theta_4) / vars (variances) / all (full theta vector)", type=str, default=None)
#         parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
#         args = parser.parse_args()

#     # ==========================================================================
#     # STEP 03: 参数读取与数据集路径生成
#     # ==========================================================================
#     n_states = args.n_states
#     n_obs = args.n_obs
#     T = args.sequence_length
#     N_samples = args.num_samples
#     type_ = args.dataset_type
#     output_path = args.output_path
#     inverse_r2_dB = args.inverse_r2_dB
#     nu_dB = args.nu_dB

#     # ✅ 自动创建输出目录（Colab友好）
#     if output_path is not None and not os.path.exists(output_path):
#         print(f"Creating output directory at {output_path} ...")
#         os.makedirs(output_path)

#     datafilename = create_filename(
#         T=T, N_samples=N_samples, m=n_states, n=n_obs,
#         type_=type_, inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB, dataset_basepath=output_path)  # Corrected order

#     # ✅ 获取模型参数 (若为占位实现, 仅用于示例)
#     ssm_parameters, _ = get_parameters(
#         N=N_samples, T=T, n_states=n_states, n_obs=n_obs,
#         inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB)

#     # ==========================================================================
#     # STEP 04: 数据集生成与保存（避免重复生成）
#     # ==========================================================================
#     if not os.path.isfile(datafilename):
#         print("Creating the data file: {}".format(datafilename))
#         create_and_save_dataset(
#             T=T, N_samples=N_samples, filename=datafilename,
#             type_=type_, parameters=ssm_parameters[type_])
#     else:
#         print("Dataset {} is already present!".format(datafilename))
#     print("Done...")
