"""
===============================================================================
File: ukf_aliter.py
Author: Anubhab Ghosh (Feb 2023) | Annotated by ChatGPT (2025)
Adopted from: https://github.com/KalmanNet/KalmanNet_TSP
-------------------------------------------------------------------------------
🎯 背景 (Background)
- 主题：非线性状态估计（Unscented Kalman Filter, UKF）在 Lorenz 等非线性系统上的实现。
- 核心：通过无迹变换（Unscented Transform, UT）传播均值与协方差，避免一阶线性化误差。

📥📤 输入输出概览 (I/O Overview)
- System functions:
  - f: 非线性状态转移函数，接口 fx(x, dt) -> x_next
  - h: 非线性观测函数，接口 hx(x) -> z_hat
- Data tensors:
  - X: 真实状态轨迹，形状 [N, T_x, n_states]
  - Y: 观测序列，形状 [N, T_y, n_obs]
  - 返回估计：
      traj_estimated: UKF 的状态后验估计，形状 [N, T_x, n_states]
      Pk_estimated: UKF 的后验协方差，形状 [N, T_x, n_states, n_states]
      MSE_UKF_linear_arr.mean(): 平均 MSE（线性尺度）
      mse_ukf_dB_avg: 平均 MSE（dB 尺度）

👥 目标读者 (Intended Audience)
- 既面向初学者（希望快速掌握 UKF 的数据维度与调用方式），也面向研究者/工程师（关注数学建模与实现细节）。
- 注释采用中英混排：中文解释工程动机；英文保持变量/函数名一致以便检索。
===============================================================================
"""

#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
# Adopted from: https://github.com/KalmanNet/KalmanNet_TSP
#####################################################
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
import torch
from torch import nn
from timeit import default_timer as timer
# from utils.utils import dB_to_lin, mse_loss
from scipy.linalg import expm
# from parameters import delta_t, J_test
import numpy as np
import math


def A_fn_exact(z, dt):
    """
    Compute exact one-step flow of the continuous-time Lorenz system using matrix exponential.
    使用矩阵指数对连续时间 Lorenz 系统进行一阶精确离散化，并作用于状态向量。

    Args:
        z (np.ndarray): 当前状态向量 current state, shape [3,], order [x, y, z].
        dt (float): 采样周期 sampling time step.

    Returns:
        np.ndarray: 下一时刻状态的线性近似结果（通过局部线性化矩阵指数得到）,
                    shape [3,].

    Tensor Dimensions:
        z ∈ R^[3], dt ∈ R, return ∈ R^[3]

    Math Notes:
        Continuous-time linearization around state z:
            A(z) =
                [[-10,   10,      0],
                 [ 28,   -1,   -z_x],
                 [  0,   z_x,  -8/3]]
        Then state propagation (local linear model):
            x_{k+1} ≈ expm(A(z_k) * dt) @ z_k
        该函数将 A(z) 的矩阵指数直接乘以当前状态，实现一阶精确离散化的近似。
    """
    # ==========================================================================
    # STEP 01: 计算状态相关的雅可比矩阵 A(z)
    # --------------------------------------------------------------------------
    # Lorenz 系统的线性化矩阵 A 依赖于当前 x 分量 (z[0]); 这里使用经典参数。
    # ==========================================================================
    return expm(np.array([
                    [-10, 10, 0],
                    [28, -1, -z[0]],
                    [0, z[0], -8.0/3]
                ]) * dt) @ z


'''
def f_lorenz_danse(x, dt):
    # NOTE: This alternative formulation uses a Taylor expansion to build F.
    # 说明：该段为替代性实现（保留为注释），通过泰勒展开近似状态转移矩阵 F。
    # 不建议在生产中直接启用，因存在硬编码依赖和潜在设备/梯度问题。

    x = torch.from_numpy(x).type(torch.FloatTensor)
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B)
    #A = torch.reshape(torch.matmul(B, x),(3,3)).T # For KalmanNet
    A += C
    #delta = delta_t # Hardcoded for now
    # Taylor Expansion for F
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()
'''


class UKF_Aliter(nn.Module):
    """This class implements an Unscented Kalman Filter (UKF) using FilterPy under a PyTorch wrapper.
    该类封装 FilterPy 的 UKF，在 PyTorch 设备上管理张量与批处理接口（不改变 FilterPy 内部实现）。

    Args:
        n_states (int): 状态维度 number of states n_x.
        n_obs (int): 观测维度 number of observations n_z.
        f (callable): 状态转移函数 fx(x, dt) -> x_next，非线性。
        h (callable): 观测函数 hx(x) -> z_hat，非线性。
        Q (np.ndarray | None): 过程噪声协方差 process noise cov, shape [n_states, n_states].
        R (np.ndarray | None): 观测噪声协方差 measurement noise cov, shape [n_obs, n_obs].
        kappa (float): UT 超参数，决定 sigma 点分布的离散度。
        alpha (float): UT 超参数，通常取 1e-3~0.5，小则 sigma 点更靠近均值。
        beta (float): UT 超参数，针对高斯分布常取 2。
        n_sigma (int | None): 未使用的占位参数（保留兼容）。
        delta_t (float): 采样周期 sampling time step 用于 FilterPy UKF。
        inverse_r2_dB (float | None): 若提供，与 nu_dB 共同决定 Q 和 R（以 dB 尺度给出）。
        nu_dB (float | None): 噪声功率差（dB），用于从 inverse_r2_dB 推出 q2。
        device (str): PyTorch 设备, e.g., "cpu" 或 "cuda".
        init_cond (torch.Tensor | None): 初始条件（当前未使用；保留接口）。

    Attributes:
        device (str): 当前设备。
        n_states (int), n_obs (int): 维度信息。
        f_k, h_k (callable): 非线性系统函数。
        Q_k, R_k (torch.FloatTensor): 过程/观测噪声协方差（驻留在 device）。
        sigma_points (MerweScaledSigmaPoints): 无迹变换 sigma 点生成器。
        ukf (filterpy.kalman.UnscentedKalmanFilter): FilterPy UKF 实例。

    Tensor Dimensions:
        Q_k ∈ R^[n_states, n_states], R_k ∈ R^[n_obs, n_obs]
        ukf.x ∈ R^[n_states], ukf.P ∈ R^[n_states, n_states]

    Math Notes:
        UT 通过选取 2n+1 个 sigma 点 {chi_i} 及权重 {W_i} 来传播非线性：
            x_pred   = sum_i W_i^m * f(chi_i, dt)
            P_pred   = sum_i W_i^c * (f(chi_i) - x_pred)(...)^T + Q
            z_pred   = sum_i W_i^m * h(chi_i)
            P_zz     = sum_i W_i^c * (h(chi_i) - z_pred)(...)^T + R
            P_xz     = sum_i W_i^c * (chi_i - x_pred)(h(chi_i) - z_pred)^T
            K        = P_xz @ inv(P_zz)
            x_post   = x_pred + K @ (z - z_pred)
            P_post   = P_pred - K @ P_zz @ K^T
    """
    def __init__(self, n_states, n_obs, f=None, h=None, Q=None, R=None, kappa=-1,
                alpha=0.1, beta=2, n_sigma=None, delta_t=1.0, inverse_r2_dB=None,
                nu_dB=None, device='cpu', init_cond=None):
        super(UKF_Aliter, self).__init__()

        # ==========================================================================
        # STEP 01: 设备与系统维度初始化 (Device & dimensions)
        # ==========================================================================
        self.device = device
        self.n_states = n_states
        self.n_obs = n_obs
        self.f_k = f
        self.h_k = h

        # ==========================================================================
        # STEP 02: 噪声协方差来源 (Noise covariance setup)
        # --------------------------------------------------------------------------
        # 若提供 inverse_r2_dB 与 nu_dB 且 Q/R 未显式给出，则使用 dB 规则构造。
        # 设计原因：便于仅通过信噪参数快速设定滤波器超参。
        # r2 = 1 / SNR_linear; q2 = 10^((nu_dB - inverse_r2_dB)/10)
        # ==========================================================================
        if (not inverse_r2_dB is None) and (not nu_dB is None) and (Q is None) and (R is None):
            r2 = 1.0 / dB_to_lin(inverse_r2_dB)
            q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            Q = q2 * np.eye(self.n_states)
            R = r2 * np.eye(self.n_obs)

        # 将 numpy 转为驻留在 device 的 torch 张量
        self.Q_k = self.push_to_device(Q)  # 过程噪声协方差 Q
        self.R_k = self.push_to_device(R)  # 观测噪声协方差 R

        # ==========================================================================
        # STEP 03: 无迹变换超参数与 sigma 点 (UT hyperparams & sigma points)
        # ==========================================================================
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.get_sigma_points()  # 设置 self.sigma_points

        # self.init_cond = init_cond  # 保留接口（当前未启用）

        # ==========================================================================
        # STEP 04: 构建 FilterPy 的 UKF 实例 (Instantiate FilterPy UKF)
        # ==========================================================================
        self.delta_t = delta_t
        self.ukf = UnscentedKalmanFilter(dim_x=self.n_states, dim_z=self.n_obs, dt=self.delta_t,
                                        fx=self.f_k, hx=self.h_k, points=self.sigma_points)
        # 将噪声与初值同步到 FilterPy UKF (FilterPy 期望 numpy)
        self.ukf.R = self.R_k.numpy()
        self.ukf.Q = self.Q_k.numpy()
        self.ukf.x = torch.ones((self.n_states,)).numpy()
        self.ukf.P = (torch.eye(self.n_states) * 1e-5).numpy()
        return None

    def initialize(self):
        """
        Reset internal UKF posterior mean and covariance.
        重置 UKF 的内部后验均值与协方差（每个样本序列开始时调用）。

        Args:
            None

        Returns:
            None

        Tensor Dimensions:
            ukf.x ∈ R^[n_states], ukf.P ∈ R^[n_states, n_states]

        Math Notes:
            x_post_0 = 1 (per component)
            P_post_0 = 1e-5 * I
            小方差初值意味着对初始状态有强先验约束；若先验不准，可能收敛较慢。
        """
        # ==========================================================================
        # STEP 01: 重置后验
        # ==========================================================================
        self.ukf.x = torch.ones((self.n_states,)).numpy()
        self.ukf.P = (torch.eye(self.n_states) * 1e-5).numpy()

    def push_to_device(self, x):
        """
        Convert a numpy array to torch.FloatTensor on the configured device.
        将 numpy 数组转换为位于指定 device 的 FloatTensor。

        Args:
            x (np.ndarray): 输入矩阵/向量 input array.

        Returns:
            torch.FloatTensor: 位于 self.device 的张量。

        Tensor Dimensions:
            保持与输入相同的形状；仅改变类型与设备。

        Math Notes:
            无数学变换，仅数据类型与内存位置转换。
        """
        # ==========================================================================
        # STEP 01: 类型与设备迁移
        # ==========================================================================
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def get_sigma_points(self):
        """
        Construct sigma point generator (MerweScaledSigmaPoints).
        构造 Merwe 缩放 sigma 点生成器。

        Args:
            None

        Returns:
            None

        Math Notes:
            MerweScaledSigmaPoints 通过 (alpha, beta, kappa) 控制点分布与权重。
            对高斯分布，beta=2 通常最优；alpha 控制分布半径；kappa 影响离散度。
        """
        # ==========================================================================
        # STEP 01: 初始化 sigma 点策略
        # ==========================================================================
        self.sigma_points = MerweScaledSigmaPoints(self.n_states, alpha=self.alpha, beta=self.beta, kappa=self.kappa)

    def run_mb_filter(self, X, Y, U=None):
        """
        Run UKF over a mini-batch of sequences and compute MSE.
        在小批量序列上运行 UKF，并返回轨迹估计及 MSE 统计。

        Args:
            X (torch.Tensor): 真实状态轨迹 ground-truth states,
                shape [N, T_x, n_states] 或 [T_x, n_states]（将自动升维为 N=1）。
            Y (torch.Tensor): 观测序列 measurements,
                shape [N, T_y, n_obs] 或 [T_y, n_obs]（将自动升维为 N=1）。
            U (torch.Tensor | None): 控制输入 control inputs（未使用，保留接口）。

        Returns:
            tuple:
                traj_estimated (torch.FloatTensor): 后验状态估计, shape [N, T_x, n_states].
                Pk_estimated (torch.FloatTensor): 后验协方差估计, shape [N, T_x, n_states, n_states].
                MSE_UKF_linear_arr.mean() (torch.FloatTensor): 平均 MSE（线性尺度标量）。
                mse_ukf_dB_avg (torch.FloatTensor): 平均 MSE（dB 尺度标量）。

        Tensor Dimensions:
            N: batch size; T_x: 状态序列长度; T_y: 观测序列长度
            n_states: 状态维度；n_obs: 观测维度

        Math Notes:
            处理流程：
                for each sequence i:
                    initialize posterior (x_post_0, P_post_0)
                    for k in 0..T_y-1:
                        predict via UT: (x_pred, P_pred)
                        update with y[k]: (x_post, P_post)
                    compute MSE_i = mse(X_i[1:], traj_i[1:])
            这里将第一步估计与观测对齐，从 k=0 开始，最后评估使用 1: 对齐索引。
        """
        # ==========================================================================
        # STEP 00: 维度解析与张量准备 (Shape parsing & allocation)
        # ==========================================================================
        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))

        # 估计轨迹与协方差缓存（驻留 device）
        traj_estimated = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)

        # 批内每个序列的 MSE（线性尺度）
        MSE_UKF_linear_arr = torch.zeros((N,)).type(torch.FloatTensor)
        # points = JulierSigmaPoints(n=SysModel.m)  # 备用 sigma 点方案（未启用）

        # ==========================================================================
        # STEP 01: 批处理主循环 (Main batch loop)
        # ==========================================================================
        start = timer()
        for i in range(0, N):
            # 每个样本序列重置滤波器初值
            self.initialize()
            # if self.init_cond is not None:
            #     self.ukf.x = torch.unsqueeze(self.init_cond[i, :], 1).numpy()

            # ----------------------------------------------------------------------
            # STEP 01a: 时间步迭代 (Time-step loop)
            # ----------------------------------------------------------------------
            for k in range(0, Ty):
                # 先验预测：UT 传播均值与协方差
                self.ukf.predict()
                # 观测更新：融合当前观测 Y[i, k, :]
                self.ukf.update(Y[i, k, :].numpy())

                # 记录后验均值与协方差（k+1 对齐到 X 的索引方案）
                traj_estimated[i, k+1, :] = torch.from_numpy(self.ukf.x)
                Pk_estimated[i, k+1, :, :] = torch.from_numpy(self.ukf.P)

            # 以对齐后的时序窗口评估 MSE
            # MSE_UKF_linear_arr[i] = mse_loss(traj_estimated[i], X[i]).item()
            MSE_UKF_linear_arr[i] = mse_loss(X[i, 1:, :], traj_estimated[i, 1:, :]).mean().item()
            # print("ukf, sample: {}, mse_loss: {}".format(i+1, MSE_UKF_linear_arr[i]))

        end = timer()
        t = end - start  # 运行时长，可用于日志
        # print("Inference Time:", t)

        # ==========================================================================
        # STEP 02: 统计与日志 (Statistics & logging)
        # ==========================================================================
        # MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
        # MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
        # MSE_UKF_linear_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
        # MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_linear_std.abs())

        mse_ukf_dB_avg = torch.mean(10 * torch.log10(MSE_UKF_linear_arr), dim=0)
        print("UKF - MSE LOSS:", mse_ukf_dB_avg, "[dB]")
        print("UKF - MSE STD:", torch.std(10 * torch.log10(MSE_UKF_linear_arr), dim=0), "[dB]")

        # ==========================================================================
        # STEP 03: 返回结果 (Return results)
        # ==========================================================================
        return traj_estimated, Pk_estimated, MSE_UKF_linear_arr.mean(), mse_ukf_dB_avg
