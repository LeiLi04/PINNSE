"""
====================================================================
项目 / 算法背景 (Background)
--------------------------------------------------------------------
本文件定义状态空间模型（SSM）与估计算法（如 DANSE, KalmanNet, KF/UKF 等）所需的参数生成器、动力学函数等。
支持 LinearSSM、Lorenz 吸引子、Sinusoidal SSM 等多种模型类型。所有参数均支持程序化配置，便于科研复现与超参数搜索。

输入输出概览 (Input/Output Overview)
--------------------------------------------------------------------
输入:
    - 基础模型维数/超参数（如 n_states, n_obs, T, N, 噪声功率等）
    - 控制参数（模型类型、Taylor 展开阶数、增益等）
输出:
    - ssm_parameters_dict: 各 SSM 的参数字典
    - estimators_dict: 各滤波/估计算法的配置字典

张量维度说明:
    - 所有动力学函数输入/输出均为 shape [n_states] 或 [n_obs]
    - H, F, Q, R, G 等为 [n, n] / [n, 1] 等型矩阵
====================================================================
"""

import numpy as np
import math
import torch
# from utils.utils import dB_to_lin, partial_corrupt
# from ssm_models import LinearSSM
from torch.autograd.functional import jacobian

# ==========================================================================
# STEP 01: 全局常数设定（时间步长、泰勒展开阶数等）
# ==========================================================================
torch.manual_seed(10)
delta_t = 0.02            # 训练/生成数据用步长
delta_t_test = 0.04       # 测试/验证用步长
J_gen = 5                 # 生成/动力学泰勒展开项数
J_test = 5                # 测试时泰勒项

# ==========================================================================
# STEP 02: Lorenz 系统动力学与观测函数
# ==========================================================================
def A_fn(z):
    z = np.asarray(z).reshape(-1)
    X, Y, Z = z[0], z[1], z[2]
    sigma = 10.0
    rho   = 28.0
    beta  = 8.0/3.0
    return np.array([
        [-sigma,  sigma,  0.0],
        [rho - Z,  -1.0, -X  ],
        [0.0,       X,   -beta]
    ], dtype=float)


def h_fn(z):
    z = np.asarray(z).reshape(-1)
    return z


# ==========================================================================
# STEP 03: Lorenz SSM 动力学（Taylor 近似），多种场景下使用
# ==========================================================================
def f_lorenz_danse_test_ukf(x, dt):
    """
    用于 UKF 测试的 Lorenz 状态转移函数（采用 delta_t_test）。
    Args:
        x (np.ndarray): 当前状态 [3]
        dt (float): 步长
    Returns:
        x_next (np.ndarray): 下一状态 [3]
    Math Notes:
    --------------------------------------
    # 连续时间模型（标准 Lorenz，σ=10, ρ=28, β=8/3）
    经典形式：
        \dot{x} = σ (y - x)
        \dot{y} = x (ρ - z) - y
        \dot{z} = x y - β z

    线性 + 双线性分解：
        设 C = [[-σ,  σ,  0],
                [ ρ, -1,  0],
                [  0,  0, -β]]

        设 (B_1, B_2, B_3) 为 3×3 矩阵，满足：
            B_1 = [[0, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]],   B_2 = 0,  B_3 = 0

        则连续系统可写为
            f(x) = C x + \sum_{n=1}^3 x_n (B_n x)
                 = A(x) x,
        其中
            A(x) := C + \sum_{n=1}^3 x_n B_n
                  = C + x_1 B_1.                       (因为 B_2=B_3=0)

        直观验证：
            B_1 x = [ 0, -z,  y ]^T
            x_1 (B_1 x) = [ 0, -x z, x y ]^T
        恰对应 Lorenz 的非线性项：\dot{y} 含 -x z，\dot{z} 含 +x y。

    --------------------------------------
        泰勒展开（截断到 J 阶）：
            F_k ≈ I + (A_k Δt) + (A_k Δt)^2/2! + ... + (A_k Δt)^J/J!
    """
    x = torch.from_numpy(x).type(torch.FloatTensor)
    # B（3，3，3）， 用来存储 Lorenz 系统里与状态变量 x = [x1, x2, x3] 相关的线性矩阵。处理耦合项
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]],
                      torch.zeros(3,3),
                      torch.zeros(3,3)]).type(torch.FloatTensor)
    # C表示线性部分, 也就是非耦合部分
    C = torch.Tensor([[-10, 10,    0],
                      [28, -1,    0],
                      [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B)
    A += C
    F = torch.eye(3)
    J = J_test
    for j in range(1, J+1):
        F_add = (torch.matrix_power(A*dt, j)/math.factorial(j)) #delta_t_test -> dt
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()

def fx_lorenz_taylor(x, dt, J=5, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    与你思路一致：先构造 A(x)=C + x1*B1，然后 F ≈ I + Σ_{j=1..J} (A dt)^j / j!
    最后返回 F @ x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    X = x[0]

    C  = np.array([[-sigma,  sigma,   0.0],
                   [  rho,   -1.0,   0.0],
                   [   0.0,   0.0, -beta]], dtype=float)
    B1 = np.array([[ 0.0,  0.0,  0.0],
                   [ 0.0,  0.0, -1.0],
                   [ 0.0,  1.0,  0.0]], dtype=float)

    A = C + X * B1
    M = A * dt

    F = np.eye(3)
    P = np.eye(3)  # 当前累积的幂/阶乘项
    for j in range(1, J+1):
        P = P @ M / j        # 逐阶：P = M^j / j!
        F = F + P
    return F @ x

def f_lorenz_danse_ukf(x, dt):
    """
    用于 UKF 训练/生成的 Lorenz 状态转移（采用 delta_t）。
    见 f_lorenz_danse_test_ukf，仅步长不同。
    """
    x = torch.from_numpy(x).type(torch.FloatTensor)
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                      [28, -1,    0],
                      [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B)
    A += C
    F = torch.eye(3)
    J = J_test
    for j in range(1, J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()

def f_lorenz_danse_test(x):
    """
    Lorenz 状态转移（用于测试/可视化）。
    """
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                      [28, -1,    0],
                      [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    F = torch.eye(3)
    J = J_test
    for j in range(1, J+1):
        F_add = (torch.matrix_power(A*delta_t_test, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def f_lorenz_danse(x):
    """
    Lorenz 状态转移（训练/生成用，delta_t）。
    """
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                      [28, -1,    0],
                      [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    F = torch.eye(3)
    J = J_test
    for j in range(1, J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

# ==========================================================================
# STEP 04: Sinusoidal SSM 模型函数
# ==========================================================================
def f_sinssm_fn(z, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01):
    """
    Sinusoidal SSM 的状态转移函数
    Math: z_next = alpha * sin(beta*z + phi) + delta
    """
    return alpha * torch.sin(beta * z + phi) + delta

def h_sinssm_fn(z, a=1, b=1, c=0):
    """
    Sinusoidal SSM 的观测函数
    Math: y = a * (b*z + c)
    """
    return a * (b * z + c)

# ==========================================================================
# STEP 05: 观测矩阵 H 构造
# ==========================================================================
def get_H_DANSE(type_, n_states, n_obs):
    """
    按模型类型自动生成观测矩阵 H，用于 DANSE/RNN 训练。
    Args:
        type_ (str): 模型类型
        n_states, n_obs (int): 维数
    Returns:
        H (ndarray): 观测矩阵 [n_obs, n_states]
    """
    if type_ == "LinearSSM":
        return LinearSSM(n_states=n_states, n_obs=n_obs).construct_H()
    elif type_ == "LorenzSSM":
        return np.eye(n_obs, n_states)
    elif type_ == "SinusoidalSSM":
        return jacobian(h_sinssm_fn, torch.randn(n_states,)).numpy()

# ==========================================================================
# 未修改的代码：
# STEP 06: 参数生成器主函数（返回SSM与Estimator配置）
#     N=1000, T=100, n_states=5, n_obs=5, q2=1.0, r2=1.0,
#     inverse_r2_dB=40, nu_dB=0, device='cpu'
# ==========================================================================
def get_parameters(
    N=1000, T=100, n_states=3, n_obs=3, q2=None, r2=None,
    inverse_r2_dB=40, nu_dB=0, device='cpu'
):
    """
    自动生成 SSM 与各估计算法的配置字典。
    ---
    Args:
        N, T: 样本数与序列长度
        n_states, n_obs: 状态/观测维数
        q2, r2: 过程/观测噪声方差
        inverse_r2_dB, nu_dB: 噪声功率（dB尺度，方便论文设定）
        device: 计算设备
    Returns:
        ssm_parameters_dict, estimators_dict
    Tensor Dimensions:
        - SSM 参数: [n, n] 或 [n] 向量/矩阵
    Math Notes:
        - r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        - q2 = dB_to_lin(nu_dB - inverse_r2_dB)
    """
    # ==================================================================
    # STEP 01: 基于 dB 自动换算噪声方差
    # ==================================================================
    H_DANSE = None
    r2 = 1.0 / dB_to_lin(inverse_r2_dB)
    q2 = dB_to_lin(nu_dB - inverse_r2_dB)

    # ==================================================================
    # STEP 02: 状态空间模型参数
    # ==================================================================
    ssm_parameters_dict = {
        "LinearSSM": {
            "n_states":n_states,
            "n_obs":n_obs,
            "F":None,
            "G":np.zeros((n_states,1)),
            "H":None,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "q2":q2,
            "r2":r2,
            "N":N,
            "T":T,
            "Q":None,
            "R":None
        },
        "LorenzSSM": {
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "A_fn":A_fn,
            "h_fn":h_fn,
            "delta_d":0.02,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "use_Taylor":True
        },
        "SinusoidalSSM": {
            "n_states":n_states,
            "alpha":0.9,
            "beta":1.1,
            "phi":0.1*math.pi,
            "delta":0.01,
            "a":1.0,
            "b":1.0,
            "c":0.0,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "use_Taylor":False
        },
    }

    # ==================================================================
    # STEP 03: 估计算法参数
    # ==================================================================
    estimators_dict = {
        "danse": {
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":np.eye(n_obs,n_obs)*r2,
            "H":H_DANSE,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-2,
                    "num_epochs":2000,
                    "min_delta":5e-2,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        "pinnse": {
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":np.eye(n_obs,n_obs)*r2,
            "H":H_DANSE,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-2,
                    "num_epochs":2000,
                    "min_delta":5e-2,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        "KF": {
            "n_states":n_states,
            "n_obs":n_obs
        },
        "EKF": {
            "n_states":n_states,
            "n_obs":n_obs
        },
        "UKF": {
            "n_states":n_states,
            "n_obs":n_obs,
            "n_sigma":n_states*2,
            "kappa":0.0,
            "alpha":1e-3
        },
        "KNetUoffline": {
            "n_states":n_states,
            "n_obs":n_obs,
            "n_layers":1,
            "N_E":100,
            "N_CV":100,
            "N_T":100,
            "unsupervised":True,
            "data_file_specification":'Ratio_{}---R_{}---T_{}',
            "model_file_specification":'Ratio_{}---R_{}---T_{}---unsupervised_{}',
            "nu_dB":0.0,
            "lr":1e-3,
            "weight_decay":1e-6,
            "num_epochs":100,
            "batch_size":100,
            "device":device
        }
    }
    return ssm_parameters_dict, estimators_dict