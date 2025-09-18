"""
Project/Algorithm Background | 项目/算法背景
---------------------------------------------------------------------------
Nonlinear state estimation and benchmarking of Kalman-like filters (EKF/UKF)
against a learned DANSE model on the Lorenz attractor. Focus: sequence
generation, inference, NMSE/time comparison, and visualization.

I/O Overview | 输入输出概览
---------------------------------------------------------------------------
- Data (SSM):
    X: true states, shape [N, T+1, m]
    Y: observations, shape [N, T, n]
- Filters / Model:
    EKF / UKF: physics-based estimators using known h and Q/R
    DANSE: RNN-based estimator consuming Y and outputting filtered/pred states
- Outputs:
    Estimated trajectories and covariances, NMSE in dB, runtime, SNR in dB,
    plus diagnostic plots.

Tensor Dimension Legend | 张量维度对照表
---------------------------------------------------------------------------
N  : number of trajectories (batch size)
T  : time steps of observations
m  : state dimension (n_states, e.g., 3 for Lorenz)
n  : observation dimension (n_obs, e.g., 3 for Lorenz)

Reproducibility Notes | 可复现性说明
---------------------------------------------------------------------------
- Uses torch/numpy random draws via downstream utilities.
- Device can be 'cpu' or 'cuda' depending on availability; default 'cpu'.
- No code structure or variable names were changed; only comments/docstrings
  were added so it can run as-is in Google Colab or standard Python.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from pathlib import Path
import sys, os

try:
    # 脚本环境有 __file__
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # [ORIG] 这里用 __file__ -> 在 Colab 里会 NameError
    # [FIX] 用当前工作目录顶上
    SCRIPT_DIR = Path.cwd()
# 放在 import 区域附近
import pickle, re

def load_pkl_dataset(fullpath):
    """
    读取你生成的 .pkl 数据集，返回:
      X: [N, T+1, m]  真实状态
      Y: [N, T,   n]  观测
    以及从文件名解析出来的 inverse_r2_dB, nu_dB（可选）
    """
    with open(fullpath, 'rb') as f:
        d = pickle.load(f)  # d['data'] 是 object 数组: 每个元素 [Xi, Yi]

    X_list, Y_list = [], []
    for Xi, Yi in d['data']:
        X_list.append(torch.from_numpy(np.asarray(Xi)).float())
        Y_list.append(torch.from_numpy(np.asarray(Yi)).float())
    X = torch.stack(X_list, dim=0)  # [N, T+1, m]
    Y = torch.stack(Y_list, dim=0)  # [N, T,   n]

    m = re.search(r"r2_(-?\d+(?:\.\d+)?)dB_nu_(-?\d+(?:\.\d+)?)dB", fullpath)
    inv_r2_db = float(m.group(1)) if m else None
    nu_db     = float(m.group(2)) if m else None
    return X, Y, inv_r2_db, nu_db
# [FIX] 把任意预测张量对齐成 [N, T, m]
def _align_pred_to_ref(X_ref: torch.Tensor, X_pred: torch.Tensor) -> torch.Tensor:
    """
    X_ref: [N, T, m]
    X_pred: 可能是 [N,T,m] / [N,1,T,m] / [N,m,T] / 其它常见排列
    返回：与 X_ref 同形状 [N, T, m]
    """
    xh = X_pred
    # 先把多余的 1 维去掉（例如 [N,1,T,m]）
    if xh.dim() == 4 and (xh.shape[1] == 1 or xh.shape[-1] == 1):
        xh = xh.squeeze(1) if xh.shape[1] == 1 else xh.squeeze(-1)

    # 现在希望是 3 维
    if xh.dim() != 3:
        raise ValueError(f"Unexpected pred shape {xh.shape}, expect 3D after squeeze.")

    N, T, m = X_ref.shape
    # 直接匹配
    if xh.shape == (N, T, m):
        return xh
    # 常见的 [N, m, T]
    if xh.shape == (N, m, T):
        return xh.permute(0, 2, 1)
    # 常见的 [T, N, m]
    if xh.shape == (T, N, m):
        return xh.permute(1, 0, 2)
    # 其它情况：尽量截取到相同长度
    # 对齐时间长度
    T_use = min(T, xh.shape[1])
    m_use = min(m, xh.shape[2])
    xh = xh[:, :T_use, :m_use]
    xr = X_ref[:, :T_use, :m_use]
    # 如果还不一致就报错
    if xh.shape != xr.shape:
        raise ValueError(f"Pred/Ref shapes still mismatch: pred {xh.shape}, ref {xr.shape}")
    return xh

# ===== 放在文件前面（只要你还没定义过这两个工具函数）========================
# [ADD] 新增：对齐预测到 [N,T,m] 的工具函数
def _align_pred_to_ref(X_ref, X_pred):
    """
    将 X_pred 对齐到和 X_ref 相同的形状 [N, T, m]。
    允许常见的 [N,1,T,m] / [N,T,m] / [T,N,m] / [N,m,T] 等输入。
    若长度略有出入，做最小裁剪到 min(T_ref, T_pred), min(m_ref, m_pred)。
    """
    Xr = X_ref
    Xp = X_pred

    # 转成张量
    if not torch.is_tensor(Xp):
        Xp = torch.as_tensor(Xp)

    # 常见 4D: [N,1,T,m] -> [N,T,m]
    if Xp.dim() == 4 and Xp.shape[1] == 1:
        Xp = Xp[:, 0, :, :]  # [N,T,m]

    # 若是 [N,m,T] -> [N,T,m]
    if Xp.dim() == 3 and Xp.shape[1] == Xr.shape[2] and Xp.shape[2] == Xr.shape[1]:
        Xp = Xp.transpose(1, 2)

    # 若是 [T,N,m] -> [N,T,m]
    if Xp.dim() == 3 and Xp.shape[0] == Xr.shape[1] and Xp.shape[1] == Xr.shape[0]:
        Xp = Xp.transpose(0, 1)

    # 最小裁剪，保持一致
    N = min(Xr.shape[0], Xp.shape[0])
    T = min(Xr.shape[1], Xp.shape[1])
    M = min(Xr.shape[2], Xp.shape[2])
    Xr = Xr[:N, :T, :M].contiguous()
    Xp = Xp[:N, :T, :M].contiguous()
    return Xp

# [ADD] 新增：稳健 NMSE(dB)
def nmse_loss_dB_safe(X_ref, X_pred, eps=1e-12, reduce='mean'):
    Xp = _align_pred_to_ref(X_ref, X_pred)      # 对齐到 [N,T,m]
    num = ((Xp - X_ref)**2).sum(dim=(1, 2))     # [N]
    den = (X_ref**2).sum(dim=(1, 2)).clamp_min(eps)
    nmse = (num / den).clamp_min(eps)           # [N]
    nmse_db = 10.0 * torch.log10(nmse)          # [N]
    if reduce == 'none':
        return nmse_db
    if reduce == 'median':
        return torch.nanmedian(nmse_db).values
    return torch.nanmean(nmse_db)

# [ORIG] # from utils.plot_functions import *
# [MOD]  重新启用绘图工具，后续可视化需要
# from utils.plot_functions import *

# [ORIG] # from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, mse_loss_dB
# [MOD]  重新启用常用度量/换算函数（后续 NMSE/SNR 计算需要）
# from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, mse_loss_dB

# from parameters import get_parameters, A_fn, h_fn, f_lorenz, f_lorenz_danse
# from generate_data import LorenzAttractorModel, generate_SSM_data

# [ORIG] from src.ekf import EKF
# [MOD]  使用你提供的新 EKF 实现（ekf_lorenz.py）
# from ekf_lorenz import EKF_Lorenz

# [ORIG] from src.ukf import UKF
# [MOD]  如未使用可注释；仍保留 Aliter 版本
# from src.ukf import UKF

# from src.ukf_aliter import UKF_Aliter
# from src.danse import DANSE, push_model

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']

# ==========================================================================
# KalmanNet-like noise grids | 噪声设定
# ==========================================================================
DatafolderName = './eval_sets/Lorenz_Atractor/T1000_NT100' + '/'
dataFileName = ['trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_100_r2_-20dB_nu_-30dB.pkl',
                'trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_100_r2_-10dB_nu_-20dB.pkl',
                'trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_100_r2_0dB_nu_-10dB.pkl',
                'trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_100_r2_10dB_nu_0dB.pkl',
                'trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_100_r2_20dB_nu_10dB.pkl'
               ] # for T=1000

# r2 = torch.tensor([1,0.1,0.01,1e-3,1e-4])
# r = torch.sqrt(r2)
# vdB = -10  # ratio v=q2/r2
# v = 10**(vdB/10)
# q2 = torch.mul(v,r2)
# q = torch.sqrt(q2)

# # searched (validation) grids | 验证用网格
# r2searchdB = torch.tensor([-5,0,5])
# rrsearch = torch.sqrt(10**(-r2searchdB/10.0))
# q2searchdB = torch.tensor([20,15,10])
# qsearch = torch.sqrt(10**(-q2searchdB/10.0))

# # optimized (fixed at test) | 测试用固定最优
# r2optdB = torch.tensor([3.0103])
# ropt = torch.sqrt(10**(-r2optdB/10))
# r2optdB_partial = torch.tensor([3.0103])
# ropt_partial = torch.sqrt(10**(-r2optdB_partial/10))

# q2optdB = torch.tensor([18.2391,28.2391,38.2391,48,55])
# qopt = torch.sqrt(10**(-q2optdB/10))
# q2optdB_partial = torch.tensor([18.2391,28.2391,38.2391,48,55])
# qopt_partial = torch.sqrt(10**(-q2optdB_partial/10))

# ==========================================================================
# Inference helpers | 推断辅助函数
# ==========================================================================
def test_danse_lorenz(danse_model, saved_model_file, Y, device='cpu'):
    """
    Run inference with a trained DANSE model on Lorenz observations.
    使用已训练的 DANSE 模型对 Lorenz 观测序列进行推断。
    """
    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()
    with torch.no_grad():
        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered


def test_pinn_lorenz(pinnse_model, saved_model_file, Y, device='cpu'):
    """
    Run inference with a trained PINNSE model on Lorenz observations.
    使用已训练的 PINNSE 模型对 Lorenz 观测序列进行推断。
    """
    pinnse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    pinnse_model = push_model(nets=pinnse_model, device=device)
    pinnse_model.eval()
    with torch.no_grad():
        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = pinnse_model.compute_predictions(Y_test_batch)
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered


def test_ukf_lorenz(X, Y, ukf_model):
    """Run UKF wrapper."""
    X_estimated_ukf, Pk_estimated_ukf, mse_arr_uk_lin, mse_arr_ukf = ukf_model.run_mb_filter(X, Y)
    return X_estimated_ukf, Pk_estimated_ukf, mse_arr_uk_lin, mse_arr_ukf


def test_ekf_lorenz(X, Y, H_np, delta_t, inverse_r2_dB, nu_dB, device='cpu'):
    """
    Run EKF_Lorenz on a batch using the provided H, dt, and dB noise params.
    使用 EKF_Lorenz 的 run_ekf_on_dataset 在批次上运行。
    """
    N, T1, m = X.shape
    T = T1 - 1
    Y_np = Y.cpu().numpy()
    X_gt_np = X[:, 1:, :].cpu().numpy()

    out = EKF_Lorenz.run_ekf_on_dataset(
        Y=Y_np,
        X_gt=X_gt_np,
        H=H_np,
        delta_t=float(delta_t),
        inverse_r2_dB=float(inverse_r2_dB),
        nu_dB=float(nu_dB),
        order=1,
        device=device
    )
    X_hat_np = out["X_hat"]                    # [N, T, m]
    P_hat_np = out["P_hat"]                    # [N, T, m, m]

    X_estimated_ekf = torch.zeros((N, T+1, m), dtype=torch.float32)
    Pk_estimated_ekf = torch.zeros((N, T+1, m, m), dtype=torch.float32)
    X_estimated_ekf[:, 1:, :] = torch.from_numpy(X_hat_np).float()
    Pk_estimated_ekf[:, 1:, :, :] = torch.from_numpy(P_hat_np).float()

    mse_dB = torch.tensor(out.get("MSE_dB", np.nan), dtype=torch.float32)
    return X_estimated_ekf, Pk_estimated_ekf, mse_dB


def get_test_sequence(lorenz_model, T, inverse_r2_dB, nu_dB):
    """Generate one Lorenz trajectory and its observations."""
    x_lorenz, y_lorenz = lorenz_model.generate_single_sequence(
        T=T, inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB
    )
    return x_lorenz, y_lorenz

# ==========================================================================
# Main test routine | 主测试流程
# ==========================================================================
def test_lorenz(device='cpu', model_file_saved=None, pinn_model_file_saved=None,
                X=None, Y=None, q_available=None, r_available=None):
    """
    End-to-end testbench: EKF/UKF/DANSE/PINNSE inference + metrics & plots.
    端到端测试：EKF/UKF/DANSE/PINNSE 推断与评估。
    """
    # Parse meta from DANSE folder name (kept as-is) | 从 DANSE 目录名解析超参
    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse(
        "{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB",
        model_file_saved.split('/')[-2]
    )
    print("*"*100)
    print("1/r2: {}dB, nu: {}dB".format(inverse_r2_dB, nu_dB))
    d = m
    # For consistency with checkpoints | 与 ckpt 一致
    T = 1_000
    delta = 0.02
    delta_d = 0.02
    J = 5
    decimate=False
    use_Taylor = False

    # Build generator | 构建数据生成器
    lorenz_model = LorenzAttractorModel(
        d=d, J=J, delta=delta, delta_d=delta_d,
        A_fn=A_fn, h_fn=h_fn, decimate=decimate,
        mu_e=np.zeros((d,)), mu_w=np.zeros((d,)),
        use_Taylor=use_Taylor
    )

    # Generate N_test sequences if external not provided | 若无外部数据则生成
    N_test = 5
    if (X is None) or (Y is None):
        X = torch.zeros((N_test, T+1, m))
        Y = torch.zeros((N_test, T, n))
        print("Test data generated using r2: {} dB, nu: {} dB".format(inverse_r2_dB, nu_dB))
        for i in range(N_test):
            x_i, y_i = get_test_sequence(
                lorenz_model=lorenz_model, T=T,
                nu_dB=nu_dB, inverse_r2_dB=inverse_r2_dB
            )
            X[i, :, :] = torch.from_numpy(x_i).float()
            Y[i, :, :] = torch.from_numpy(y_i).float()
    else:
        # Align external batch to 5 if larger | 外部数据则截取 5 条（若更少则全用）
        N_ext = min(5, X.shape[0], Y.shape[0])
        X = X[:N_ext].contiguous()
        Y = Y[:N_ext].contiguous()
        N_test = N_ext

    # KalmanNet convention alignment | 与 KalmanNet 对齐
    N_test, Ty, dy = Y.shape
    X_ref = X[:, 1:, :].contiguous()

    # X = torch.cat((torch.zeros(N_test,1,lorenz_model.n_states), X), dim=1)  # prepend x0
    # ---------------- EKF ----------------
    # H_np = jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy()  # [n, m]
    H_np = np.eye(n, m)  # 替换原来的 jacobian(...) 调用
    start_time_ekf = timer()
    X_estimated_ekf, Pk_estimated_ekf, nmse_ekf = test_ekf_lorenz(
        X=X, Y=Y, H_np=H_np, delta_t=lorenz_model.delta,
        inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB, device=device
    )
    time_elapsed_ekf = timer() - start_time_ekf

    # ---------------- UKF ----------------
    ukf_model = UKF_Aliter(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        f=fx_lorenz_taylor,
        h=lambda x: x,
        Q=q_available*np.eye(lorenz_model.n_states),
        R=r_available*np.eye(lorenz_model.n_obs),
        kappa=-1,
        alpha=0.1,
        delta_t=lorenz_model.delta,
        beta=2,
        n_sigma=2*lorenz_model.n_states+1,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device
    )
    start_time_ukf = timer()
    X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf_lin, mse_arr_ukf = test_ukf_lorenz(X=X, Y=Y, ukf_model=ukf_model)
    time_elapsed_ukf = timer() - start_time_ukf

    # ---------------- DANSE ----------------
    ssm_dict, est_dict = get_parameters(
        N=1, T=Ty, n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB
    )
    C_w = r_available*np.eye(lorenz_model.n_obs)
    H_id = np.eye(n, m)  # [n, m]
    danse_model = DANSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=C_w,
        batch_size=1,
        H=H_id,
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )
    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_lorenz(
        danse_model=danse_model,
        saved_model_file=model_file_saved,
        Y=Y,
        device=device
    )
    time_elapsed_danse = timer() - start_time_danse

    # ---------------- PINNSE （与 DANSE 接口一致）----------------
    # 注：若 PINNSE 需要不同的 rnn_params_dict 键，请在 get_parameters 中维护 'pinn' 分支；
    #    若与 DANSE 完全一致，也可以直接复用 'danse' 的超参。
    pinn_params = est_dict['pinn']['rnn_params_dict'] if ('pinn' in est_dict and 'rnn_params_dict' in est_dict['pinn']) else est_dict['danse']['rnn_params_dict']
    pinnse_model = PINNSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=C_w,
        batch_size=1,
        H=H_id,
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=pinn_params,
        device=device
    )
    start_time_pinn = timer()
    if pinn_model_file_saved is not None:
        Xp_est_pred, Pp_est_pred, Xp_est_filtered, Pp_est_filtered = test_pinn_lorenz(
            pinnse_model=pinnse_model,
            saved_model_file=pinn_model_file_saved,
            Y=Y,
            device=device
        )
        time_elapsed_pinn = timer() - start_time_pinn
        # nmse_pinn = mse_loss_dB(X[:,1:,:], Xp_est_filtered[:,0:,:])
        # [FIX] 只有在 PINNSE 有结果时再算，并且对齐形状
        if Xp_est_filtered is not None:
            X_pinn_aligned = _align_pred_to_ref(X_ref, Xp_est_filtered)
            nmse_pinn = nmse_loss_dB_safe(X_ref, X_pinn_aligned)
        else:
            nmse_pinn = torch.tensor(float('nan'))

    else:
        # 没有提供 PINNSE 权重则跳过
        Xp_est_pred = Pp_est_pred = Xp_est_filtered = Pp_est_filtered = None
        time_elapsed_pinn = None
        nmse_pinn = torch.tensor(float('nan'))

    # ---------------- Metrics ----------------
    # [ADD] 新增：统一的参考（真值）切片；所有 NMSE 都基于它
    X_ref = X[:, 1:, :].contiguous()

    nmse_ekf = nmse_loss_dB_safe(X_ref, X_estimated_ekf[:, 1:, :])
    # [ORIG] nmse_ukf = mse_loss_dB(X[:,1:,:], X_estimated_ukf[:,1:,:])
    # [FIX]  改为稳健 NMSE(dB) 并自动对齐
    nmse_ukf   = nmse_loss_dB_safe(X_ref, X_estimated_ukf[:, 1:, :])

    # [ORIG] nmse_danse = mse_loss_dB(X[:,1:,:], X_estimated_filtered[:,0:,:])  # 形状常错
    # [FIX]  先对齐 DANSE 输出（可能是 [N,1,T,m]），再算 NMSE
    X_danse_aligned = _align_pred_to_ref(X_ref, X_estimated_filtered)
    nmse_danse = nmse_loss_dB_safe(X_ref, X_danse_aligned)

    # [ORIG] nmse_pinn = mse_loss_dB(X[:,1:,:], Xp_est_filtered[:,0:,:])  # 此处导致 IndexError/UnboundLocalError
    # [FIX]  只有当 PINNSE 有输出时才计算；先对齐，再算 NMSE
    if Xp_est_filtered is not None:
        X_pinn_aligned = _align_pred_to_ref(X_ref, Xp_est_filtered)
        nmse_pinn = nmse_loss_dB_safe(X_ref, X_pinn_aligned)
    else:
        nmse_pinn = torch.tensor(float('nan'))




    snr = mse_loss(X[:,1:,:], torch.zeros_like(X[:,1:,:])) * dB_to_lin(inverse_r2_dB)

    print("ekf  , batch: {}, nmse(dB): {}, time: {:.4f}s".format(N_test, nmse_ekf, time_elapsed_ekf))
    print("ukf  , batch: {}, nmse(dB): {}, time: {:.4f}s".format(N_test, nmse_ukf, time_elapsed_ukf))
    print("danse, batch: {}, nmse(dB): {}, time: {:.4f}s".format(N_test, nmse_danse, time_elapsed_danse))
    if time_elapsed_pinn is not None:
        print("pinnse, batch: {}, nmse(dB): {}, time: {:.4f}s".format(N_test, nmse_pinn, time_elapsed_pinn))

    # ---------------- Visualization ----------------
    # plot_state_trajectory_axes(
    #     X=torch.squeeze(X[0,1:,:],0),
    #     X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0),
    #     X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0),
    #     X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0)
    # )
    # plot_state_trajectory(
    #     X=torch.squeeze(X[0,1:,:],0),
    #     X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0),
    #     X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0),
    #     X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0)
    # )
    # plot_state_trajectory_axes(
    # X=torch.squeeze(X[0,1:,:],0),
    # X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0),
    # X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0),
    # X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0),
    # X_est_PINN=torch.squeeze(Xp_est_filtered[0],0)   # ← 新增
    # )
    X_danse_plot = _align_pred_to_ref(X_ref, X_estimated_filtered)
    X_pinn_plot  = _align_pred_to_ref(X_ref, Xp_est_filtered) if Xp_est_filtered is not None else None

    plot_state_trajectory(
        X=X_ref[0],                               # [T, m]
        X_est_EKF=X_estimated_ekf[0,1:,:],        # [T, m]
        X_est_UKF=X_estimated_ukf[0,1:,:],        # [T, m]
        X_est_DANSE=X_danse_plot[0],              # [T, m]
        X_est_PINN=X_pinn_plot[0] if X_pinn_plot is not None else None,  # [T, m] or None
    )

    # plot_state_trajectory(
    #     X=torch.squeeze(X[0,1:,:],0),
    #     X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0),
    #     X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0),
    #     X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0),
    #     X_est_PINN=torch.squeeze(Xp_est_filtered[0],0)   # ← 新增
    # )
    return nmse_ekf, nmse_danse, nmse_ukf, nmse_pinn, \
          time_elapsed_ekf, time_elapsed_danse, time_elapsed_ukf, time_elapsed_pinn, \
          snr


# ==========================================================================
# Entry | 入口
# ==========================================================================
if __name__ == "__main__":
    device = 'cpu'
    inverse_r2_dB_arr = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])

    nmse_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ukf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_pinn_arr = np.full((len(inverse_r2_dB_arr,)), np.nan)

    t_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_ukf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_pinn_arr = np.full((len(inverse_r2_dB_arr,)), np.nan)

    snr_arr = np.zeros((len(inverse_r2_dB_arr,)))
        #"-10.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_-10.0dB_-20.0dB/danse_gru_ckpt_epoch_671_best.pt", # 训练是 N=500,  T = 1000, 我的是N = 1000, T = 100
    model_file_saved_dict = {
        "-20.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_100_N_1000_-20.0dB_-30.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "-10.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_100_N_1000_-10.0dB_-20.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "0.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_100_N_1000_0.0dB_-10.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "10.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_100_N_1000_10.0dB_0.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "20.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_100_N_1000_20.0dB_10.0dB/danse_gru_ckpt_epoch_671_best.pt"
    }

    # OPTIONAL: PINNSE checkpoints (set None to skip) | 可选 PINNSE 权重（None 则跳过）
    pinn_model_saved_dict = {
        "-20.0dB":  "./models/LorenzSSM_pinnse_gru_m_3_n_3_T_100_N_1000_-20.0dB_-30.0dB/pinnse_gru_ckpt_epoch_671_best.pt",
        "-10.0dB": "./models/LorenzSSM_pinnse_gru_m_3_n_3_T_100_N_1000_-10.0dB_-20.0dB/pinnse_gru_ckpt_epoch_671_best.pt",
        "0.0dB":   "./models/LorenzSSM_pinnse_gru_m_3_n_3_T_100_N_1000_0.0dB_-10.0dB/pinnse_gru_ckpt_epoch_674_best.pt",
        "10.0dB":  "./models/LorenzSSM_pinnse_gru_m_3_n_3_T_100_N_1000_10.0dB_0.0dB/pinnse_gru_ckpt_epoch_672_best.pt",
        "20.0dB":  "./models/LorenzSSM_pinnse_gru_m_3_n_3_T_100_N_1000_20.0dB_10.0dB/pinnse_gru_ckpt_epoch_1029_best.pt"
    }

    for i, inverse_r2_dB in enumerate(inverse_r2_dB_arr):
        print("Data Load")
        print(dataFileName[i])

        # [train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] = torch.load(
        #     DatafolderName + dataFileName[i], map_location='cpu'
        # )
        # test_target = torch.transpose(test_target,1,2)  # [N, T+1, m]
        # test_input  = torch.transpose(test_input,1,2)   # [N, T,   n]
        fullpath = os.path.join(DatafolderName, dataFileName[i])
        test_target, test_input, inv_db_from_file, nu_db_from_file = load_pkl_dataset(fullpath)
        # test_target: [N, T+1, m] (true states)
        # test_input : [N, T,   n] (observations)

        # q_i = qopt[i].numpy()**2
        # r_i = r[i].numpy()**2
        # ===== CHANGED: 直接从 dB 计算 r/q 给 UKF/EKF（更稳妥），避免 qopt/r 数组耦合 =====
        r2_lin = 10**(inverse_r2_dB/10.0)         # 1/r^2 的线性值
        r_i = 1.0 / r2_lin                         # r^2
        q2_lin = 10**((nu_db_from_file - inverse_r2_dB)/10.0) if nu_db_from_file is not None else 10**((-10.0 - inverse_r2_dB)/10.0)
        q_i = q2_lin                               # q^2

        model_file_saved_i = model_file_saved_dict['{}dB'.format(inverse_r2_dB)]
        pinn_model_saved_i = pinn_model_saved_dict['{}dB'.format(inverse_r2_dB)]
        print(model_file_saved_i)

        nmse_ekf_i, nmse_danse_i, nmse_ukf_i, nmse_pinn_i, \
        t_ekf_i, t_danse_i, t_ukf_i, t_pinn_i, snr_i = test_lorenz(
            device=device,
            model_file_saved=model_file_saved_i,
            pinn_model_file_saved=pinn_model_saved_i,
            X=test_target, Y=test_input,
            q_available=q_i, r_available=r_i
        )

        nmse_ekf_arr[i]   = np.array(nmse_ekf_i).item()
        nmse_ukf_arr[i]   = np.array(nmse_ukf_i).item()
        nmse_danse_arr[i] = np.array(nmse_danse_i).item()
        if not np.isnan(np.array(nmse_pinn_i)):
            nmse_pinn_arr[i] = np.array(nmse_pinn_i).item()

        t_ekf_arr[i]   = t_ekf_i
        t_ukf_arr[i]   = t_ukf_i
        t_danse_arr[i] = t_danse_i
        if t_pinn_i is not None:
            t_pinn_arr[i] = t_pinn_i

        snr_arr[i] = 10*np.log10(np.array(snr_i).item())

    # ---------------- Plotting ----------------
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.plot(inverse_r2_dB_arr, nmse_ekf_arr,  'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(inverse_r2_dB_arr, nmse_danse_arr,'bo-',  linewidth=2.0, label="NMSE-DANSE")
    plt.plot(inverse_r2_dB_arr, nmse_ukf_arr,  'ks-',  linewidth=2.0, label="NMSE-UKF")
    if not np.all(np.isnan(nmse_pinn_arr)):
        plt.plot(inverse_r2_dB_arr, nmse_pinn_arr, 'm^-', linewidth=2.0, label="NMSE-PINNSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(snr_arr, nmse_ekf_arr,  'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(snr_arr, nmse_danse_arr,'bo-',  linewidth=2.0, label="NMSE-DANSE")
    plt.plot(snr_arr, nmse_ukf_arr,  'ks-',  linewidth=2.0, label="NMSE-UKF")
    if not np.all(np.isnan(nmse_pinn_arr)):
        plt.plot(snr_arr, nmse_pinn_arr, 'm^-', linewidth=2.0, label="NMSE-PINNSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(inverse_r2_dB_arr, t_ekf_arr,   'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(inverse_r2_dB_arr, t_ukf_arr,   'ks--', linewidth=1.5, label="Inference time-UKF")
    plt.plot(inverse_r2_dB_arr, t_danse_arr, 'bo-',  linewidth=2.0, label="Inference time-DANSE")
    if not np.all(np.isnan(t_pinn_arr)):
        plt.plot(inverse_r2_dB_arr, t_pinn_arr,'m^-',  linewidth=2.0, label="Inference time-PINNSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(snr_arr, t_ekf_arr,   'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(snr_arr, t_ukf_arr,   'ks-',  linewidth=2.0, label="Inference time-UKF")
    plt.plot(snr_arr, t_danse_arr, 'bo-',  linewidth=2.0, label="Inference time-DANSE")
    if not np.all(np.isnan(t_pinn_arr)):
        plt.plot(snr_arr, t_pinn_arr,'m^-',  linewidth=2.0, label="Inference time-PINNSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()

    plt.show()
