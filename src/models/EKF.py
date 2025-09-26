"""
===============================================================================
File: ekf_lorenz.py
Author: You (with ChatGPT assist, 2025)

🥁 What’s new vs EKF_Aliter:
- 不再依赖 FilterPy；全部用 PyTorch/Numpy 自己实现（便于 CUDA、可控数值稳定）。
- 统一支持 dB 参数：inverse_r2_dB 与 nu_dB -> R, Q。
- 预测步可选一阶/二阶离散化（order=1/2）；更新步用 Joseph 形式。
- Cholesky 白化求解 S^{-1}（不显式求逆），可选输出白化创新/ NIS 便于诊断。
===============================================================================
"""

import math
import pickle
import numpy as np
import torch
from torch import nn

# ---------- utils ----------
def dB_to_lin(x_dB: float) -> float:
    return 10.0 ** (x_dB / 10.0)

def lin_to_dB(x_lin: float, eps: float = 1e-12) -> float:
    return 10.0 * np.log10(max(x_lin, eps))

def mse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

#   ------- Numerical Jacobian (Central Difference) ----------
def numerical_jacobian_central(func, x: np.ndarray, dt: float = None, eps: float = 1e-5):
    x = np.asarray(x, dtype=float)
    n = x.size
    fx = func(x, dt) if dt is not None else func(x)
    fx = np.asarray(fx, dtype=float)
    m = fx.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        x_f = x.copy(); x_b = x.copy()
        x_f[i] += eps;   x_b[i] -= eps
        f_f = func(x_f, dt) if dt is not None else func(x_f)
        f_b = func(x_b, dt) if dt is not None else func(x_b)
        J[:, i] = (np.asarray(f_f) - np.asarray(f_b)) / (2.0 * eps)
    return J

# ---------- Lorenz-63 dynamics + Jacobian ----------
def lorenz_f(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x1, x2, x3 = x
    return np.array([
        sigma * (x2 - x1),
        x1 * (rho - x3) - x2,
        x1 * x2 - beta * x3
    ], dtype=float)

def lorenz_jacobian(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x1, x2, x3 = x
    return np.array([
        [-sigma,     sigma,   0.0],
        [ rho - x3, -1.0,   -x1  ],
        [   x2,       x1,   -beta]
    ], dtype=float)

# ---------- EKF class ----------
class EKF_Lorenz(nn.Module):
    """
    Extended Kalman Filter for Lorenz-63.
    - f, h: nonlinear transition/measurement. 默认 h(x)=H x。
    - F_jac, H_jac: Jacobians. 默认 Lorenz 的解析 Jc 与常数 H。
    - order=1/2: 预测步的一阶/二阶离散近似。
    - Joseph form + Cholesky solve.

    Q/R 设定：
      若传入 inverse_r2_dB 与 nu_dB，则：
        r^2 = 10^(-inverse_r2_dB/10),  q^2 = 10^((nu_dB - inverse_r2_dB)/10),
        R = r^2 I, Q = q^2 I
      否则使用 Q, R 显式参数。
    """
    def __init__(self, n_states=3, n_obs=3,
                 f=lorenz_f, h=None,
                 F_jacobian=lorenz_jacobian, H_mat=None, H_jacobian=None,
                 Q=None, R=None, delta_t=0.01,
                 inverse_r2_dB=None, nu_dB=None,
                 order=1, device="cpu", jitter=1e-6):
        super().__init__()
        self.nx = n_states
        self.nz = n_obs
        self.f = f
        self.h = h if h is not None else (lambda x: (H_mat @ x))
        self.F_jac = F_jacobian
        self.H_jac = H_jacobian if H_jacobian is not None else (lambda x: H_mat)
        self.dt = float(delta_t)
        self.order = int(order)
        self.device = device
        self.jitter = float(jitter)

        # H 默认单位阵
        if H_mat is None:
            H_mat = np.eye(self.nz, self.nx, dtype=float)
        self.H = H_mat.astype(float)

        # 从 dB 建立 Q/R（若未提供）
        if (inverse_r2_dB is not None) and (nu_dB is not None) and (Q is None) and (R is None):
            r2 = 10.0 ** (-inverse_r2_dB / 10.0)
            q2 = 10.0 ** ((nu_dB - inverse_r2_dB) / 10.0)
            Q = q2 * np.eye(self.nx)
            R = r2 * np.eye(self.nz)

        assert Q is not None and R is not None, "Q/R 未提供，也未从 dB 参数推导。"

        self.Q = torch.from_numpy(np.asarray(Q)).float().to(device)
        self.R = torch.from_numpy(np.asarray(R)).float().to(device)
        self.H_torch = torch.from_numpy(self.H).float().to(device)

        # 滤波内部态
        self.x = torch.ones(self.nx, device=device).float()
        self.P = torch.eye(self.nx, device=device).float() * 1e-5

    # ----- 初始化 -----
    def initialize(self, x0=None, P0=None):
        if x0 is None:
            self.x = torch.ones(self.nx, device=self.device).float()
        else:
            self.x = torch.from_numpy(np.asarray(x0)).float().to(self.device)
        if P0 is None:
            self.P = torch.eye(self.nx, device=self.device).float() * 1e-5
        else:
            self.P = torch.from_numpy(np.asarray(P0)).float().to(self.device)


    # =========================Transition State===================================================
    # ----- 预测 -----
    # ---------- EKF Prediction step ----------
    # 数学公式说明：
    #
    # 状态预测：
    #   一阶近似 (order=1):
    #       x_{k+1} ≈ x_k + Δt f(x_k)
    #   二阶近似 (order=2):
    #       x_{k+1} ≈ x_k + Δt f(x_k) + 1/2 Δt^2 J(x_k) f(x_k)
    #
    # 状态转移矩阵 F：
    #   一阶近似 (order=1):
    #       F_k ≈ I + Δt J(x_k)
    #   二阶近似 (order=2):
    #       F_k ≈ I + Δt J(x_k) + 1/2 (Δt J(x_k))^2
    #
    # 协方差传播：
    #       P_{k+1} = F_k P_k F_k^T + Q
    #
    # 其中：
    #   f(x)   : 连续时间动力学 (Lorenz-63)
    #   J(x)   : f(x) 的 Jacobian
    #   Δt     : 时间步长 self.dt
    #   Q      : 过程噪声协方差
    def predict_step(self):
        x_np = self.x.detach().cpu().numpy()
        f = self.f(x_np)                       # R^nx
        J = self.F_jac(x_np)                  # (nx,nx)
        I = torch.eye(self.nx, device=self.device).float()
        J_t = torch.from_numpy(J).float().to(self.device)

        if self.order == 1:
            F = I + self.dt * J_t  
            x_pred = self.x + self.dt * torch.from_numpy(f).float().to(self.device)
        elif self.order == 2:
            Jdt = self.dt * J_t
            F = I + Jdt + 0.5 * (Jdt @ Jdt)
            x_pred = self.x + self.dt * torch.from_numpy(f).float().to(self.device) \
                     + 0.5 * (self.dt**2) * (J_t @ torch.from_numpy(f).float().to(self.device))
        else:
            raise ValueError("order must be 1 or 2")

        P_pred = F @ self.P @ F.T + self.Q
        self.x = x_pred
        self.P = 0.5 * (P_pred + P_pred.T)    # 对称化，避免数值漂移

        return F  # 便于调试/保存

    # ----- 更新（Joseph + Cholesky） -----
    # ---------- 协方差更新 (Joseph form) ----------
    # 数学公式：
    #
    #   P_k+ = (I - K_k H_k) P_k- (I - K_k H_k)^T + K_k R_k K_k^T
    #
    # 其中：
    #   P_k- : 先验协方差
    #   P_k+ : 后验协方差
    #   K_k  : 卡尔曼增益
    #   H_k  : 观测矩阵
    #   R_k  : 观测噪声协方差

    def update_step(self, z_k: np.ndarray, return_diag=False):
        H = self.H_torch
        z_pred = H @ self.x if self.h is None else torch.from_numpy(self.h(self.x.detach().cpu().numpy())).float().to(self.device)
        innov = torch.from_numpy(np.asarray(z_k)).float().to(self.device) - z_pred

        S = H @ self.P @ H.T + self.R
        S = 0.5 * (S + S.T) + self.jitter * torch.eye(self.nz, device=self.device)
        L = torch.linalg.cholesky(S)
        
        # 计算 Kalman 增益
        # K = P H^T S^{-1} via Cholesky
        HP = H @ self.P                      # (nz,nx)
        U = torch.cholesky_solve(HP, L)      # S^{-1} (H P)
        K = U.T                              # (nx,nz)

        # 更新
        self.x = self.x + K @ innov
        I = torch.eye(self.nx, device=self.device)
        P_post = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        self.P = 0.5 * (P_post + P_post.T)

        if return_diag:
            # 白化创新与 NIS（便于诊断）
            nu = torch.cholesky_solve(innov.unsqueeze(1), L).squeeze(1)  # S^{-1} innov
            NIS = (innov * nu).sum()
            return K, L, innov, NIS
        return K

    # ----- 跑一条序列 -----
    @torch.no_grad()
    def filter_sequence(self, Y: torch.Tensor, X_gt: torch.Tensor = None):
        """
        Y: [T, nz];  X_gt: [T, nx] (optional, 仅用于评估)
        返回：
          X_hat: [T, nx]  后验均值
          P_hat: [T, nx, nx]  后验协方差
          stats: dict (可含 MSE/NIS)
        约定：在 t=0 使用初始化 (x0,P0)，从第一帧观测 Y[0] 开始 predict+update。
        """
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        T = Y.shape[0]

        X_hat = torch.zeros((T, self.nx), device=self.device).float()
        P_hat = torch.zeros((T, self.nx, self.nx), device=self.device).float()
        nis_list = []

        for t in range(T):
            self.predict_step()
            _, L, innov, NIS = self.update_step(Y[t].cpu().numpy(), return_diag=True)
            nis_list.append(NIS.item())
            X_hat[t] = self.x
            P_hat[t] = self.P

        stats = {}
        if X_gt is not None:
            X_gt = X_gt.to(self.device)
            mse_lin = mse_loss(X_gt, X_hat)
            stats["mse_lin"] = mse_lin.item()
            stats["mse_dB"] = lin_to_dB(mse_lin.item())
        stats["NIS_mean"] = float(np.mean(nis_list))
        return X_hat, P_hat, stats


    @staticmethod
    @torch.no_grad()
    def run_ekf_on_dataset(
        Y,                      # np.array [N, T, n_obs]
        X_gt=None,              # np.array [N, T, n_states]（评估用，可 None）
        H=None,                 # np.array [n_obs, n_states]，默认 I
        delta_t=0.01,           # 时间步长
        inverse_r2_dB=None,     # (1/r^2)_dB
        nu_dB=None,             # 固定 Q=-10dB 时：nu_dB = inverse_r2_dB - 10
        order=1,
        device="cpu",
        x0=None, P0=None
    ):
        N, T, n_obs = Y.shape
        if H is None:
            # 若没给 H，则假设 n_states == n_obs，用单位阵
            n_states = X_gt.shape[2] if X_gt is not None else n_obs
            H = np.eye(n_obs, n_states, dtype=float)
        else:
            n_states = H.shape[1]

        # 直接用类名实例化；不再 import 自己
        ekf = EKF_Lorenz(n_states=n_states, n_obs=n_obs,
                         H_mat=H, delta_t=delta_t,
                         inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB,
                         order=order, device=device)

        X_hat = np.zeros((N, T, n_states), dtype=float)
        P_hat = np.zeros((N, T, n_states, n_states), dtype=float)
        mse_list, nis_list = [], []

        for i in range(N):
            # ---- 更稳妥的默认初始化：用首帧观测估个 x0；P0 取大一点 ----
            if x0 is None:
                y0 = Y[i, 0]
                # H=I 时 x0=y0；一般情形用伪逆
                x0_i = y0 if (H.shape[0] == H.shape[1] and np.allclose(H, np.eye(n_obs))) \
                       else np.linalg.pinv(H) @ y0
            else:
                x0_i = x0
            P0_i = (np.eye(n_states) * 5.0) if (P0 is None) else P0

            ekf.initialize(x0=x0_i, P0=P0_i)

            Xhat_i, Phat_i, stats = ekf.filter_sequence(
                Y=torch.from_numpy(Y[i]).float(),
                X_gt=(torch.from_numpy(X_gt[i]).float() if X_gt is not None else None)
            )
            X_hat[i] = Xhat_i.cpu().numpy()
            P_hat[i] = Phat_i.cpu().numpy()
            nis_list.append(stats["NIS_mean"])
            if "mse_lin" in stats: mse_list.append(stats["mse_lin"])
        # 汇总输出， if 有 X 则加上,  否则只输出 /hat{x}, /hat{P}, NIS
        out = {
            "X_hat": X_hat,
            "P_hat": P_hat,
            "NIS_mean": float(np.mean(nis_list)),
        }
        if mse_list:
            mse_lin = float(np.mean(mse_list))
            out.update({
                "MSE_lin": mse_lin,
                "MSE_dB": 10*np.log10(max(mse_lin, 1e-20))
            })
        return out


def _load_lorenz_dataset(dataset_path):
    """Load trajectory pickle generated on another host (handles numpy._core rename)."""
    class _CompatUnpickler(pickle.Unpickler):  # local helper to keep scope tight
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with open(dataset_path, "rb") as handle:
        return _CompatUnpickler(handle).load()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="EKF_Lorenz quick-look demo on a stored dataset.")
    parser.add_argument(
        "--dataset",
        default="src/data/trajectories/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_20.0dB_nu_-10.0dB.pkl",
        help="Path to the trajectory pickle (default: provided Lorenz dataset).",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualise (default: 0).")
    parser.add_argument("--order", type=int, default=1, choices=(1, 2), help="Prediction Taylor order.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    payload = _load_lorenz_dataset(dataset_path)
    num_samples = payload["num_samples"]
    if not (0 <= args.index < num_samples):
        raise IndexError(f"index {args.index} out of range (0 ≤ idx < {num_samples}).")

    sample_X, sample_Y = payload["data"][args.index]
    sample_X = np.asarray(sample_X, dtype=float)
    sample_Y = np.asarray(sample_Y, dtype=float)

    ekf = EKF_Lorenz(
        n_states=sample_X.shape[1],
        n_obs=sample_Y.shape[1],
        delta_t=0.02,
        inverse_r2_dB=0.0,
        nu_dB=-10.0,
        order=args.order,
        device="cpu",
    )

    ekf.initialize(x0=sample_Y[0], P0=np.eye(sample_X.shape[1]) * 5.0)

    torch_Y = torch.from_numpy(sample_Y).float()
    torch_X = torch.from_numpy(sample_X[1:]).float()
    X_hat, _, stats = ekf.filter_sequence(torch_Y, torch_X)

    x_hat_np = X_hat.cpu().numpy()
    mse_per_step = np.mean((x_hat_np - sample_X[1:]) ** 2, axis=1)

    time_axis = np.arange(sample_Y.shape[0])
    '''
    #     Project / Algorithm 背景:
    #     非线性系统(Lorenz)下的状态估计可视化：展示真值轨迹、观测数据与滤波估计，并绘制逐步均方误差曲线。

    # Inputs / 输入:
    #     sample_X : ndarray, shape [T, 3]
    #         True state trajectory x(t). 列为 (x1, x2, x3)。
    #     sample_Y : ndarray, shape [T, 3]
    #         Observation sequence y(t)。与状态维度一致的观测向量。
    #     x_hat_np : ndarray, shape [T, 3]
    #         Estimated state trajectory x_hat(t)，来自 EKF/UKF/Transformer-Filter 等估计器。
    #     time_axis : ndarray, shape [T]
    #         离散时间索引，用于横轴绘制 MSE 曲线。
    #     mse_per_step : ndarray, shape [T]
    #         每个时间步的状态估计均方误差，定义如 mean((x_hat - x_true)^2) over state dims。
    #     stats : dict
    #         汇总统计量，例如:
    #           - 'NIS_mean' : 平均 NIS 指标（若使用一致性检验）
    #           - 'mse_lin'  : 线性尺度的整体 MSE (例如全时序平均)
    #           - 'mse_dB'   : 以 10*log10(mse_lin) 表示的 dB 尺度误差

    # Outputs / 输出:
    #     - "lorenz_ekf_states.png": 3D 轨迹图 (真值 / 观测 / 估计)。
    #     - "lorenz_ekf_mse.png"   : 逐步 MSE 曲线图。
    #     - 控制台打印统计摘要 (NIS_mean, MSE_lin, MSE_dB)。

    # Tensor Dimensions / 张量维度对照:
    #     T : 时间步长度，state_dim = 3
    #     sample_X[t]     ∈ R^3
    #     sample_Y[t]     ∈ R^3
    #     x_hat_np[t]     ∈ R^3
    #     mse_per_step[t] ∈ R
    #     time_axis[t]    ∈ R

    # Math Notes / 数学说明 (纯文本公式):
    #     per-step MSE:
    #         mse[t] = mean( (x_hat_np[t, :] - sample_X[t, :])^2 )
    #     Summary metrics (示例):
    #         mse_lin = mean_t mse[t]
    #         mse_dB  = 10 * log10( mse_lin )
    #     以上仅示意，具体计算应与上游统计模块保持一致。

    # Target Audience / 目标读者:
    #     熟悉数值计算与信号处理的研究者与工程师，希望复现实验图与评估指标。
    '''
    
# ==========================================================================
# STEP 01: 配置 Matplotlib 后端与兼容性补丁
# ==========================================================================
# 说明:
# - 使用 "Agg" 后端以便在无显示环境(如服务器/CI/Colab 后台)生成并保存静态图像。
# - 一些老版本/特定发行版的 matplotlib.cbook 中缺失内部符号。
#   通过 hasattr 检查并注入最小替代实现，确保后续代码在多环境下可运行。
# 注意:
# - 这些补丁是工程性兼容措施，不改变绘图语义，仅避免 AttributeError。
import matplotlib
matplotlib.use("Agg")

import matplotlib.cbook as _mpl_cbook

if not hasattr(_mpl_cbook, "_is_pandas_dataframe"):
    # ----------------------------------------------------------------------
    # 补丁 1: 标记对象是否为 pandas.DataFrame
    # 在某些版本中该内部工具函数不存在，定义一个保守返回 False 的替代。
    def _is_pandas_dataframe(_obj):
        return False

    _mpl_cbook._is_pandas_dataframe = _is_pandas_dataframe

if not hasattr(_mpl_cbook, "_Stack") and hasattr(_mpl_cbook, "Stack"):
    # ----------------------------------------------------------------------
    # 补丁 2: 将公开的 Stack 映射为内部名 _Stack，满足旧代码引用。
    _mpl_cbook._Stack = _mpl_cbook.Stack

if not hasattr(_mpl_cbook, "_ExceptionInfo"):
    # ----------------------------------------------------------------------
    # 补丁 3: 定义 _ExceptionInfo 命名元组，便于统一异常信息封装。
    from collections import namedtuple

    _mpl_cbook._ExceptionInfo = namedtuple("_ExceptionInfo", "exc_class exc traceback")

if not hasattr(_mpl_cbook, "_broadcast_with_masks"):
    # ----------------------------------------------------------------------
    # 补丁 4: 定义 _broadcast_with_masks，最小实现仅进行 numpy 广播。
    # 用途: 对输入数组进行形状广播，返回广播后的元组。
    import numpy as np
    def _broadcast_with_masks(*arrays):
        broadcasted = np.broadcast_arrays(*[np.asarray(arr) for arr in arrays])
        return tuple(broadcasted)

    _mpl_cbook._broadcast_with_masks = _broadcast_with_masks

# ==========================================================================
# STEP 02: 导入绘图 API 与 3D 工具
# ==========================================================================
# - pyplot: 面向状态机的绘图接口，便捷生成 Figure/Axes。
# - mpl_toolkits.mplot3d: 以 3D 投影绘制 Lorenz 轨迹（x1, x2, x3）。
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (projection side-effects)

# ==========================================================================
# STEP 03: 创建画布与三个 3D 子图 (真值 / 观测 / 估计)
# ==========================================================================
# 布局:
#   [1, 3] 的子图栅格:
#     - ax_true:  真值 x(t)
#     - ax_obs:   观测 y(t)
#     - ax_est:   估计 x_hat(t)
# 说明:
#   - figsize 采用横向宽屏，便于并排比较三条轨迹的几何差异。
fig = plt.figure(figsize=(18, 5))
ax_true = fig.add_subplot(1, 3, 1, projection="3d")
ax_obs = fig.add_subplot(1, 3, 2, projection="3d")
ax_est = fig.add_subplot(1, 3, 3, projection="3d")

# ==========================================================================
# STEP 04: 绘制真值轨迹 x(t)
# ==========================================================================
# 数据约定:
#   sample_X[1:, i] 对应第 i 个状态分量的时间序列，i ∈ {0,1,2}
# 细节:
#   - 从 1 开始索引，避开可能的初始过渡点 (若首点不可视化)。
#   - label 用于图例区分。
ax_true.plot(sample_X[1:, 0], sample_X[1:, 1], sample_X[1:, 2], label="x (true)", color="tab:blue", linewidth=1.5)
ax_true.set_title("True state x")
ax_true.set_xlabel("x1")
ax_true.set_ylabel("x2")
ax_true.set_zlabel("x3")
ax_true.legend(fontsize="small")

# ==========================================================================
# STEP 05: 绘制观测轨迹 y(t)
# ==========================================================================
# 说明:
#   - 观测 y 可能受测量噪声/观测模型影响，与真值存在偏差。
#   - 通过 3D 轨迹形态对比可直观看到噪声与观测函数的影响。
ax_obs.plot(sample_Y[:, 0], sample_Y[:, 1], sample_Y[:, 2], label="y (obs)", color="tab:green", linewidth=1.2)
ax_obs.set_title("Observation y")
ax_obs.set_xlabel("y1")
ax_obs.set_ylabel("y2")
ax_obs.set_zlabel("y3")
ax_obs.legend(fontsize="small")

# ==========================================================================
# STEP 06: 绘制估计轨迹 x_hat(t)
# ==========================================================================
# 说明:
#   - 估计轨迹应尽可能贴近真值轨迹；偏差与延迟反映滤波器性能。
#   - 若采用 EKF/UKF/Particle/NN-Filter，差异来源包括线性化误差、噪声建模、网络表达等。
ax_est.plot(x_hat_np[:, 0], x_hat_np[:, 1], x_hat_np[:, 2], label="$\\hat{x}$ (estimate)", color="tab:orange", linewidth=1.5)
ax_est.set_title("Estimate $\\hat{x}$")
ax_est.set_xlabel("x1")
ax_est.set_ylabel("x2")
ax_est.set_zlabel("x3")
ax_est.legend(fontsize="small")

# 优化整体布局，避免子图元素重叠；保存状态轨迹图。
fig.tight_layout()
fig.savefig("lorenz_ekf_states.png", dpi=300)

# ==========================================================================
# STEP 07: 绘制逐步 MSE 曲线
# ==========================================================================
# 含义:
#   - mse_per_step[t] 度量 t 时刻的估计误差强度。曲线越低越好。
#   - 以 time_axis 为横轴，有助于定位误差峰值与稳态收敛段。
# 设计:
#   - 使用细虚线 baseline=0 便于参照。
fig_mse = plt.figure(figsize=(10, 3))
plt.plot(time_axis, mse_per_step, label="per-step MSE", color="tab:red")
plt.axhline(0.0, color="black", linewidth=0.8, linestyle=":", label="zero baseline")
plt.ylabel("MSE")
plt.xlabel("Time step")
plt.title("Per-step state estimation MSE")
plt.legend()
plt.grid(True, linewidth=0.3, alpha=0.7)
fig_mse.tight_layout()
fig_mse.savefig("lorenz_ekf_mse.png", dpi=600)

# ==========================================================================
# STEP 08: 打印汇总统计
# ==========================================================================
# 说明:
#   - NIS_mean: 若实现了 NIS 一致性检验，其均值应接近状态维度 (在理想假设下)。
#   - mse_lin / mse_dB: 便于在日志中快速比较不同模型或超参配置。
print(
    f"Summary stats -> NIS_mean: {stats['NIS_mean']:.3f}, "
    f"MSE_lin: {stats['mse_lin']:.3f}, MSE_dB: {stats['mse_dB']:.3f}"
)

# ==========================================================================
# STEP 09: 刷新绘图
# ==========================================================================
# 说明:
#   - 即便使用 "Agg" 后端，调用 plt.show() 在交互环境下也可触发渲染，
#     在无显示环境中不会弹窗，但不影响图像保存至文件。
plt.show()






