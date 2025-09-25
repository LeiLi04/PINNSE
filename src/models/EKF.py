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
