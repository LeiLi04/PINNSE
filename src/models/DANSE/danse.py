from __future__ import annotations

import math

import torch
from torch import nn

from .rnn import RNN_model


def create_diag(values: torch.Tensor) -> torch.Tensor:
    """Embed the last dimension of ``values`` as the diagonal of a matrix."""

    if not torch.is_tensor(values):
        values = torch.as_tensor(values, dtype=torch.float32)
    return torch.diag_embed(values)

# ==========================================================================
# STEP 02: DANSE 滤波主框架 (DANSE Estimator Main Class)
# ==========================================================================
class DANSE(nn.Module):
    """
    DANSE: Data-driven Non-linear State Estimator

    Args:
        n_states (int): 状态变量维度
        n_obs (int): 观测变量维度
        mu_w (np.ndarray): 观测噪声均值 [n_obs, 1]
        C_w (np.ndarray): 观测噪声协方差 [n_obs, n_obs]
        H (np.ndarray): 观测矩阵 [n_obs, n_states]
        mu_x0 (np.ndarray): 初始状态均值 [n_states, 1]
        C_x0 (np.ndarray): 初始状态协方差 [n_states, n_states]
        batch_size (int): 批量大小
        rnn_type (str): RNN 类型
        rnn_params_dict (dict): RNN参数字典
        device (str): 运算设备

    Returns:
        None

    Tensor Dimensions:
        Y (torch.Tensor): [batch_size, T, n_obs]
        mu_xt_yt_prev: [batch_size, T, n_states]
        L_xt_yt_prev: [batch_size, T, n_states, n_states]

    Math Notes:
        - 先验：mu_xt_yt_prev, L_xt_yt_prev
        - 边缘分布: mu_yt_current, L_yt_current
        - 后验更新: 卡尔曼更新公式
    """
    def __init__(self, n_states, n_obs, mu_w, C_w, H, mu_x0, C_x0, batch_size, rnn_type, rnn_params_dict, device='cpu'):
        super(DANSE, self).__init__()
        self.device = device
        self.n_states = n_states
        self.n_obs = n_obs
        self.mu_x0 = self.push_to_device(mu_x0)
        self.C_x0 = self.push_to_device(C_x0)
        self.mu_w = self.push_to_device(mu_w)
        self.C_w = self.push_to_device(C_w)
        self.H = self.push_to_device(H)
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.rnn = RNN_model(**rnn_params_dict[self.rnn_type]).to(self.device)
        self.mu_xt_yt_current = None
        self.L_xt_yt_current = None
        self.mu_yt_current = None
        self.L_yt_current = None
        self.mu_xt_yt_prev = None
        self.L_xt_yt_prev = None
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # self.deltaK_net = DeltaKNet(n_state=self.n_states, n_meas=self.n_obs, hidden=128, alpha=0.1).to(self.device)
        # self.H_is_I = torch.allclose(self.H, torch.eye(self.n_obs, self.n_states).to(self.device))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def push_to_device(self, x):
        """
        将 numpy 数组转换并拷贝到目标设备

        Args:
            x (np.ndarray): 输入张量或数组

        Returns:
            torch.Tensor: 转换后张量，float32
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def compute_prior_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        """
        计算先验状态分布参数

        Args:
            mu_xt_yt_prev (torch.Tensor): 先验均值 [batch, T, n_states]
            L_xt_yt_prev (torch.Tensor): 先验协方差对角元素 [batch, T, n_states]

        Returns:
            mu_xt_yt_prev (torch.Tensor): 先验均值
            L_xt_yt_prev (torch.Tensor): 先验协方差对角阵 [batch, T, n_states, n_states]

        Math Notes:
            L_xt_yt_prev = diag(L_xt_yt_prev)
        """
        self.mu_xt_yt_prev = mu_xt_yt_prev
        # create_diag(.)
        # 如果是标量 → 1×1 矩阵
        # 如果是向量 → 向量元素依次放在对角线
        self.L_xt_yt_prev = create_diag(L_xt_yt_prev)
        return self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        # -----------------------------------------------------------------------------
        # 2. 边缘分布 (Marginal Distribution)
        #
        # “边缘分布”来源于对状态变量 x_t 的积分（边缘化）。
        # 根据状态空间模型：
        #     y_t = H x_t + w_t
        #
        # 其中：
        #     x_t | y_{1:t-1} ~ N( μ_{x_t|y_{1:t-1}}, L_{x_t|y_{1:t-1}} )   (先验)
        #     w_t ~ N( μ_w, C_w )
        #
        # 因此观测的边缘分布为：
        #     p(y_t | y_{1:t-1}) = ∫ p(y_t | x_t) p(x_t | y_{1:t-1}) dx_t
        #
        # 在高斯情况下，积分结果仍为高斯分布：
        #     y_t | y_{1:t-1} ~ N( H μ_{x_t|y_{1:t-1}} + μ_w,
        #                          H L_{x_t|y_{1:t-1}} H^T + C_w )
        #
        # 在代码中，对应的参数为：
        #     mu_yt_current = H μ_{x_t|y_{1:t-1}} + μ_w
        #     L_yt_current  = H L_{x_t|y_{1:t-1}} H^T + C_w
        # -----------------------------------------------------------------------------

        """
        计算观测的边缘分布参数

        Args:
            mu_xt_yt_prev (torch.Tensor): 状态先验均值 [batch, T, n_states]
            L_xt_yt_prev (torch.Tensor): 状态先验协方差 [batch, T, n_states, n_states]

        Returns:
            None

        Math Notes:
            mu_yt = H @ mu_xt_yt_prev + mu_w, || y_{t}|y_{t-1} <--H(.)-- x_t|y_{t-1}
            L_yt = H @ L_xt_yt_prev @ H.T + C_w
        """
        # (B,T,D)+D.squeeze(-1)
        self.mu_yt_current = torch.einsum('ij,ntj->nti',self.H, mu_xt_yt_prev) + self.mu_w.squeeze(-1)
        self.L_yt_current = self.H @ L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w
    # -------------------------Original---------------------------------
    def compute_posterior_mean_vars(self, Yi_batch):
        """
        卡尔曼后验分布参数更新

        Args:
            Yi_batch (torch.Tensor): 当前观测 [batch, T, n_obs]

        Returns:
            mu_xt_yt_current (torch.Tensor): 后验均值
            L_xt_yt_current (torch.Tensor): 后验协方差

        Math Notes:
            Re_t = H @ L_xt_yt_prev @ H.T + C_w
            K_t = L_xt_yt_prev @ H.T @ Re_t^{-1}
            mu = mu_xt_yt_prev + K_t @ (Y - H @ mu_xt_yt_prev)
            L = L_xt_yt_prev - K_t @ Re_t @ K_t.T
        """

        Re_t_inv = torch.inverse(self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w)
        self.K_t = (self.L_xt_yt_prev @ (self.H.T @ Re_t_inv))
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.einsum(
            'ntij,ntj->nti',
            self.K_t, (Yi_batch - torch.einsum('ij,ntj->nti',self.H,self.mu_xt_yt_prev))
        )
        self.L_xt_yt_current = self.L_xt_yt_prev - torch.einsum(
            'ntij,ntkl->ntik',
            torch.einsum('ntij,ntjk->ntik',
                         self.K_t, self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w),
            self.K_t
        )
        return self.mu_xt_yt_current, self.L_xt_yt_current
    # -------------------------Original_End---------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++结合了DeltaK的++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def compute_posterior_mean_vars(self, Yi_batch):
        """
        卡尔曼后验分布参数更新（引入增量 Kalman-Gain 网络）
        """
        # =========【STEP 1】标准 Kalman-Gain (baseline) 解析公式 ==========
        # Re_t: 观测协方差 [B, T, m, m]
        Re_t = self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w
        Re_t_inv = torch.inverse(Re_t)
        # K_lin: baseline 的 K, [B, T, n, m]
        K_t = self.L_xt_yt_prev @ self.H.transpose(-2, -1) @ Re_t_inv
        # =========【STEP 2】准备增量 K 的神经网络输入 ==========
        # 预测观测 y_pred [B, T, m]
        y_pred = torch.einsum('ij,ntj->nti', self.H, self.mu_xt_yt_prev) + self.mu_w.squeeze(-1)
        #  残差绝对值 |y_t - y_pred|  [B, T, m]
        # resid_abs = torch.abs(Yi_batch - y_pred)
        # 先验协方差主对角 diag [B, T, n]
        # sigma_diag = torch.diagonal(self.L_xt_yt_prev, dim1=-2, dim2=-1)
        # =========【STEP 3】增量 Kalman-Gain 网络（MLP）预测 ΔK ==========
        # delta_K_list = []
        # for t in range(Yi_batch.shape[1]):
        #     # 逐时间步丢进 MLP, 输出 ΔK [B, n, m]
        #     delta_K = self.deltaK_net(resid_abs[:, t, :], sigma_diag[:, t, :])
        #     delta_K_list.append(delta_K)
        # 堆回 [B, T, n, m]
        # delta_K = torch.stack(delta_K_list, dim=1)
        # =========【STEP 4】合成最终 Kalman-Gain ==========
        # K_t = K_lin + delta_K  # “解析解 + 神经网络微调”
        # =========【STEP 5】标准 Kalman 更新 ==========
        # 创新项 [B, T, m, 1]
        innov = (Yi_batch - y_pred).unsqueeze(-1)
        # 均值后验 [B, T, n]
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.matmul(K_t, innov).squeeze(-1)
        # 协方差后验 [B, T, n, n]
        # \Sigma_posterior = (I - K_t H) L_{prior}
        I = torch.eye(self.n_states, device=self.device)
        # KH = torch.matmul(K_t, self.H.unsqueeze(0).expand(K_t.shape[0], -1, -1))  # [B,T,n,n]
        # K_t: [B,T,n,m], H: [m,n]  -> KH: [B,T,n,n]
        # KH = torch.einsum('btnm,mn->btnn', K_t, self.H)
        KH = torch.einsum('btmn,nq->btmq', K_t, self.H)   # 结果形状 [B,T,m,m]
        self.L_xt_yt_current = (I.unsqueeze(0).unsqueeze(0) - KH) @ self.L_xt_yt_prev
        return self.mu_xt_yt_current, self.L_xt_yt_current
    #+++++++++++++++++++结合了DeltaK_End++++++++++++++++++++++++++++++++++++
    # ====== [修改] 采用 m=states, n=obs 的维度约定 ======
    '''
    def compute_posterior_mean_vars(self, Yi_batch):
        """
        后验参数更新：K = K_lin + ΔK
        维度约定：
            m = self.n_states  # 状态维
            n = self.n_obs     # 观测维

        [MOD] 本函数将 ΔK-Net 的输入从 |Δy_t| 改为“白化创新” ν_t = L_t^{-1} Δy_t，
              其中 S_t = H Σ^- H^T + R_0,  S_t = L_t L_t^T （Cholesky 因子）。
        """
        B, T, _ = Yi_batch.shape
        m, n = self.n_states, self.n_obs

        H  = self.H                    # [n, m]
        P  = self.L_xt_yt_prev         # [B, T, m, m] = Σ^-_t
        mu = self.mu_xt_yt_prev        # [B, T, m]

        # --- 观测预测与创新 Δy_t ---
        y_pred = torch.einsum('nm,btm->btn', H, mu) + self.mu_w.squeeze(-1)   # [B, T, n]
        innov  = Yi_batch - y_pred                                           # [B, T, n] = Δy_t

        # --- 创新协方差 S_t = H P H^T + R_0 ---
        A = torch.einsum('nm,btmk->btnk', H, P)                              # [B, T, n, m] = H P
        S = torch.einsum('btnk,nm->btnm', A, H) + self.C_w                   # [B, T, n, n]
        eps = 1e-6
        I_n = torch.eye(n, device=self.device).expand(B, T, n, n)            # [B, T, n, n]
        S_pd = S + eps * I_n                                                 # [B, T, n, n]
        chol_S = torch.linalg.cholesky(S_pd)                                  # [B, T, n, n], 下三角 L_t

        # --- baseline K_lin = P H^T S^{-1}（用 Cholesky 稳定求解） ---
        # 先算 X = S^{-1} A = (H P)^T 的“被 S^{-1} 左乘”，再转置得到 K_lin


        # [MOD] 数值稳性：给 S_t 加 jitter 再 Cholesky
        # 先算 X = S^{-1} A = (H P)^T 的“被 S^{-1} 左乘”，再转置得到 K_lin

        # torch.cholesky_solve(A, chol_S) 等价于解线性方程组：
        #     S X = A   ⟺   (L L^T) X = A
        # 其中 chol_S = L 是 S 的 Cholesky 分解（下三角因子）。

        # 求解过程等价于：
        #   1) L Y = A        → 解出 Y
        #   2) L^T X = Y      → 解出 X
        #   ⇒ X = S^{-1} A

        # 接着转置：
        #   K_lin = X^T
        #         = (S^{-1} A)^T
        #         = A^T S^{-T}

        # 由于 S 对称正定，S^{-T} = S^{-1}。
        # 又因为 A = (H P)^T，且 P 对称 ⇒ A^T = P H^T

        # 所以：
        #   K_lin = P H^T S^{-1}

        # 这正是基线 Kalman 增益 (Kalman gain)，形状为 [B, T, m, n]。


        X = torch.cholesky_solve(A, chol_S)                                  # [B, T, n, m]
        K_lin = X.transpose(-2, -1)                                          # [B, T, m, n]

        # [MOD] 计算“白化创新” ν_t = L_t^{-1} Δy_t  —— 不要取 abs，保留符号信息
        # solve L_t * z = Δy_t  →  z = ν_t

        # [MOD] 计算“白化创新” ν_t = L_t^{-1} Δy_t
        # ---------------------------------------------------------
        # torch.linalg.solve_triangular 专门用来解三角线性方程组。
        # 这里 chol_S = L_t 是 S_t 的 Cholesky 分解（下三角矩阵）。
        #
        # 我们要解的方程是：
        #     L_t z = Δy_t
        # 解得：
        #     z = ν_t = L_t^{-1} Δy_t
        #
        # 这一步实现了创新 Δy_t 的“白化 (whitening)”，
        # 使得 ν_t ~ N(0, I)，数值稳定且便于计算似然。
        #

        nu = torch.linalg.solve_triangular(
            chol_S,                      # L_t  [B, T, n, n]
            innov.unsqueeze(-1),         # Δy_t [B, T, n, 1]
            upper=False           # 表示 chol_S 是下三角
        ).squeeze(-1)                    # [B, T, n]

        # 先验对角：diag(Σ^-_t)
        sigma_diag = torch.diagonal(P, dim1=-2, dim2=-1)                     # [B, T, m]

        # [MOD] 向量化喂入 ΔK-Net：输入改为 (nu_t, diagΣ^-)
        BT = B * T
        delta_K = self.deltaK_net(                                           # [BT, m, n]
            resid_abs = nu.view(BT, n),            # << 现在传 ν_t
            sigma_diag = sigma_diag.view(BT, m)
        ).view(B, T, m, n)                                                   # [B, T, m, n]

        # 合成增益并更新
        K_t    = K_lin + delta_K                                             # [B, T, m, n]
        mu_post = mu + torch.einsum('btmn,btn->btm', K_t, innov)             # [B, T, m]

        # 协方差更新
        I  = torch.eye(m, device=self.device).expand(B, T, m, m)             # [B, T, m, m]
        KH = torch.einsum('btmn,np->btmp', K_t, H)                           # [B, T, m, m]

        # [OPT] 更稳的 Joseph 形式（保持半正定）：
        # P_post = (I - K H) P (I - K H)^T + K R_0 K^T
        # 取消注释以启用 Joseph 更新；默认保持你原先的简式。
        M = I - KH
        # P_post = M @ P @ M.transpose(-2, -1) + torch.einsum('btmn,nk,btmk->btmm', K_t, self.C_w, K_t)
        # Joseph 形式：P_post = (I - K H) P (I - K H)^T + K R_0 K^T
        MP     = torch.matmul(M, P)                                # [B, T, m, m]
        MPMT   = torch.matmul(MP, M.transpose(-2, -1))             # [B, T, m, m]
        KR     = torch.matmul(K_t, self.C_w)                       # [B, T, m, n]
        KRKt   = torch.matmul(KR, K_t.transpose(-2, -1))           # [B, T, m, m]
        P_post = MPMT + KRKt


        # [MOD] 若你先用简式，可保留如下行；若改 Joseph，上面两行替换此行：
        # P_post = torch.matmul(I - KH, P)                                     # [B, T, m, m]

        self.mu_xt_yt_current = mu_post
        self.L_xt_yt_current  = P_post
        return self.mu_xt_yt_current, self.L_xt_yt_current
    '''

    # ====== [修改] 采用 m=states, n=obs 的维度约定_End ======

    def compute_logpdf_Gaussian(self, Y):
        """
        观测的对数概率密度（多元高斯）

        Args:
            Y (torch.Tensor): 输入观测 [batch, T, n_obs]

        Returns:
            logprob (torch.Tensor): 对数似然 [batch]

        Math Notes:
            logp = -0.5*n_obs*T*log(2π) - 0.5*log|L_yt| - 0.5*(Y-mu_yt)^T L_yt^{-1} (Y-mu_yt)
        """
        _, T, _ = Y.shape
        logprob = 0.5 * self.n_obs * T * math.log(math.pi*2) - 0.5 * torch.logdet(self.L_yt_current).sum(1) \
            - 0.5 * torch.einsum('nti,nti->nt',
            (Y - self.mu_yt_current),
            torch.einsum('ntij,ntj->nti',torch.inverse(self.L_yt_current), (Y - self.mu_yt_current))).sum(1)
        return logprob

    def compute_predictions(self, Y_test_batch):
        """
        根据观测序列预测隐状态分布

        Args:
            Y_test_batch (torch.Tensor): 输入观测 [batch, T, n_obs]

        Returns:
            mu_xt_yt_prev_test (torch.Tensor): 状态先验均值
            L_xt_yt_prev_test (torch.Tensor): 状态先验方差
            mu_xt_yt_current_test (torch.Tensor): 滤波后状态均值
            L_xt_yt_current_test (torch.Tensor): 滤波后状态方差

        Tensor Dimensions:
            均为 [batch, T, n_states] / [batch, T, n_states, n_states]
        """
        mu_x_given_Y_test_batch, vars_x_given_Y_test_batch = self.rnn.forward(x=Y_test_batch)
        mu_xt_yt_prev_test, L_xt_yt_prev_test = self.compute_prior_mean_vars(
            mu_xt_yt_prev=mu_x_given_Y_test_batch,
            L_xt_yt_prev=vars_x_given_Y_test_batch
        )
        mu_xt_yt_current_test, L_xt_yt_current_test = self.compute_posterior_mean_vars(Yi_batch=Y_test_batch)
        return mu_xt_yt_prev_test, L_xt_yt_prev_test, mu_xt_yt_current_test, L_xt_yt_current_test

    def forward(self, Yi_batch):
      """
      前向流程，输出观测序列的归一化对数似然

      Args:
          Yi_batch (torch.Tensor): 输入观测序列 [batch, T, n_obs]

      Returns:
          log_pYT_batch_avg (torch.Tensor): 平均对数似然标量

      处理流程:
          1) 通过RNN预测状态分布（均值和对角方差），输出 [batch, T, n_states]
          2) 计算状态的先验分布参数
          3) 将状态分布通过观测模型边缘化，得到观测的高斯分布参数（均值和协方差）
          4) 用观测分布计算输入观测序列的对数似然，按序列长度和观测维度归一化
          5) 输出平均对数似然（通常作为损失函数或指标）

      Tensor Dimensions:
          Yi_batch: [B, T, n_obs] 观测序列
          mu_batch: [B, T, n_states] RNN预测的状态均值
          vars_batch: [B, T, n_states] RNN预测的状态方差（对角元素）
          mu_yt_current: [B, T, n_obs] 观测边缘分布均值
          L_yt_current: [B, T, n_obs, n_obs] 观测边缘分布协方差

      Math Notes:
          1) mu_x, vars_x = RNN(Y)       # 状态分布
          2) mu_yt = H @ mu_x + mu_w     # 边缘观测均值
             L_yt  = H @ vars_x @ H.T + C_w   # 边缘观测协方差
          3) log p(Y | model) = log N(Y; mu_yt, L_yt)
          4) 对 batch, T, n_obs 归一化
      """
      # ==========================================================================
      # STEP 01: 用RNN预测每个时间步的状态分布
      # ==========================================================================
      # 输入观测序列Yi_batch，得到状态均值与对角方差（实际shape: [B, T, n_states]）
      mu_batch, vars_batch = self.rnn.forward(x=Yi_batch)

      # ==========================================================================
      # STEP 02: 计算先验参数（对状态均值和协方差做格式转化等）
      # ==========================================================================
      mu_xt_yt_prev, L_xt_yt_prev = self.compute_prior_mean_vars(
          mu_xt_yt_prev=mu_batch,
          L_xt_yt_prev=vars_batch
      )

      # ==========================================================================
      # STEP 03: 边缘化到观测空间，得到观测的均值和协方差
      # ==========================================================================
      self.compute_marginal_mean_vars(
          mu_xt_yt_prev=mu_xt_yt_prev,
          L_xt_yt_prev=L_xt_yt_prev
      )

      # ==========================================================================
      # STEP 04: 计算观测的高斯对数似然（分母归一化）
      # STEP 004&05：
      # (1 / (B * T * D)) * log p(所有 batch 的观测数据)
      #    即：每个观测分量的平均 log-likelihood，
      #    跨 batch 归一化后作为标量输出。
      # ==========================================================================
      logprob_batch = self.compute_logpdf_Gaussian(Y=Yi_batch) / (Yi_batch.shape[1] * Yi_batch.shape[2])

      # ==========================================================================
      # STEP 05: 按 batch 求均值，输出
      # ==========================================================================
      log_pYT_batch_avg = logprob_batch.mean(0)

      return log_pYT_batch_avg


    # def compute_logpdf_Gaussian(self, Y):
    #     """
    #     计算观测数据Y在当前高斯分布（由 mu_yt_current 和 L_yt_current 描述）下的对数概率

    #     Args:
    #         Y (torch.Tensor): 输入观测序列 [batch, T, n_obs]

    #     Returns:
    #         logprob (torch.Tensor): 对数概率 [batch]

    #     Tensor Dimensions:
    #         Y:                [B, T, n_obs]
    #         mu_yt_current:    [B, T, n_obs]
    #         L_yt_current:     [B, T, n_obs, n_obs]

    #     Math Notes:
    #         logp = -0.5 * n_obs * T * log(2π)
    #                -0.5 * sum_t log|L_yt|
    #                -0.5 * sum_t (Y - mu_yt)^T @ L_yt^{-1} @ (Y - mu_yt)
    #     """
    #     # ==========================================================================
    #     # STEP 01: 维度及常数项计算
    #     # ==========================================================================
    #     _, T, _ = Y.shape
    #     n_obs = self.n_obs

    #     # ==========================================================================
    #     # STEP 02: 计算 log |Σ| 和逆协方差
    #     # ==========================================================================
    #     logdet = torch.logdet(self.L_yt_current).sum(1)  # shape: [batch]
    #     L_inv = torch.inverse(self.L_yt_current)         # shape: [batch, T, n_obs, n_obs]

    #     # ==========================================================================
    #     # STEP 03: 计算马氏距离项
    #     # ==========================================================================
    #     diff = (Y - self.mu_yt_current)                  # shape: [batch, T, n_obs]
    #     quad = torch.einsum('nti,ntij,ntj->nt', diff, L_inv, diff).sum(1)  # [batch]

    #     # ==========================================================================
    #     # STEP 04: 合成最终log概率
    #     # ==========================================================================
    #     logprob = (
    #         -0.5 * n_obs * T * math.log(2 * math.pi)
    #         -0.5 * logdet
    #         -0.5 * quad
    #     )
    #     return logprob
# ============================= DANSE 类定义到此为止 ============================
# 请确保class DANSE: ...类体内缩进一致
