# ===== kalman_gain_net.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaKNet(nn.Module):
     """
    输入:
        # [MOD] resid_abs 现在表示“白化后的创新” ν_t = L^{-1} Δy_t，shape [B, n]
        #      （沿用变量名 resid_abs 以保持兼容；你也可以改名为 resid_feat/nu_t）
        resid_abs              shape [B, n]
        sigma_pred_diag        shape [B, m] = diag(Σ_{t|t-1})

    输出:
        delta_K_flat           shape [B, m*n]  (reshape → [B, m, n])
        其中 m = n_state(状态维), n = n_meas(观测维)
    """
     def __init__(self, n_state: int, n_meas: int, hidden: int = 128, alpha: float = 0.1):
        super().__init__()
        self.n_state = n_state
        self.n_meas  = n_meas
        self.alpha   = alpha

        in_dim  = n_state + n_meas
        out_dim = n_state * n_meas

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        # 初始化最后一层为零（起步不干扰线性K）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

     def forward(self, resid_abs: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
        """
        resid_abs:  [B, m]
        sigma_diag: [B, n]
        return   :  delta_K  [B, n, m]  (after reshape)
        """
        x = torch.cat([resid_abs, sigma_diag], dim=-1)       # [B, n+m]
        delta_k_flat = torch.tanh(self.mlp(x)) * self.alpha  # constrain & scale
        return delta_k_flat.view(-1, self.n_state, self.n_meas)