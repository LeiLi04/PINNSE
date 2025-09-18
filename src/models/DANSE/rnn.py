"""
==============================================================================
【科研论文 × 开源工程】高质量注释版

1. 项目 / 算法背景 (Project/Algorithm Context)：
   本文件实现基于RNN的非线性状态估计方法 DANSE (Data-driven Non-linear State Estimation)，
   主要应用于时间序列的隐变量建模与贝叶斯状态滤波，结合 Kalman-like 滤波与神经网络表征能力。

2. 输入输出概览 (Inputs/Outputs Overview)：
   - 主要数据流：三维张量输入 Y ∈ [batch_size, T, n_obs]，输出状态估计均值和方差 X_hat ∈ [batch_size, T, n_states]。
   - 网络输出包括状态分布参数（均值/方差）与观测概率对数似然。
   - 关键张量维度均已注释，便于与物理建模和深度学习兼容。
==============================================================================
"""

#####################################################
# Creators: Anubhab Ghosh, Antoine Honoré
# Feb 2023
#####################################################

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim, distributions
from timeit import default_timer as timer
import sys
import copy
import math
import os
# from utils.utils import compute_log_prob_normal, create_diag, compute_inverse, count_params, ConvergenceMonitor
# from utils.plot_functions import plot_state_trajectory, plot_state_trajectory_axes
import torch.nn.functional as F

# ==========================================================================
# STEP 01: RNN 基础模型定义 (Base RNN Model)
# ==========================================================================
class RNN_model(nn.Module):
    """
    Recurrent Neural Network 基础模型（支持 LSTM/GRU/RNN）

    Args:
        input_size (int): 输入数据特征维度。
        output_size (int): 输出数据特征维度。
        n_hidden (int): 隐层单元数。
        n_layers (int): RNN 堆叠层数。
        model_type (str): RNN 类型 ("lstm" / "gru" / "rnn")。
        lr (float): 学习率。
        num_epochs (int): 训练轮数。
        n_hidden_dense (int): 全连接层隐藏单元数，默认32。
        num_directions (int): RNN方向数（单向=1，双向=2）。
        batch_first (bool): 输入是否 batch 作为首维。
        min_delta (float): 收敛容忍度。
        device (str): 设备选择（"cpu" 或 "cuda"）。

    Returns:
        None

    Tensor Dimensions:
        x: [batch_size, T, input_size]
        output: 均值 mu, 方差 vars，均为 [batch_size, T, output_size]

    Math Notes:
        - RNN/LSTM/GRU 负责处理序列建模: h_t = RNN(x_t, h_{t-1})
        - 输出映射：y = ReLU(FC(r_out)), mu = FC_mean(y), vars = softplus(FC_vars(y))
    """
    def __init__(self, input_size, output_size, n_hidden, n_layers,
        model_type, lr, num_epochs, n_hidden_dense=32, num_directions=1, batch_first = True, min_delta=1e-2, device='cpu'):
        super(RNN_model, self).__init__()
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.device=device
        self.num_directions = num_directions
        self.batch_first = batch_first
        # === 新增：让 RNN_model 拥有维度信息（你需要的就是这两句） ===
        self.n_states = output_size     # X 的维度
        self.n_obs    = input_size      # 观测的维度
        # 选择 RNN 类型（LSTM/GRU/RNN）
        if model_type.lower() == "rnn":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim,
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim,
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim,
                num_layers=self.num_layers, batch_first=self.batch_first)
        else:
            print("Model type unknown:", model_type.lower())
            sys.exit()

        # 输出全连接层（映射到下游估计/分布参数）
          # self.hidden_dim * self.num_directions, n_hidden_dense
        '''
        RNN/GRU 输出隐藏状态  →  fc (特征变换)
                        →  fc_mean  (得到均值)
                        →  fc_vars  (得到方差)
        '''
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size).to(self.device)

    def init_h0(self, batch_size):
        """
        [Helper] for forward
        初始化 RNN 隐状态

        Args:
            batch_size (int): 批量大小。

        Returns:
            h0 (torch.Tensor): 初始隐状态 [n_layers, batch_size, hidden_dim]。

        Tensor Dimensions:
            h0: [n_layers, batch_size, hidden_dim]
        """
        # 采用标准正态分布初始化（有利于训练收敛）（num_layers, B, hidden_dim）
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0

    def forward(self, x):
        """
        前向传播，计算序列均值和方差

        Args:
            x (torch.Tensor): 输入序列 [batch_size, T, input_size]

        Returns:
            mu (torch.Tensor): 输出均值 [batch_size, T, output_size]
            vars (torch.Tensor): 输出方差 [batch_size, T, output_size]

        Tensor Dimensions:
            x: [B, T, input_size]
            mu/vars: [B, T, output_size]

        Math Notes:
            - r_out = RNN(x)
            - y = relu(FC(r_out))
            - mu = FC_mean(y)
            - vars = softplus(FC_vars(y)) , log(1+exp(.))
        """
        batch_size = x.shape[0]

        # ======================================================================
        # STEP 01: 计算 RNN 输出 (Sequence Embedding via RNN)
        # ======================================================================
        r_out, _ = self.rnn(x)
        r_out_all_steps = r_out.contiguous().view(batch_size, -1, self.num_directions * self.hidden_dim)

        # ======================================================================
        # STEP 02: 全连接层映射 (Feature Mapping)
        # ======================================================================
        y = F.relu(self.fc(r_out_all_steps))

        # ======================================================================
        # STEP 03: 预测均值和方差 (Mean/Variance Estimation)
        # ======================================================================
        mu_2T_1 = self.fc_mean(y)
        # 这里需要用softplus来保证非负
        vars_2T_1 = F.softplus(self.fc_vars(y)) #(B,T,d)

        # ======================================================================
        # STEP 04: 首时刻初始化拼接 (First-step Initialization & Concatenation)
        # 物理意义：
        #   对应着 x_t = RNN(y_{t-1})
        # ======================================================================
        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:]))).view(batch_size, 1, -1) # (B, 1, d)
        # [-1,:,:] 取最后一层（或最后方向）的初始隐状态，得到 [B, hidden_dim]
        # self.fc(...) → ReLU → fc_mean / fc_vars
        # .view(B,1,-1)， 把 [B, d] reshape 成 [B, 1, d]，表示这是序列的 第一个时间步 的输出
        var_1 = F.softplus(self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:])))).view(batch_size, 1, -1) #(B, 1, d)

        # 拼接首时刻与后续估计，形成完整输出
        mu = torch.cat((mu_1, mu_2T_1[:,:-1,:]), dim=1) #(B,T,d)
        vars = torch.cat((var_1, vars_2T_1[:,:-1,:]), dim=1)
        return mu, vars

def save_model(model, filepath):
    """
    保存模型参数到文件

    Args:
        model (nn.Module): PyTorch 模型对象
        filepath (str): 存储路径

    Returns:
        None
    """
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    """
    将模型推送到指定设备

    Args:
        nets (nn.Module): PyTorch 模型
        device (str): 目标设备

    Returns:
        nets (nn.Module): 转移后的模型
    """
    nets = nets.to(device=device)
    return nets
