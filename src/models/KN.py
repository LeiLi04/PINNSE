"""KalmanNet-inspired nonlinear state estimator for sensor fusion tasks.

项目背景 (Project Background):
    本模块实现 KalmanNet 风格的神经网络，用于非线性状态估计与滤波，结合
    Recurrent NN 与 Kalman Gain 学习，解决传统滤波器对模型敏感的问题。

输入输出概览 (Input/Output Overview):
    - 训练阶段: 接收观测序列 y ∈ ℝ^[batch_size, T, n_obs]，输出状态估计
      x_hat ∈ ℝ^[batch_size, T, n_states] 并学习动态增益。
    - 推理阶段: `forward` 接收单步观测 y_t ∈ ℝ^[batch_size, n_obs]，返回后验
      状态均值 x_post ∈ ℝ^[batch_size, n_states]。
"""

# Adpoted from: https://github.com/KalmanNet/Unsupervised_EUSIPCO_22
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import gc
import sys


def count_params(model: nn.Module):
    """统计模型参数数量 / Count total and trainable parameters.

    Args:
        model (nn.Module): 待评估的 PyTorch 模型实例，需实现 `.parameters()` 迭代器。

    Returns:
        tuple[int, int]:
            - total_params: 模型全部参数个数。
            - trainable_params: 需要梯度更新的参数个数。

    Tensor Dimensions:
        - 参数向量视作一维数组，总元素个数以 `numel()` 计数。

    Math Notes:
        total_params = Σ param.numel()
        trainable_params = Σ param.numel() for param.requires_grad is True
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class KalmanNetNN(nn.Module):
    """KalmanNet 神经网络骨干 / Neural backbone for adaptive Kalman gain.

    Args:
        n_states (int): 状态维度 m；即系统隐藏变量的维度。
        n_obs (int): 观测维度 n；对应传感器输出长度。
        n_layers (int): GRU 隐层层数，用于建模时间上下文。
        device (str): 计算设备标识，如 'cpu' 或 'cuda:0'。
        rnn_type (str): 日志和模型命名用的标识，保持与其他模块一致。

    Returns:
        KalmanNetNN: 实例化后的模型，可调用 `Build`、`forward` 等方法完成滤波。

    Tensor Dimensions:
        - 输入观测 y_t: [batch_size, n_obs]
        - 状态输出 x_t: [batch_size, n_states]
        - GRU 隐状态 h_t: [n_layers, batch_size, hidden_dim]

    Math Notes:
        模型通过前馈层与 GRU 映射学习 Kalman 增益 K_t，使得
        x_post = x_prior + K_t @ innovation，其中 innovation = y_t - y_hat。
    """

    def __init__(self, n_states, n_obs, n_layers=1, device='cpu', rnn_type='gru'):
        """初始化 KalmanNet 架构 / Build layer dimensions and attributes.

        Args:
            n_states (int): 状态维度 m。
            n_obs (int): 观测维度 n。
            n_layers (int): GRU 堆叠层数，默认 1。
            device (str): 计算设备字符串，例如 'cpu' 或 'cuda:0'。
            rnn_type (str): RNN 结构标签，用于日志命名。

        Returns:
            None: 直接在实例上设置层与形状信息。

        Tensor Dimensions:
            - d_in = n_states + n_obs (Kalman Gain 输入特征长度)
            - hidden_dim = (n_states^2 + n_obs^2) * 10 (GRU 隐状态)
            - d_out = n_obs * n_states (Kalman Gain 输出展平)

        Math Notes:
            h1_knet = (n_states + n_obs) * 80  # 扩大表示容量
            h2_knet = (n_states + n_obs) * 10
        """

        # ==================================================================
        # STEP 01: 基础超参数与设备设置 (Hyper-parameter setup)
        # ------------------------------------------------------------------
        super(KalmanNetNN, self).__init__()

        self.n_states = n_states # Setting the number of states of the KalmanNet model
        self.n_obs = n_obs # Setting the number of observations of the KalmanNet model
        self.device = device # Setting the device
        self.rnn_type = rnn_type  # For logging/checkpoint naming alignment

        # ==================================================================
        # STEP 02: 计算网络层尺寸 (Derive layer dimensions)
        # ------------------------------------------------------------------
        # Setting the no. of neurons in hidden layers
        self.h1_knet = (self.n_states +  self.n_obs) * (10) * 8
        self.h2_knet = (self.n_states + self.n_obs) * (10) * 1
        self.d_in = self.n_states + self.n_obs # Input vector dimension for KNet
        self.d_out = int(self.n_obs * self.n_states) # Output vector dimension for KNet

        # ==================================================================
        # STEP 03: GRU 维度配置 (GRU configuration)
        # ------------------------------------------------------------------
        # Setting the GRU specific nets
        self.input_dim = self.h1_knet # Input Dimension for the RNN
        self.hidden_dim = (self.n_states ** 2 + self.n_obs **2) * 10 # Hidden Dimension of the RNN
        self.n_layers = n_layers # Number of Layers in the GRU
        self.batch_size = 1 # Batch Size in the GRU
        self.seq_len_input = 1 # Input Sequence Length for the GRU
        self.seq_len_hidden = self.n_layers  # Hidden Sequence Length for the GRU (initilaized as the number of layers)

        # batch_first = False
        # dropout = 0.1 ;
        # STEP 04: 预留返回 (No return, stateful init)
        # ------------------------------------------------------------------
        return None

    def Build(self, f, h):
        """构建 KalmanNet 所需的层与系统函数 / Build neural modules.

        Args:
            f (Callable): 状态转移函数，输入 x ∈ ℝ^[n_states, 1]，输出同形状。
            h (Callable): 观测映射函数，输入 x_prior，同样输出 ℝ^[n_obs, 1]。

        Returns:
            None: 内部保存函数与神经网络层实例。

        Tensor Dimensions:
            - f(x): [n_states, 1]
            - h(x): [n_obs, 1]
            - KGain 输入向量: [batch_size, d_in]

        Math Notes:
            innovation = y_t - y_hat
            gain_vector = GRU(ReLU(Linear(innovation, delta_x)))
        """

        # ==================================================================
        # STEP 01: 包装系统函数为张量输出 (Wrap system dynamics)
        # ------------------------------------------------------------------

        # Initialize the dynamics of the underlying ssm (equivalent of InitSystemDynamics(self, F, H) in original code)
        # self.f_k = f
        # self.h_k = h
        # 统一把 f/h 的输出转成 torch.Tensor（float32 + 正确 device）
        def _ensure_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(torch.device("cpu"), dtype=torch.float32)
            return torch.as_tensor(x, dtype=torch.float32)

        def _ensure_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        def _to_tensor(z):
            if isinstance(z, torch.Tensor):
                return z.to(self.device, dtype=torch.float32)
            return torch.as_tensor(z, dtype=torch.float32, device=self.device)

        # 保存包装后的函数
        self.f_k = lambda x: _to_tensor(f(_ensure_tensor(x)))
        self.h_k = lambda x: _to_tensor(h(_ensure_numpy(x)))

        # ==================================================================
        # STEP 02: 初始化 Kalman Gain 网络层 (Build Kalman Gain network)
        # ------------------------------------------------------------------
        ##############################################
        # Initializing the Kalman Gain network
        # This network is: FC + RNN (e.g. GRU) + FC
        ##############################################
        # Input features:
        # 1. Innovation at time t:
        #    δy_t = y_t - ŷ_{t|t-1}, shape: (self.n_obs, 1)
        # 2. Forward update difference:
        #    δx_{t-1} = ŷ_{t-1|t-1} - ŷ_{t-1|t-2}, shape: (self.n_states, 1)
        #
        # Output:
        # Kalman Gain at time t:
        # K_t, shape: (self.n_states, self.n_obs)

        #############################
        ### Input Layer of KNet ###
        #############################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(self.d_in, self.h1_knet, bias=True).to(self.device, non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###################################
        ### RNN Network inside KNet ###
        ###################################
        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers,batch_first= True).to(self.device,non_blocking = True)

        #####################################
        ### Penultimate Layer of KNet ###
        #####################################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, self.h2_knet, bias=True).to(self.device,non_blocking = True)
        self.KG_relu2 = torch.nn.ReLU() # ReLU (Rectified Linear Unit) Activation Function

        #####################################
        ### Output Layer of KNet ###
        #####################################
        self.KG_l3 = torch.nn.Linear(self.h2_knet, self.d_out, bias=True).to(self.device,non_blocking = True)
        return None

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):
        """初始化滤波循环中的先验 / Initialize prior state sequence.

        Args:
            M1_0 (torch.Tensor): 初始状态均值，形状 [n_states, 1]。

        Returns:
            None: 内部缓存先验与后验均值副本。

        Tensor Dimensions:
            - m1x_prior, m1x_posterior: [n_states, batch_size]

        Math Notes:
            x_prior_{0|−1} = repeat(M1_0, batch_size)
        """

        # ==================================================================
        # STEP 01: 适配批量维度 (Broadcast initial mean)
        # ------------------------------------------------------------------

        # Adjust for batch size
        M1_0 = M1_0.repeat(1, self.batch_size)   # (n_states, batch_size)
        self.m1x_prior = M1_0.detach().to(self.device, non_blocking = True) # Initial value of x_{t|t-1}
        self.m1x_posterior = M1_0.detach().to(self.device, non_blocking = True) # Initial value of x_{t-1|t-1}
        self.state_process_posterior_0 = M1_0.detach().to(self.device, non_blocking = True) # Initial value of x_{t-1|t-1} for state process

    #########################################################
    ### Set Batch Size and initialize hidden state of GRU ###
    #########################################################
    def SetBatch(self,batch_size):
        """设置批量大小并重置 GRU 隐状态 / Configure batch and hidden state.

        Args:
            batch_size (int): 当前 mini-batch 样本数。

        Returns:
            None: 更新内部 `batch_size` 与 `hn`。

        Tensor Dimensions:
            - hn: [n_layers, batch_size, hidden_dim]

        Math Notes:
            hn = N(0, I) 初始化以提供随机起点，避免死区。
        """

        # ==================================================================
        # STEP 01: 更新 batch_size 并重建随机隐状态 (Reset hidden state)
        # ------------------------------------------------------------------
        self.batch_size = batch_size
        self.hn = torch.randn(self.seq_len_hidden,self.batch_size,self.hidden_dim,requires_grad=False).to(self.device)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        """单步先验预测 / Perform prior prediction for state and observation.

        Returns:
            None: 更新 `state_process_prior_0`, `obs_process_0`, `m1x_prior`, `m1y`。

        Tensor Dimensions:
            - state_process_prior_0: [n_states, batch_size]， 表示 真实状态过程的先验 (state prior trajectory)。
            - obs_process_0: [n_obs, batch_size]， 真实观测过程的预测 (observation prior trajectory)。
            - m1x_prior: [n_states, batch_size]，   表示 状态的均值先验 (mean prior of state)。
            - m1y: [n_obs, batch_size]，          表示 观测分布的一阶矩 （均值）

        Math Notes:
            x_prior = f(x_post_prev)
            y_hat = h(x_prior)
        """

        # ==================================================================
        # STEP 01: 生成状态先验 (State prior propagation)
        # ------------------------------------------------------------------
        # Formula: x_{t|t-1} = f(x_{t-1|t-1})
        batch_size = self.state_process_posterior_0.shape[1]
        state_prior_list = []
        obs_prior_list = []
        for i in range(batch_size):
            prior_state_vec = self.f_k(self.state_process_posterior_0[:, i].reshape(-1, 1)).view(-1)
            state_prior_list.append(prior_state_vec)
            obs_prior_vec = self.h_k(prior_state_vec.reshape(-1, 1)).view(-1)
            obs_prior_list.append(obs_prior_vec)

        self.state_process_prior_0 = torch.stack(state_prior_list, dim=1)
        self.obs_process_0 = torch.stack(obs_prior_list, dim=1)

        # ==================================================================
        # STEP 03: 更新状态与观测的均值 (Update means for recursion)
        # ------------------------------------------------------------------
        # Predict the 1-st moment of x
        # Formula: m1x_prior = f(m1x_posterior)
        self.m1x_prev_prior = self.m1x_prior
        m1x_prior_list = []
        m1y_list = []
        for i in range(self.m1x_posterior.shape[1]):
            m1x_prior_vec = self.f_k(self.m1x_posterior[:, i].reshape(-1, 1)).view(-1)
            m1x_prior_list.append(m1x_prior_vec)
            m1y_vec = self.h_k(m1x_prior_vec.reshape(-1, 1)).view(-1)
            m1y_list.append(m1y_vec)

        self.m1x_prior = torch.stack(m1x_prior_list, dim=1)
        self.m1y = torch.stack(m1y_list, dim=1)

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        """执行 KalmanNet 单步更新 / Perform one KalmanNet update.

        Args:
            y (torch.Tensor): 当前观测，形状 [n_obs, batch_size]。

        Returns:
            torch.Tensor: 后验状态均值 `m1x_posterior`，形状 [batch_size, n_states] 展平。

        Tensor Dimensions:
            - y: [n_obs, batch_size]
            - KGain: [batch_size, n_states, n_obs]
            - INOV: [n_states, batch_size]

        Math Notes:
            dy = y - y_hat
            x_post = x_prior + K_t @ dy
        """

        # ==================================================================
        # STEP 01: 预测先验 (Predict prior)
        # ------------------------------------------------------------------

        self.step_prior() # Compute Priors

        # ==================================================================
        # *STEP 02: 估计 Kalman 增益 (Estimate Kalman gain)
        # ------------------------------------------------------------------
        self.step_KGain_est(y)  # Compute Kalman Gain

        # ==================================================================
        # STEP 03: 计算创新项并归一化 (Compute innovation)
        # ------------------------------------------------------------------
        dy = y - self.m1y # Compute the innovation

        #NOTE: My own change!!
        dy = func.normalize(dy, p=2, dim=0, eps=1e-12, out=None) # Extra normalization

        # ==================================================================
        # STEP 04: 更新后验 (Posterior update)
        # ------------------------------------------------------------------
        # Compute the 1-st posterior moment
        # Initialize array of Innovations
        INOV = torch.empty((self.n_states, self.batch_size),device= self.device)

        for batch in range(self.batch_size):
            # Calculate the Inovation for each KGain
            #print("batch: {}, KG norm: {}, dy norm: {}".format(batch+1, torch.norm(self.KGain[batch]).detach().cpu(), torch.norm(dy[:,batch]).detach().cpu()))
            INOV[:,batch] = torch.matmul(self.KGain[batch],dy[:,batch]).squeeze()
            assert torch.isnan(self.KGain[batch].detach().cpu()).any() == False, "NaNs in KG computation"
            assert torch.isnan(dy[:,batch].detach().cpu()).any() == False, "NaNs in innovation diff."

        self.m1x_posterior = self.m1x_prior + INOV

        del INOV,dy,y

        return torch.squeeze(self.m1x_posterior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        """估计 Kalman 增益向量 / Estimate Kalman gain via neural network.

        Args:
            y (torch.Tensor): 当前观测 [n_obs, batch_size]。

        Returns:
            None: 更新 `KGain` 属性。

        Tensor Dimensions:
            - dm1x_norm: [n_states]
            - dm1y_norm: [n_obs]
            - KGainNet_in: [d_in]
            - KGain: [batch_size, n_states, n_obs]

        Math Notes:
            dm1x = x_post_prev - x_prior_prev
            dm1y = y - y_hat
            KGain = reshape(NeuralNet([dm1y_norm, dm1x_norm]))
        """

        # ==================================================================
        # STEP 01: 构建状态差异特征 (State residual features)
        # ------------------------------------------------------------------

        # Reshape and Normalize the difference in X prior
        # dm1x = self.m1x_prior - self.state_process_prior_0
        # (this is equivalent to Forward update difference: 
        # Forward update difference: 
        # δx_{t-1} = x̂_{t-1|t-1} - x̂_{t-1|t-2}, shape = (n_states, 1)
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # ==================================================================
        # STEP 02: 构建观测差异特征 (Observation residual)
        # ------------------------------------------------------------------
        # Normalize y
        # (this is equivalent to Innovation at time t: \delta y_t = y_t - \hat{y}_{t \vert t-1} (self.n_obs, 1))
        dm1y = y - torch.squeeze(self.m1y) #y.squeeze() - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)   # dm1x_norm = dm1x / ( ||dm1x||_2 + eps )

        # ==================================================================
        # STEP 03: 拼接输入并运行网络 (Run gain network)
        # ------------------------------------------------------------------
        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in.T)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.n_states, self.n_obs))
        del KG,KGainNet_in,dm1y,dm1x,dm1y_norm,dm1x_norm,dm1x_reshape


    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):
        """前向传播 Kalman 增益网络 / Forward pass through gain estimator.

        Args:
            KGainNet_in (torch.Tensor): 拼接后的特征，形状 [batch_size, d_in]。

        Returns:
            torch.Tensor: 展平的 Kalman 增益向量，形状 [batch_size, d_out]。

        Tensor Dimensions:
            - L1_out: [batch_size, h1_knet]
            - GRU_out: [batch_size, seq_len_input, hidden_dim]
            - L3_out: [batch_size, d_out]

        Math Notes:
            gain_vec = Linear2(ReLU2(GRU(ReLU1(Linear1(KGainNet_in)))))
        """

        # ==================================================================
        # STEP 01: 前馈输入层 (Feed-forward input layer)
        # ------------------------------------------------------------------

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)
        assert torch.isnan(La1_out).any() == False, "NaNs in La1_out computation"

        # ==================================================================
        # STEP 02: GRU 序列建模 (Temporal modeling)
        # ------------------------------------------------------------------
        ###########
        ### GRU ###
        ###########
        GRU_in = La1_out.reshape((self.batch_size,self.seq_len_input,self.input_dim))
        # 建议：每步都 detach 隐状态，截断反传，避免图在时间维滚雪球
        self.hn = self.hn.detach()
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (self.batch_size, self.hidden_dim))
        assert torch.isnan(GRU_out_reshape).any() == False, "NaNs in GRU_output computation"

        # ==================================================================
        # STEP 03: 隐层与输出层 (Hidden and output layers)
        # ------------------------------------------------------------------
        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)
        assert torch.isnan(La2_out).any() == False, "NaNs in La2_out computation"

        ####################
        ### Output Layer ###
        ####################
        self.L3_out = self.KG_l3(La2_out)
        assert torch.isnan(self.L3_out).any() == False, "NaNs in L3_out computation"

        del L2_out,La2_out,GRU_out,GRU_in,GRU_out_reshape,L1_out,La1_out
        return self.L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        """模型前向接口 / Forward entry point matching nn.Module API.

        Args:
            yt (torch.Tensor): 当前时间步观测，形状 [batch_size, n_obs]。

        Returns:
            torch.Tensor: 后验状态估计，形状 [batch_size, n_states] 展平。

        Tensor Dimensions:
            - yt.T: [n_obs, batch_size]
            - 返回值 squeeze 后 [n_states]

        Math Notes:
            通过 `KNet_step` 完成完整的预测-更新循环。
        """

        # ==================================================================
        # STEP 01: 转置观测并执行单步更新 (Process observation)
        # ------------------------------------------------------------------
        yt = yt.T.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        """初始化 GRU 隐状态张量 / Reset hidden state buffer.

        Returns:
            None: 更新 `hn` 属性。

        Tensor Dimensions:
            - hidden: [n_layers, batch_size, hidden_dim]

        Math Notes:
            使用零初始化，保持与标准 GRU 一致，避免引入额外偏差。
        """

        # ==================================================================
        # STEP 01: 基于当前参数形状生成零张量 (Zero initialization)
        # ------------------------------------------------------------------
        weight = next(self.parameters()).data
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim,
                     device=weight.device, dtype=weight.dtype)
        self.hn = hidden.data

    def compute_predictions(self, y_test_batch):
        """批量推理生成状态估计 / Compute predictions over a sequence.

        Args:
            y_test_batch (torch.Tensor): 测试观测，形状 [N_T, T, n_obs]。

        Returns:
            torch.Tensor: 状态估计时间序列，形状 [N_T, n_states, T]。

        Tensor Dimensions:
            - 输入 y_test_batch: [batch, time, obs]
            - x_out_test: [batch, n_states, time]
            - y_out_test: [batch, n_obs, time]

        Math Notes:
            对每个时间步调用 forward，实现循环滤波。
        """

        # ==================================================================
        # STEP 01: 数据整理与缓存初始化 (Prepare tensors)
        # ------------------------------------------------------------------

        test_input = y_test_batch.to(self.device)
        N_T, Ty, dy = test_input.shape

        self.SetBatch(N_T)
        self.InitSequence(torch.zeros(self.ssModel.n_states, 1))

        x_out_test = torch.empty(N_T, self.n_states, Ty, device=self.device)
        y_out_test = torch.empty(N_T, self.n_obs, Ty, device=self.device)

        test_input = torch.transpose(test_input, 1, 2).type(torch.FloatTensor)

        # ==================================================================
        # STEP 02: 循环执行 KalmanNet 更新 (Iterative inference)
        # ------------------------------------------------------------------
        for t in range(0, Ty):
            x_out_test[:,:, t] = self.forward(test_input[:,:, t]).T
            y_out_test[:,:, t] = self.m1y.T

        # ==================================================================
        # STEP 03: 返回状态轨迹 (Return estimates)
        # ------------------------------------------------------------------
        return x_out_test

def train_KalmanNetNN(model, options, train_loader, val_loader, nepochs,
                    logfile_path, modelfile_path, save_chkpoints, device='cpu',
                    tr_verbose=False, unsupervised=True):
    """训练 KalmanNet 模型 / Train KalmanNet neural estimator.

    Args:
        model (KalmanNetNN): 待训练的 KalmanNet 模型实例。
        options (dict): 训练超参数字典，需包含 `lr`, `weight_decay` 等键。
        train_loader (DataLoader): 训练数据迭代器，输出 (y_batch, x_batch)。
        val_loader (DataLoader): 验证数据迭代器，格式同训练。
        nepochs (int): 训练迭代轮数。
        logfile_path (str or None): 日志文件路径；None 时使用默认位置。
        modelfile_path (str or None): 模型保存目录；None 时默认 `./models/`。
        save_chkpoints (str or None): 'all'/'some'/None 定义保存策略。
        device (str): 计算设备。
        tr_verbose (bool): 是否输出详细训练日志。
        unsupervised (bool): True 时使用观测重建损失。

    Returns:
        tuple[np.ndarray, np.ndarray, KalmanNetNN]:
            - MSE_train_dB_epoch_obs: 训练集观测损失 (dB) 数组。
            - MSE_cv_dB_epoch_obs: 验证集观测损失 (dB) 数组。
            - model: 训练后的模型实例。

    Tensor Dimensions:
        - y_batch: [batch, T, n_obs]
        - train_target: [batch, T, n_states]
        - x_out_training: [batch, n_states, T]

    Math Notes:
        loss_state = MSE(x_hat, x_gt)
        loss_obs = MSE(y_hat, y)
        优化目标根据 `unsupervised` 选择上述之一。
    """

    # ==========================================================================
    # STEP 01: 模型推送与日志初始化 (Model to device & logging setup)
    # --------------------------------------------------------------------------

    # Set pipeline parameters: setting ssModel and model
    # 1. Set the KalmanNet model and push to its device
    model = model.to(device)

    # 2. Set the training parameters
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=options["lr"],
                                 weight_decay=options["weight_decay"])

    loss_fn = nn.MSELoss(reduction='mean')
    MSE_cv_linear_epoch = np.empty([nepochs])
    MSE_cv_dB_epoch = np.empty([nepochs])

    MSE_train_linear_epoch = np.empty([nepochs])
    MSE_train_dB_epoch = np.empty([nepochs])

    MSE_cv_linear_epoch_obs = np.empty([nepochs])
    MSE_cv_dB_epoch_obs = np.empty([nepochs])

    MSE_train_linear_epoch_obs = np.empty([nepochs])
    MSE_train_dB_epoch_obs = np.empty([nepochs])

    # 3. Start the training and keep time statistics
    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    if save_chkpoints == "all" or save_chkpoints == "some":
        if logfile_path is None:
            training_logfile = "./log/knet_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path
    else:
        if logfile_path is None:
            training_logfile = "./log/gs_training_knet_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    total_num_params, total_num_trainable_params = count_params(model)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # ==========================================================================
    # STEP 02: 逐 epoch 训练与验证 (Epoch loop)
    # --------------------------------------------------------------------------
    for ti in range(0, nepochs):

        #################################
        ### Validation Sequence Batch ###
        #################################
        # Cross Validation Mode
        model.eval()

        # ------------------------------------------------------------------
        # step 2a) 验证集前向推理与损失统计
        # ------------------------------------------------------------------
        with torch.no_grad():
          # Load obserations and targets from CV data
          y_cv, cv_target = next(iter(val_loader))
          N_CV, Ty, dy = y_cv.shape
          #_, _, dx = cv_target.shape
          y_cv = torch.transpose(y_cv, 1, 2).type(torch.FloatTensor).to(model.device)
          cv_target = torch.transpose(cv_target, 1, 2).type(torch.FloatTensor).to(model.device)

          model.SetBatch(N_CV)
          model.InitSequence(torch.zeros(model.n_states,1))

          x_out_cv = torch.empty(N_CV, model.ssModel.n_states, Ty, device= device).to(model.device)
          y_out_cv = torch.empty(N_CV, model.ssModel.n_obs, Ty, device= device).to(model.device)

          for t in range(0, Ty):
              #print("Time instant t:{}".format(t+1))
              x_out_cv[:,:, t] = model(y_cv[:,:, t]).T
              y_out_cv[:,:,t] = model.m1y.squeeze().T   #观测的预测均值（滤波器内部 m1y）

        # Compute Training Loss
        cv_loss = loss_fn(x_out_cv[:,:,:Ty], cv_target[:,:,1:]).item()
        cv_loss_obs =  loss_fn(y_out_cv[:,:,:Ty], y_cv[:,:,:Ty]).item()

        # Average
        MSE_cv_linear_epoch[ti] = np.mean(cv_loss)
        MSE_cv_dB_epoch[ti] = 10 * np.log10(MSE_cv_linear_epoch[ti])

        MSE_cv_linear_epoch_obs[ti] = np.mean(cv_loss_obs)
        MSE_cv_dB_epoch_obs[ti] = 10*np.log10(MSE_cv_linear_epoch_obs[ti])

        relevant_loss = cv_loss_obs if unsupervised else cv_loss
        relevant_loss = 10 * np.log10(relevant_loss)

        if (relevant_loss < MSE_cv_dB_opt):
            MSE_cv_dB_opt = relevant_loss
            MSE_cv_idx_opt = ti
            print("Saving model ...")
            torch.save(model.state_dict(), os.path.join(model_filepath, "knet_ckpt_epoch_best.pt"))

        ###############################
        ### Training Sequence Batch ###
        ###############################

        # Training Mode
        model.train()

        # Init Hidden State
        model.init_hidden()

        Batch_Optimizing_LOSS_sum = 0

        # ------------------------------------------------------------------
        # step 2b) 取训练 batch 并执行前向-反向
        # ------------------------------------------------------------------
        # Load random batch sized data, creating new iter ensures the data is shuffled
        y_training, train_target = next(iter(train_loader))
        N_E, Ty, dy = y_training.shape
        y_training = torch.transpose(y_training, 1, 2).type(torch.FloatTensor).to(model.device)
        train_target = torch.transpose(train_target, 1, 2).type(torch.FloatTensor).to(model.device)

        model.SetBatch(N_E)
        model.InitSequence(torch.zeros(model.n_states,1))

        x_out_list = []
        y_out_list = []

        for t in range(0, Ty):
            state_pred = model(y_training[:, :, t]).T  # [N_E, n_states]

            obs_pred = model.m1y
            if obs_pred.dim() == 3:
                obs_pred = obs_pred.squeeze(-1)
            obs_pred = obs_pred.transpose(0, 1).contiguous()  # [N_E, n_obs]

            x_out_list.append(state_pred.unsqueeze(-1))
            y_out_list.append(obs_pred.unsqueeze(-1))

        x_out_training = torch.cat(x_out_list, dim=-1)  # [N_E, n_states, Ty]
        y_out_training = torch.cat(y_out_list, dim=-1)  # [N_E, n_obs, Ty]


        # 计算用于反传的张量损失
        loss_t  = loss_fn(x_out_training[:,:,:Ty], train_target[:,:,1:])
        loss_obs_t  = loss_fn(y_out_training[:,:,:Ty], y_training[:,:,:Ty])

        # 选择反传的那个（保持是 Tensor）
        LOSS = loss_obs_t if unsupervised else loss_t

        # ------- 下面是“仅用于日志/统计”的数值，务必 .detach().item() -------
        loss_val = loss_t.detach().item()
        loss_obs_val = loss_obs_t.detach().item()

        MSE_train_linear_epoch[ti] = loss_val
        MSE_train_dB_epoch[ti] = 10 * np.log10(MSE_train_linear_epoch[ti])

        MSE_train_linear_epoch_obs[ti] = loss_obs_val
        MSE_train_dB_epoch_obs[ti] = 10 * np.log10(MSE_train_linear_epoch_obs[ti])

        '''
        # Compute Training Loss
        loss  = loss_fn(x_out_training[:,:,:Ty], train_target[:,:,1:])
        loss_obs  = loss_fn(y_out_training[:,:,:Ty], y_training[:,:,:Ty])

        # Select loss, from which to update the gradient
        LOSS = loss_obs if unsupervised else loss

        # Average
        MSE_train_linear_epoch[ti] = loss
        MSE_train_dB_epoch[ti] = 10 * np.log10(MSE_train_linear_epoch[ti])

        MSE_train_linear_epoch_obs[ti] = loss_obs
        MSE_train_dB_epoch_obs[ti] = 10*np.log10(MSE_train_linear_epoch_obs[ti])
        '''
        ##################
        ### Optimizing ###
        ##################

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        # Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
        LOSS.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        # ------------------------------------------------------------------
        # step 2c) 记录训练与验证指标 (Logging metrics)
        # ------------------------------------------------------------------
        ########################
        ### Training Summary ###
        ########################
        train_print = MSE_train_dB_epoch_obs[ti] if unsupervised else MSE_train_dB_epoch[ti]
        cv_print = MSE_cv_dB_epoch_obs[ti] if unsupervised else MSE_cv_dB_epoch[ti]
        print(ti, "MSE Training :", train_print, "[dB]", "MSE Validation :", cv_print,"[dB]")
        print(ti, "MSE Training :", train_print, "[dB]", "MSE Validation :", cv_print,"[dB]", file=orig_stdout)

        if (ti > 1):
            d_train = MSE_train_dB_epoch_obs[ti] - MSE_train_dB_epoch_obs[ti - 1] if unsupervised \
                    else MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]


            d_cv = MSE_cv_dB_epoch_obs[ti] - MSE_cv_dB_epoch_obs[ti - 1] if unsupervised \
                    else MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]

            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]", file=orig_stdout)


        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")
        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]", file=orig_stdout)

        # reset hidden state gradient
        model.hn.detach_()

        # Reset the optimizer for faster convergence
        if ti % 50 == 0 and ti != 0:
            optimizer = torch.optim.Adam(model.parameters(),
                                 lr=options["lr"],
                                 weight_decay=options["weight_decay"])
            print('Optimizer has been reset')
            print('Optimizer has been reset', file=orig_stdout)

    sys.stdout = orig_stdout
    return MSE_train_dB_epoch_obs, MSE_cv_dB_epoch_obs, model

def test_KalmanNetNN(model_test, test_loader, options, device, model_file=None, test_logfile_path = None):
    """评估 KalmanNet 模型性能 / Evaluate trained KalmanNet.

    Args:
        model_test (KalmanNetNN): 已训练模型实例。
        test_loader (DataLoader): 测试数据迭代器。
        options (dict): 含 `N_T` 等评估超参数。
        device (str): 计算设备。
        model_file (str or None): 权重文件路径。
        test_logfile_path (str or None): 测试日志输出路径。

    Returns:
        tuple[float, float, torch.Tensor]:
            - MSE_test_dB_avg: 状态估计 MSE (dB)。
            - MSE_test_dB_avg_obs: 观测重建 MSE (dB)。
            - x_out_test: 状态估计时间序列 [N_T, n_states, T]。

    Tensor Dimensions:
        - test_input: [batch, T, n_obs]
        - x_out_test: [batch, n_states, T]

    Math Notes:
        loss_state = mean((x_hat - x)^2)
        loss_obs = mean((y_hat - y)^2)
    """

    # ==========================================================================
    # STEP 01: 加载参数并准备张量 (Load weights & tensors)
    # --------------------------------------------------------------------------

    with torch.no_grad():

        N_T = options["N_T"]
        # Load test data and create iterator
        test_data_iter = iter(test_loader)

        # Allocate Array
        MSE_test_linear_arr = torch.empty([N_T],device = device)
        MSE_test_linear_arr_obs = torch.empty([N_T],device= device)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='none')
        # Set model in evaluation mode
        model_test.load_state_dict(torch.load(model_file))
        model_test = model_test.to(device)
        model_test.eval()

        # Load training data from iter
        test_input,test_target = next(test_data_iter)
        test_input = torch.transpose(test_input, 1, 2)
        test_target = torch.transpose(test_target, 1, 2)
        test_target = test_target.to(device)
        test_input = test_input.to(device)
        _, Ty, dy = test_input.shape

        model_test.SetBatch(N_T)
        model_test.InitSequence(model_test.ssModel.m1x_0)

        if not test_logfile_path is None:
            test_log = "./log/test_knet.log"
        else:
            test_log = test_logfile_path

        x_out_test = torch.empty(N_T, model_test.n_states, Ty, device=device)
        y_out_test = torch.empty(N_T, model_test.n_obs, Ty, device=device)

        # ==========================================================================
        # STEP 02: 序列推理与损失累计 (Iterate over timeline)
        # --------------------------------------------------------------------------
        for t in range(0, Ty):
            x_out_test[:,:, t] = model_test(test_input[:,:, t]).T
            y_out_test[:,:, t] = model_test.m1y.T

        loss_unreduced = loss_fn(x_out_test[:,:,:Ty],test_target[:,:,:Ty])
        loss_unreduced_obs = loss_fn(y_out_test[:,:,:Ty],test_input[:,:,:Ty])

        # Create the linear loss from the total loss for the batch
        loss = torch.mean(loss_unreduced,axis = (1,2))
        loss_obs = torch.mean(loss_unreduced_obs,axis = (1,2))


        MSE_test_linear_arr[:] = loss
        MSE_test_linear_arr_obs[:] = loss_obs

        # Average
        MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
        MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg).item()

        MSE_test_linear_avg_obs = torch.mean(MSE_test_linear_arr_obs)
        MSE_test_dB_avg_obs = 10 * torch.log10(MSE_test_linear_avg_obs).item()

        # ==========================================================================
        # STEP 03: 写入日志并返回 (Log and return)
        # --------------------------------------------------------------------------
        with open(test_log, "a") as logfile_test:
            logfile_test.write('Test MSE loss: {:.3f}, Test MSE loss obs: {:.3f} using weights from file: {}'.format(MSE_test_dB_avg, MSE_test_dB_avg_obs, model_file))
        # Print MSE Cross Validation
        #str = self.modelName + "-" + "MSE Test:"
        #print(str, self.MSE_test_dB_avg, "[dB]")

    return MSE_test_dB_avg, MSE_test_dB_avg_obs, x_out_test
