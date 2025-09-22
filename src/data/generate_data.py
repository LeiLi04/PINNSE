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
    - 生成并保存标准化的数据集文件（PKL），内容为状态/观测对 [T, D] × N_samples

张量维度/数据结构说明:
    - 每条数据 [X, Y]: [T, n_states], [T, n_obs]
    - 数据集内的 Z_XY["data"]: [N_samples, 2] (object数组, 每行为[X, Y])

典型应用场景:
    - 状态估计算法、深度序列模型、RNN训练及基准实验
====================================================================
"""

# ==========================================================================
# STEP 01: 导入依赖
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

from .state_space_model.LinearSSM import LinearSSM
from .state_space_model.LorenzAttractorModel import LorenzAttractorModel
from .state_space_model.SinusoidalSSM import SinusoidalSSM

try:
    from ..parameters import get_parameters
except ImportError:
    SRC_DIR = Path(__file__).resolve().parents[1]
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from parameters import get_parameters

SUPPORTED_DATASET_TYPES = ("LinearSSM", "LorenzSSM", "SinusoidalSSM")


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


#+++++++++++++++++修改开始+++++++++++++++++++++
def generate_state_observation_pairs(type_, parameters, T=200, N_samples=1000):
    """
    批量生成 N_samples 对 (X, Y) 状态/观测对。
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
    # STEP 02: 打包成object 数组, 兼容不同形状
    # ==========================================================================
    Z_XY["data"] = np.array(Z_XY_data, dtype=object)  # 每行 [Xi, Yi]
    Z_XY["trajectory_lengths"] = np.array(Z_XY_data_lengths)    # 每条序列长度 T
    return Z_XY


def create_filename(T, N_samples, m, n, type_, inverse_r2_dB, nu_dB, dataset_basepath = "src/data/trajectories"):
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


def create_and_save_dataset(T, N_samples, filename, parameters, type_="LorenzSSM"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create datasets by simulating state space models."
    )
    '''
             class Args:
                 n_states = 3
                 n_obs = 3
                 num_samples = 100
                 sequence_length = 1000
                  | (1/r^2) [dB] | r^2 [dB] | r^2 (线性) | ν [dB] |   ν (线性)   |
                  |-------------:|---------:|-----------:|-------:|-------------:|
                  |          -20 |      +20 |       100  |   -30  |   0.001      |
                  |          -10 |      +10 |        10  |   -20  |   0.01       |
                  |           -5 |       +5 |     3.1623 |   -15  |   0.0316     |
                  |            0 |        0 |         1  |   -10  |   0.1        |
                  |            5 |       -5 |     0.3162 |    -5  |   0.3162     |
                  |           10 |      -10 |       0.1  |     0  |   1          |
                  |           20 |      -20 |      0.01  |    10  |   10         |
                 inverse_r2_dB = -40
                 nu_dB = -50
                 dataset_type = 'LorenzSSM'
                 # output_path = './data'
                 output_path = './eval_sets/Lorenz_Atractor/T1000_NT100/' + '/'
    '''
    parser.add_argument("--n_states", type=int, default=3, help="Latent state dimension")
    parser.add_argument("--n_obs", type=int, default=3, help="Observation dimension")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of trajectories to simulate")
    parser.add_argument("--sequence_length", type=int, default=200, help="Length of each trajectory")
    parser.add_argument("--inverse_r2_dB", type=float, default=40.0, help="Inverse measurement noise power in dB")
    parser.add_argument("--nu_dB", type=float, default=0.0, help="Process/measurement noise ratio in dB")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="LorenzSSM",
        choices=SUPPORTED_DATASET_TYPES,
        help="Which SSM to simulate",
    )
    parser.add_argument("--output_path", type=str, default="src/data/trajectories", help="Directory to store the generated dataset")
    args = parser.parse_args()

    n_states = args.n_states
    n_obs = args.n_obs
    T = args.sequence_length
    N_samples = args.num_samples
    dataset_type = args.dataset_type
    inverse_r2_dB = args.inverse_r2_dB
    nu_dB = args.nu_dB

    default_base = Path(__file__).resolve().parents[2] / "data"
    base_path = Path(args.output_path) if args.output_path else default_base
    base_path.mkdir(parents=True, exist_ok=True)

    datafilename = create_filename(
        T=T,
        N_samples=N_samples,
        m=n_states,
        n=n_obs,
        type_=dataset_type,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        dataset_basepath=str(base_path),
    )

    ssm_parameters, _ = get_parameters(
        N=N_samples,
        T=T,
        n_states=n_states,
        n_obs=n_obs,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
    )

    if dataset_type not in ssm_parameters:
        available = ", ".join(sorted(ssm_parameters.keys()))
        raise ValueError(f"Unknown dataset_type {dataset_type!r}. Available: {available}")

    if not os.path.isfile(datafilename):
        print(f"Creating the data file: {datafilename}")
        create_and_save_dataset(
            T=T,
            N_samples=N_samples,
            filename=datafilename,
            parameters=ssm_parameters[dataset_type],
            type_=dataset_type,
        )
    else:
        print(f"Dataset {datafilename} is already present!")

    print("Done...")
