"""
====================================================================
项目 / 算法背景 (Background)
--------------------------------------------------------------------
本文件主要实现多种滤波器（如 KF/EKF/UKF/DANSE/KalmanNet）与真值轨迹的可视化，
支持状态估计轨迹和观测数据的二维/三维绘制，便于直观比较滤波算法性能，常用于动力系统/状态空间模型研究与论文实验复现。

输入输出概览 (Input/Output Overview)
--------------------------------------------------------------------
输入:
    - X, Y: 系统状态或观测数据 (numpy.ndarray)，形状 [T, D]，T为时间步数，D为维度数(2或3)
    - X_est_*, Y_est: 各滤波方法输出的状态/观测估计 (numpy.ndarray, shape 同上)
    - savefig, savefig_name: 控制是否保存图片及文件名 (bool, str)

输出:
    - 图形显示或保存至本地，支持 tikzplotlib 导出 LaTeX 绘图代码（可选）
    - 无显式返回值（均为 None）

张量维度/数据结构说明:
    - X, Y, X_est_*, Y_est: numpy.ndarray, 形状 [T, D]
      - T: 时间步数（time steps），D: 变量维数（2或3）
    - 适用 2D/3D 状态与观测轨迹批量可视化

典型应用场景:
    - 状态估计算法对比、仿真轨迹绘制、论文图表生成
====================================================================
"""

# ### CHANGED: 移除无用/误导性导入
# from cProfile import label
# from turtle import color

import os
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # ### CHANGED: 对现代 Matplotlib 非必需，可移除

# ### NEW: 将 tikzplotlib 设为可选依赖，避免 ImportError 中断
_TIKZ_AVAILABLE = True
try:
    import tikzplotlib  # type: ignore
except Exception as _e:  # 包含 ImportError 及兼容性错误
    _TIKZ_AVAILABLE = False
    _TIKZ_ERRMSG = str(_e)

# ### NEW: 统一的保存与可选 tikz 导出辅助函数
def _maybe_save(fig, savefig: bool, savefig_name: str | None):
    if not savefig:
        return
    # 自动生成文件名
    if not savefig_name:
        savefig_name = "figure.png"  # 也可按需改成时间戳
    # 保存图片（建议在 tight_layout 后）
    fig.savefig(savefig_name, dpi=300, bbox_inches="tight")
    # 可选导出 tikz
    if _TIKZ_AVAILABLE:
        try:
            tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
        except Exception as e:
            print(f"[tikzplotlib] 导出失败：{e}")
    else:
        # tikz 不可用时给出一次性提示（不抛异常）
        print("[tikzplotlib] 未启用：当前环境无法导入 tikzplotlib。"
              f"原因：{_TIKZ_ERRMSG if '_TIKZ_ERRMSG' in globals() else '未知'}")

def plot_state_trajectory(
    X,
    X_est_EKF=None,
    X_est_UKF=None,
    X_est_DANSE=None,
    X_est_PINN=None,   # ← 新增，可选
):
    """
    期望每个输入都是 [T, m] 张量；m=3 时画三条状态随时间的曲线。
    只对非 None 的方法画曲线。
    """
    import matplotlib.pyplot as plt

    # 简单示例：m=3，分别画三条
    T, m = X.shape
    t = range(T)

    labels = []
    plt.figure()
    for idx in range(m):
        plt.plot(t, X[:, idx], linestyle='-', label=f'True x{idx+1}')
        if X_est_EKF is not None:   plt.plot(t, X_est_EKF[:, idx], linestyle='--', label=f'EKF x{idx+1}')
        if X_est_UKF is not None:   plt.plot(t, X_est_UKF[:, idx], linestyle='-.', label=f'UKF x{idx+1}')
        if X_est_DANSE is not None: plt.plot(t, X_est_DANSE[:, idx], linestyle=':', label=f'DANSE x{idx+1}')
        if X_est_PINN is not None:  plt.plot(t, X_est_PINN[:, idx], label=f'PINNSE x{idx+1}')

    plt.xlabel('t')
    plt.ylabel('state')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_state_trajectory(
#     X,
#     X_est_KF=None,
#     X_est_EKF=None,
#     X_est_UKF=None,
#     X_est_DANSE=None,
#     X_est_KNET=None,
#     savefig: bool = False,
#     savefig_name: str | None = None
# ):
#     """
#     绘制系统真实状态轨迹与多种估计算法的二维/三维轨迹对比曲线。
#     支持 KF, EKF, UKF, DANSE, KalmanNet 等主流方法。可选择保存为图片与 LaTeX 绘图源码。
#     ---
#     Args:
#         X (np.ndarray): 真实状态轨迹，形状 [T, D]，T为时间步，D为状态维数（2或3）
#         X_est_KF (np.ndarray, optional): KF估计轨迹，形状同 X
#         X_est_EKF (np.ndarray, optional): EKF估计轨迹，形状同 X
#         X_est_UKF (np.ndarray, optional): UKF估计轨迹，形状同 X
#         X_est_DANSE (np.ndarray, optional): DANSE估计轨迹，形状同 X
#         X_est_KNET (np.ndarray, optional): KalmanNet估计轨迹，形状同 X
#         savefig (bool, optional): 是否保存图片（默认为 False）
#         savefig_name (str, optional): 图片保存路径名

#     Returns:
#         None

#     Tensor Dimensions:
#         - X, X_est_KF, ...: [T, D], T=时间步, D=2(二维)或3(三维)

#     Math Notes:
#         - 绘图对比真实轨迹 x_true 与各算法估计 x_hat
#         - 对于每个算法: plot(x_est[:,0], x_est[:,1], ...) 与 x_true 比较

#     设计说明:
#         - 支持自动区分2D与3D轨迹
#         - tikzplotlib 导出用于高质量论文插图（可选）
#     """
#     # ======================================================================
#     # STEP 01: 判断轨迹维度并选择2D或3D绘图
#     # ======================================================================
#     if X.shape[-1] == 2:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         # 真实轨迹
#         ax.plot(X[:, 0], X[:, 1], 'k-', label=r'$\mathbf{x}^{true}$')  # ### CHANGED: raw string
#         # 叠加估计轨迹
#         if X_est_KF is not None:    # ### CHANGED
#             ax.plot(X_est_KF[:, 0], X_est_KF[:, 1], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}$')  # ### CHANGED
#         if X_est_EKF is not None:   # ### CHANGED
#             ax.plot(X_est_EKF[:, 0], X_est_EKF[:, 1], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}$')  # ### CHANGED
#         if X_est_UKF is not None:   # ### CHANGED
#             ax.plot(X_est_UKF[:, 0], X_est_UKF[:, 1], '-.', label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}$')  # ### CHANGED
#         if X_est_DANSE is not None: # ### CHANGED
#             ax.plot(X_est_DANSE[:, 0], X_est_DANSE[:, 1], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}$')  # ### CHANGED
#         if X_est_KNET is not None:  # ### CHANGED
#             ax.plot(X_est_KNET[:, 0], X_est_KNET[:, 1], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}$')  # ### CHANGED
#         ax.set_xlabel(r'$X_1$')  # ### CHANGED: raw string
#         ax.set_ylabel(r'$X_2$')  # ### CHANGED: raw string
#         ax.legend()
#         plt.tight_layout()

#     elif X.shape[-1] > 2:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         # 三维真实轨迹与估计
#         ax.plot(X[:, 0], X[:, 1], X[:, 2], 'k-', label=r'$\mathbf{x}^{true}$')  # ### CHANGED
#         if X_est_KF is not None:    # ### CHANGED
#             ax.plot(X_est_KF[:, 0], X_est_KF[:, 1], X_est_KF[:, 2], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}$')  # ### CHANGED
#         if X_est_EKF is not None:
#             ax.plot(X_est_EKF[:, 0], X_est_EKF[:, 1], X_est_EKF[:, 2], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}$', lw=1.3)  # ### CHANGED
#         if X_est_UKF is not None:
#             ax.plot(X_est_UKF[:, 0], X_est_UKF[:, 1], X_est_UKF[:, 2], 'x-', ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}$', lw=1.3)  # ### CHANGED: 去掉显式 color 以降低风格耦合
#         if X_est_KNET is not None:
#             ax.plot(X_est_KNET[:, 0], X_est_KNET[:, 1], X_est_KNET[:, 2], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}$', lw=1.3)  # ### CHANGED
#         if X_est_DANSE is not None:
#             ax.plot(X_est_DANSE[:, 0], X_est_DANSE[:, 1], X_est_DANSE[:, 2], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}$', lw=1.3)  # ### CHANGED
#         ax.set_xlabel(r'$X_1$')  # ### CHANGED
#         ax.set_ylabel(r'$X_2$')  # ### CHANGED
#         ax.set_zlabel(r'$X_3$')  # ### CHANGED

#         handles, labels = ax.get_legend_handles_labels()
#         order = range(len(handles))
#         ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=5, fontsize=12)
#         plt.tight_layout()

#     # ======================================================================
#     # STEP 02: 保存（如需）
#     # ======================================================================
#     _maybe_save(plt.gcf(), savefig, savefig_name)
#     return None


def plot_measurement_data(Y, savefig: bool = False, savefig_name: str | None = None):
    """
    绘制观测数据的二维/三维轨迹，用于直观展示测量序列。
    ---
    Args:
        Y (np.ndarray): 观测数据，形状 [T, D]，T为时间步，D为观测量维数（2或3）
        savefig (bool, optional): 是否保存图片（默认为 False）
        savefig_name (str, optional): 图片保存路径名

    Returns:
        None
    """
    fig = plt.figure()
    if Y.shape[-1] == 2:
        ax = fig.add_subplot(111)
        ax.plot(Y[:, 0], Y[:, 1], '--', label=r'$\mathbf{y}^{measured}$')  # ### CHANGED
        ax.set_xlabel(r'$Y_1$')  # ### CHANGED
        ax.set_ylabel(r'$Y_2$')  # ### CHANGED
        ax.legend()
        plt.tight_layout()

    elif Y.shape[-1] > 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], '--', label=r'$\mathbf{y}^{measured}$')  # ### CHANGED
        ax.set_xlabel(r'$Y_1$')  # ### CHANGED
        ax.set_ylabel(r'$Y_2$')  # ### CHANGED
        ax.set_zlabel(r'$Y_3$')  # ### CHANGED
        ax.legend()
        plt.tight_layout()

    _maybe_save(fig, savefig, savefig_name)
    return None


# def plot_state_trajectory_axes(
#     X,
#     X_est_KF=None,
#     X_est_EKF=None,
#     X_est_UKF=None,
#     X_est_KNET=None,
#     X_est_DANSE=None,
#     savefig: bool = False,
#     savefig_name: str | None = None
# ):
#     """
#     按分量分别绘制状态真实值与多算法估计的时间序列曲线，支持二维与三维状态向量。
#     """
#     Tx, _ = X.shape
#     T_end = min(200, Tx)  # ### CHANGED: 防越界

#     if X.shape[-1] == 2:
#         fig = plt.figure(figsize=(20, 10))

#         # X1
#         plt.subplot(311)
#         plt.plot(X[:T_end, 0], '--', label=r'$\mathbf{x}^{true}\ (x\text{-}component)$')  # ### CHANGED
#         if X_est_KF is not None:
#             plt.plot(X_est_KF[:T_end, 0], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}\ (x)$')  # ### CHANGED
#         if X_est_DANSE is not None:
#             plt.plot(X_est_DANSE[:T_end, 0], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}\ (x)$')  # ### CHANGED
#         if X_est_KNET is not None:
#             plt.plot(X_est_KNET[:T_end, 0], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}\ (x)$')  # ### CHANGED
#         if X_est_EKF is not None:
#             plt.plot(X_est_EKF[:T_end, 0], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}\ (x)$')  # ### CHANGED
#         if X_est_UKF is not None:
#             plt.plot(X_est_UKF[:T_end, 0], '-x', ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}\ (x)$')  # ### CHANGED
#         plt.ylabel(r'$X_1$')  # ### CHANGED
#         plt.xlabel(r'$t$')    # ### CHANGED
#         plt.legend()

#         # X2
#         plt.subplot(312)
#         plt.plot(X[:T_end, 1], '--', label=r'$\mathbf{x}^{true}\ (y\text{-}component)$')  # ### CHANGED
#         if X_est_DANSE is not None:
#             plt.plot(X_est_DANSE[:T_end, 1], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}\ (y)$')  # ### CHANGED
#         if X_est_KNET is not None:
#             plt.plot(X_est_KNET[:T_end, 1], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}\ (y)$')  # ### CHANGED
#         if X_est_KF is not None:
#             plt.plot(X_est_KF[:T_end, 1], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}\ (y)$')  # ### CHANGED
#         if X_est_EKF is not None:
#             plt.plot(X_est_EKF[:T_end, 1], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}\ (y)$')  # ### CHANGED
#         if X_est_UKF is not None:
#             plt.plot(X_est_UKF[:T_end, 1], 'x-', ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}\ (y)$')  # ### CHANGED
#         plt.ylabel(r'$X_2$')  # ### CHANGED
#         plt.xlabel(r'$t$')    # ### CHANGED
#         plt.legend()

#         plt.tight_layout()

#     elif X.shape[-1] > 2:
#         T_start = 33
#         T_end = min(165, Tx)  # ### CHANGED: 防越界
#         idim = 2   # 只绘制第三维
#         lw = 1.3
#         plt.rcParams['font.size'] = 16
#         fig, ax = plt.subplots(figsize=(9, 5))

#         if X_est_UKF is not None:
#             ax.plot(X_est_UKF[T_start:T_end, idim], 'x-', ms=5, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}$', lw=lw)  # ### CHANGED
#         if X_est_DANSE is not None:
#             ax.plot(X_est_DANSE[T_start:T_end, idim], 'rs-', lw=lw, ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}$')  # ### CHANGED
#         if X_est_KNET is not None:
#             ax.plot(X_est_KNET[T_start:T_end, idim], 'c-.', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}$')  # ### CHANGED
#         if X_est_KF is not None:
#             ax.plot(X_est_KF[T_start:T_end, idim], ':', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}$')  # ### CHANGED
#         if X_est_EKF is not None:
#             ax.plot(X_est_EKF[T_start:T_end, idim], 'b.-', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}$')  # ### CHANGED
#         ax.plot(X[T_start:T_end, idim], 'k-', lw=lw, label=r'$\mathbf{x}^{true}$')  # ### CHANGED

#         ax.set_ylabel(rf'$x_{idim+1}$')  # ### CHANGED: raw string + f-string
#         ax.set_xlabel(r'$t$')            # ### CHANGED

#         handles, labels = ax.get_legend_handles_labels()
#         order = range(len(handles))
#         ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
#                   ncol=5, loc=(-0.02, 1.01), fontsize=16)
#         plt.tight_layout()

#     _maybe_save(plt.gcf(), savefig, savefig_name)
#     return None

def plot_state_trajectory_axes(
    X,
    X_est_KF=None,
    X_est_EKF=None,
    X_est_UKF=None,
    X_est_KNET=None,
    X_est_DANSE=None,
    X_est_PINN=None,               # ### ADDED (PINNSE)
    savefig: bool = False,
    savefig_name: str | None = None
):
    """
    按分量分别绘制状态真实值与多算法估计的时间序列曲线，支持二维与三维状态向量。
    Now also supports PINNSE curves via `X_est_PINN`.  现在新增支持 PINNSE 曲线。

    Args:
        X:               真实状态 [T, m]
        X_est_KF:        KF 估计 [T, m]（可选）
        X_est_EKF:       EKF 估计 [T, m]（可选）
        X_est_UKF:       UKF 估计 [T, m]（可选）
        X_est_KNET:      KalmanNet 估计 [T, m]（可选）
        X_est_DANSE:     DANSE 估计 [T, m]（可选）
        X_est_PINN:      PINNSE 估计 [T, m]（可选）  # ### ADDED (PINNSE)
        savefig:         是否保存图像
        savefig_name:    保存文件名
    """
    Tx, _ = X.shape
    T_end = min(200, Tx)  # ### CHANGED: 防越界

    if X.shape[-1] == 2:
        fig = plt.figure(figsize=(20, 10))

        # X1
        plt.subplot(311)
        plt.plot(X[:T_end, 0], '--', label=r'$\mathbf{x}^{true}\ (x\text{-}component)$')  # ### CHANGED
        if X_est_KF is not None:
            plt.plot(X_est_KF[:T_end, 0], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}\ (x)$')  # ### CHANGED
        if X_est_DANSE is not None:
            plt.plot(X_est_DANSE[:T_end, 0], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}\ (x)$')  # ### CHANGED
        if X_est_KNET is not None:
            plt.plot(X_est_KNET[:T_end, 0], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}\ (x)$')  # ### CHANGED
        if X_est_EKF is not None:
            plt.plot(X_est_EKF[:T_end, 0], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}\ (x)$')  # ### CHANGED
        if X_est_UKF is not None:
            plt.plot(X_est_UKF[:T_end, 0], '-x', ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}\ (x)$')  # ### CHANGED
        if X_est_PINN is not None:  # ### ADDED (PINNSE)
            plt.plot(X_est_PINN[:T_end, 0], 'm^-', label=r'$\hat{\mathbf{x}}_{\mathrm{PINNSE}}\ (x)$')
        plt.ylabel(r'$X_1$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        # X2
        plt.subplot(312)
        plt.plot(X[:T_end, 1], '--', label=r'$\mathbf{x}^{true}\ (y\text{-}component)$')  # ### CHANGED
        if X_est_DANSE is not None:
            plt.plot(X_est_DANSE[:T_end, 1], 'r--', label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}\ (y)$')  # ### CHANGED
        if X_est_KNET is not None:
            plt.plot(X_est_KNET[:T_end, 1], 'c-.', label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}\ (y)$')  # ### CHANGED
        if X_est_KF is not None:
            plt.plot(X_est_KF[:T_end, 1], ':', label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}\ (y)$')  # ### CHANGED
        if X_est_EKF is not None:
            plt.plot(X_est_EKF[:T_end, 1], 'b.-', label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}\ (y)$')  # ### CHANGED
        if X_est_UKF is not None:
            plt.plot(X_est_UKF[:T_end, 1], 'x-', ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}\ (y)$')  # ### CHANGED
        if X_est_PINN is not None:  # ### ADDED (PINNSE)
            plt.plot(X_est_PINN[:T_end, 1], 'm^-', label=r'$\hat{\mathbf{x}}_{\mathrm{PINNSE}}\ (y)$')
        plt.ylabel(r'$X_2$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.tight_layout()

    elif X.shape[-1] > 2:
        T_start = 33
        T_end = min(165, Tx)  # ### CHANGED: 防越界
        idim = 2   # 只绘制第三维
        lw = 1.3
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots(figsize=(9, 5))

        if X_est_UKF is not None:
            ax.plot(X_est_UKF[T_start:T_end, idim], 'x-', ms=5, label=r'$\hat{\mathbf{x}}_{\mathrm{UKF}}$', lw=lw)  # ### CHANGED
        if X_est_DANSE is not None:
            ax.plot(X_est_DANSE[T_start:T_end, idim], 'rs-', lw=lw, ms=4, label=r'$\hat{\mathbf{x}}_{\mathrm{DANSE}}$')  # ### CHANGED
        if X_est_KNET is not None:
            ax.plot(X_est_KNET[T_start:T_end, idim], 'c-.', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{KNET}}$')  # ### CHANGED
        if X_est_KF is not None:
            ax.plot(X_est_KF[T_start:T_end, idim], ':', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{KF}}$')  # ### CHANGED
        if X_est_EKF is not None:
            ax.plot(X_est_EKF[T_start:T_end, idim], 'b.-', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{EKF}}$')  # ### CHANGED
        if X_est_PINN is not None:  # ### ADDED (PINNSE)
            ax.plot(X_est_PINN[T_start:T_end, idim], 'm^-', lw=lw, label=r'$\hat{\mathbf{x}}_{\mathrm{PINNSE}}$')
        ax.plot(X[T_start:T_end, idim], 'k-', lw=lw, label=r'$\mathbf{x}^{true}$')  # ### CHANGED

        ax.set_ylabel(rf'$x_{idim+1}$')  # ### CHANGED: raw string + f-string
        ax.set_xlabel(r'$t$')            # ### CHANGED

        handles, labels = ax.get_legend_handles_labels()
        order = range(len(handles))
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                  ncol=5, loc=(-0.02, 1.01), fontsize=16)
        plt.tight_layout()

    _maybe_save(plt.gcf(), savefig, savefig_name)
    return None



def plot_measurement_data_axes(
    Y,
    Y_est=None,
    savefig: bool = False,
    savefig_name: str | None = None
):
    """
    按分量分别绘制测量数据（及其估计）的时间序列曲线，支持二维/三维观测量。
    """
    fig = plt.figure(figsize=(20, 10))
    if Y.shape[-1] == 2:
        plt.subplot(311)
        plt.plot(Y[:, 0], '--', label=r'$\mathbf{Y}^{true}\ (x\text{-}component)$')  # ### CHANGED
        if Y_est is not None:
            plt.plot(Y_est[:, 0], '--', label=r'$\hat{\mathbf{Y}}\ (x)$')  # ### CHANGED
        plt.ylabel(r'$Y_1$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.subplot(312)
        plt.plot(Y[:, 1], '--', label=r'$\mathbf{Y}^{true}\ (y\text{-}component)$')  # ### CHANGED
        if Y_est is not None:
            plt.plot(Y_est[:, 1], '--', label=r'$\hat{\mathbf{Y}}\ (y)$')  # ### CHANGED
        plt.ylabel(r'$Y_2$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.tight_layout()

    elif Y.shape[-1] > 2:
        plt.subplot(311)
        plt.plot(Y[:, 0], '--', label=r'$\mathbf{Y}^{true}\ (x\text{-}component)$')  # ### CHANGED
        if Y_est is not None:
            plt.plot(Y_est[:, 0], '--', label=r'$\hat{\mathbf{Y}}\ (x)$')  # ### CHANGED
        plt.ylabel(r'$Y_1$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.subplot(312)
        plt.plot(Y[:, 1], '--', label=r'$\mathbf{Y}^{true}\ (y\text{-}component)$')  # ### CHANGED
        if Y_est is not None:
            plt.plot(Y_est[:, 1], '--', label=r'$\hat{\mathbf{Y}}\ (y)$')  # ### CHANGED
        plt.ylabel(r'$Y_2$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.subplot(313)
        plt.plot(Y[:, 2], '--', label=r'$\mathbf{Y}^{true}\ (z\text{-}component)$')  # ### CHANGED
        if Y_est is not None:
            plt.plot(Y_est[:, 2], '--', label=r'$\hat{\mathbf{Y}}\ (z)$')  # ### CHANGED
        plt.ylabel(r'$Y_3$')  # ### CHANGED
        plt.xlabel(r'$t$')    # ### CHANGED
        plt.legend()

        plt.tight_layout()

    _maybe_save(fig, savefig, savefig_name)
    return None
