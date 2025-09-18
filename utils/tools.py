# ==========================================================================
# 工具函数
# ==========================================================================
def to_float(x):
    # 防御式：去空格、去“dB”、替换逗号，最后转 float
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace('dB','').replace(',', '.')
    return float(s)

def dB_to_lin(x):
    """
    dB值转线性刻度 (Decibel to linear scale).
    Args:
        x (float or np.ndarray): dB value(s).
    Returns:
        float or np.ndarray: linear value(s), x_lin = 10^(x/10).
    Tensor Dimensions:
        - Supports scalar or vector [N,].
    Math Notes:
        - lin = 10 ** (x / 10).
    """
    return 10**(x/10)

def lin_to_dB(x):
    """
    线性刻度转dB值 (Linear to dB).
    Args:
        x (float or np.ndarray): linear value(s). must be non-zero.
    Returns:
        float or np.ndarray: dB value(s).
    Math Notes:
        - dB = 10 * log10(x).
    """
    assert x != 0, "X is zero"
    return 10*np.log10(x)

def partial_corrupt(x, p=0.7, bias=0.0):
    """
    部分扰动数据，模拟噪声偏移 (simulate corruption).
    Args:
        x (float): original value.
        p (float): perturbation ratio, default 0.7.
        bias (float): additive bias term.
    Returns:
        float: perturbed value.
    Math Notes:
        - if x < 0, sign of p is flipped.
        - y = x * (1 + p) + bias.
    """
    if x < 0:
        p *= -1
    return x*(1+p)

def generate_normal(N, mean, Sigma2):
    """
    生成多元高斯采样 (Multivariate Normal sampling).
    Args:
        N (int): number of samples.
        mean (array-like): mean vector [D].
        Sigma2 (array-like): covariance matrix [D, D].
    Returns:
        np.ndarray: samples [N, D].
    Math Notes:
        - n ~ N(mean, Sigma2).
    """
    n = np.random.multivariate_normal(mean=mean, cov=Sigma2, size=(N,))
    return n