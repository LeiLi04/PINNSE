"""
====================================================================
项目 / 算法背景 (Background)
--------------------------------------------------------------------
本文件用于生成机器学习/深度学习模型（如 LSTM/GRU/RNN）超参数组合，实现自动化网格搜索优化。
支持将多个可变参数以笛卡尔积方式组合，自动生成所有参数配置字典，用于训练和实验复现。

输入输出概览 (Input/Output Overview)
--------------------------------------------------------------------
输入:
    - param_dict: 各超参数可选值组成的字典。例如 {"n_hidden": [20, 40], "lr": [1e-3, 1e-4]}
    - options: 固定配置项字典，包含模型设置
    - model_type: 字符串，模型类型（"lstm"/"gru"/"rnn"）
输出:
    - list_of_param_combinations: 参数字典列表，每个元素为一次实验的完整参数集

张量维度/数据结构说明:
    - param_dict: dict(str, list)，每个 key 对应参数名，value 为可选取值列表
    - list_of_param_combinations: List[dict]，每个 dict 为一次模型训练的参数配置

典型应用场景：
    - 超参数调优、批量实验设计、结果复现
====================================================================
"""

import numpy as np
#import matplotlib.pyplot as plt
import itertools
import copy

def create_combined_param_dict(param_dict):
    """
    生成所有超参数组合字典，用于网格搜索实验。
    支持将 param_dict 中所有参数按笛卡尔积全排列生成组合。
    ---
    Args:
        param_dict (dict): 超参数字典，如
            {
                "n_hidden": [20, 30, 40, 50, 60],
                "lr": [1e-3, 1e-4],
                "num_epochs": [2000, 5000],
                "num_layers": [1, 2]
            }
            各 key 为参数名，value 为可选取值列表。

    Returns:
        list_of_param_combinations (list of dict):
            - 每个元素为参数组合字典，例如
                {"n_hidden":20, "lr":1e-3, "num_epochs":2000, "num_layers":1, ...}
            - 用于支持批量训练/评测。
    Tensor Dimensions:
        - param_dict: dict[str, list]，如 n_hidden: [5], lr: [2]，共 5×2=10 种组合
        - list_of_param_combinations: List[dict]，长度等于所有参数取值数的乘积

    Math Notes:
        - 参数组合采用笛卡尔积实现：
          param_combinations = product(param1_values, param2_values, ...)
          每个组合映射为 dict(zip(keys, values))

    """
    # =====================================================================
    # STEP 01: 提取参数名与取值列表，生成笛卡尔积组合
    # =====================================================================
    # step 1a) 获取所有 key/value 对 (参数名, 取值列表)
    keys, values = zip(*param_dict.items())
    # step 1b) itertools.product 生成全部排列组合
    # step 1c) 通过 zip 组装成参数字典
    list_of_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return list_of_param_combinations

def create_list_of_dicts(options, model_type, param_dict):
    """
    结合固定参数和所有超参数组合，生成完整的参数字典列表。
    每个字典可直接用于模型训练，便于自动批量实验。
    ---
    Args:
        options (dict): 固定参数配置，包含模型全局设置，如数据路径、设备等
        model_type (str): 模型类型，支持 "lstm" / "gru" / "rnn"
        param_dict (dict): 超参数字典，内容同上

    Returns:
        params_dict_list_all (list of dict):
            - 每个 dict 包含完整实验参数（固定参数 + 当前组合的超参数）
            - 可直接传递给模型训练主函数

    Tensor Dimensions:
        - options: dict
        - params_dict_list_all: List[dict]，长度等于所有参数组合数

    Math Notes:
        - 对于每一组 param_dict 组合：
          params = deepcopy(options)
          params['rnn_params_dict'][model_type][key] = value  # 逐项更新
          最终生成所有配置字典

    设计选择说明:
        - 使用 deepcopy 防止参数引用导致的串扰（如多个实验间共享同一字典对象）。
        - 支持灵活扩展到多种 RNN 类型，便于后续模型切换或扩展。

    """
    # =====================================================================
    # STEP 01: 生成所有超参数排列组合
    # =====================================================================
    param_combinations_dict = create_combined_param_dict(param_dict)

    # =====================================================================
    # STEP 02: 合并固定参数，构建全量实验参数字典
    # =====================================================================
    params_dict_list_all = []
    for p_dict in param_combinations_dict:
        # -----------------------------------------------------------------
        # step 2a) 深拷贝基础配置，避免对象引用导致覆盖
        tmp_dict = copy.deepcopy(options)
        # step 2b) 将当前超参数组合写入指定模型类型配置
        for key in p_dict.keys():
            tmp_dict['rnn_params_dict'][model_type][key] = p_dict[key]
        # step 2c) 保存完整配置
        params_dict_list_all.append(copy.deepcopy(tmp_dict))

    # =====================================================================
    # STEP 03: 返回参数组合列表
    # =====================================================================
    return params_dict_list_all
