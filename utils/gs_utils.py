"""Grid-search helper utilities for hyper-parameter sweeps.

项目背景 (Project Background):
    为 RNN/DANSE/PINN 等模型生成超参数组合，支撑自动化网格搜索实验。

输入输出概览 (Input/Output Overview):
    - 输入: 超参数候选字典 `param_dict` 与基础配置 `options`。
    - 输出: 所有组合后的配置列表，可直接喂给训练脚本。
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


def create_combined_param_dict(param_dict: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """生成超参数笛卡尔组合 / Build full cartesian product.

    Args:
        param_dict (Mapping[str, Sequence[Any]]): 超参数候选集合，键为参数名。

    Returns:
        List[Dict[str, Any]]: 每个元素都是一次实验的参数字典。

    Tensor Dimensions:
        - param_dict[k]: [N_k]
        - return: [Π N_k]

    Math Notes:
        combos = product(*param_dict.values())
    """

    # ==================================================================
    # STEP 01: 参数合法性检查 (Validate search space)
    # ------------------------------------------------------------------
    if not param_dict:
        raise ValueError("param_dict must contain at least one parameter")
    keys, values = zip(*param_dict.items())
    for key, candidates in zip(keys, values):
        if len(candidates) == 0:
            raise ValueError(f"Parameter '{key}' has an empty candidate list")
    # ==================================================================
    # STEP 02: 生成组合 (Compute cartesian product)
    # ------------------------------------------------------------------
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _copy_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """浅拷贝字典 / Clone mapping to avoid mutation leaks.

    Args:
        mapping (Mapping[str, Any]): 原始配置字典。

    Returns:
        Dict[str, Any]: 浅拷贝结果，嵌套 dict 同样复制。

    Tensor Dimensions:
        - 不涉及张量。

    Math Notes:
        无数学运算。
    """

    cloned: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def create_list_of_dicts(
    options: Mapping[str, Any],
    model_type: str,
    param_dict: Mapping[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    """合成完整实验配置 / Merge base options with hyper-parameter combos.

    Args:
        options (Mapping[str, Any]): 基础配置，需包含 `rnn_params_dict`。
        model_type (str): 目标模型类型键（如 "gru"）。
        param_dict (Mapping[str, Sequence[Any]]): 超参数候选集合。

    Returns:
        List[Dict[str, Any]]: 合并后的配置列表。

    Tensor Dimensions:
        - return: [Π N_k]

    Math Notes:
        cfg_model_params = base_params.copy(); cfg_model_params.update(combo)
    """

    # ==================================================================
    # STEP 01: 校验基础配置 (Validate base options)
    # ------------------------------------------------------------------
    if "rnn_params_dict" not in options:
        raise KeyError("options must contain 'rnn_params_dict'")

    base_rnn_params = options["rnn_params_dict"]
    if model_type not in base_rnn_params:
        raise KeyError(f"model_type '{model_type}' not found in options['rnn_params_dict']")

    combinations = create_combined_param_dict(param_dict)
    expanded_configs: List[Dict[str, Any]] = []

    # ==================================================================
    # STEP 02: 合并每个组合 (Merge each hyper-parameter combo)
    # ------------------------------------------------------------------
    for combo in combinations:
        cfg = _copy_mapping(options)
        cfg_rnn_dict = _copy_mapping(cfg.setdefault("rnn_params_dict", {}))
        cfg_model_params = _copy_mapping(cfg_rnn_dict.setdefault(model_type, {}))
        cfg_model_params.update(combo)
        cfg_rnn_dict[model_type] = cfg_model_params
        cfg["rnn_params_dict"] = cfg_rnn_dict
        expanded_configs.append(cfg)

    return expanded_configs


__all__ = ["create_combined_param_dict", "create_list_of_dicts"]
