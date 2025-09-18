"""Grid-search helper utilities for hyper-parameter sweeps."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


def create_combined_param_dict(param_dict: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Return the full cartesian product of hyper-parameter candidates.

    Args:
        param_dict: Mapping from parameter name to an iterable of candidate
            values. Each iterable must be non-empty.
    """

    if not param_dict:
        raise ValueError("param_dict must contain at least one parameter")
    keys, values = zip(*param_dict.items())
    for key, candidates in zip(keys, values):
        if len(candidates) == 0:
            raise ValueError(f"Parameter '{key}' has an empty candidate list")
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _copy_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow copy mapping, cloning nested dicts to avoid mutation leaks."""

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
    """Expand ``options`` with all hyper-parameter combinations for ``model_type``.

    Args:
        options: Baseline configuration dictionary that must include an
            ``'rnn_params_dict'`` entry.
        model_type: Key inside ``options['rnn_params_dict']`` to be augmented.
        param_dict: Hyper-parameter search space per ``create_combined_param_dict``.
    """

    if "rnn_params_dict" not in options:
        raise KeyError("options must contain 'rnn_params_dict'")

    base_rnn_params = options["rnn_params_dict"]
    if model_type not in base_rnn_params:
        raise KeyError(f"model_type '{model_type}' not found in options['rnn_params_dict']")

    combinations = create_combined_param_dict(param_dict)
    expanded_configs: List[Dict[str, Any]] = []

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
