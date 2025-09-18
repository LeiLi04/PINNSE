import numpy as np
import pytest

from utils.tools import (
    dB_to_lin,
    generate_normal,
    lin_to_dB,
    partial_corrupt,
    to_float,
)
from utils.gs_utils import create_combined_param_dict, create_list_of_dicts


def test_to_float_parses_strings():
    assert to_float(" 10 dB ") == pytest.approx(10.0)
    assert to_float("-3,5") == pytest.approx(-3.5)
    with pytest.raises(ValueError):
        to_float("not-a-number")


def test_db_conversions_roundtrip():
    values = np.array([0.1, 1.0, 10.0])
    converted = lin_to_dB(values)
    restored = dB_to_lin(converted)
    assert np.allclose(values, restored)
    with pytest.raises(ValueError):
        lin_to_dB([0.0, 1.0])


def test_partial_corrupt_handles_arrays():
    data = np.array([1.0, -2.0])
    result = partial_corrupt(data, p=0.5, bias=0.2)
    expected = np.array([1.0 * (1 + 0.5) + 0.2, -2.0 * (1 - 0.5) + 0.2])
    assert np.allclose(result, expected)


def test_generate_normal_validates_inputs():
    rng = np.random.default_rng(0)
    samples = generate_normal(5, mean=[0.0, 0.0], Sigma2=np.eye(2), rng=rng)
    assert samples.shape == (5, 2)
    with pytest.raises(ValueError):
        generate_normal(0, [0.0], [[1.0]])
    with pytest.raises(ValueError):
        generate_normal(1, [[0.0]], [[1.0]])


def test_grid_utils_generate_combinations():
    combos = create_combined_param_dict({"lr": [1e-3, 1e-4], "epochs": [10, 20]})
    assert len(combos) == 4
    observed = {tuple(sorted(c.items())) for c in combos}
    expected = {
        (('epochs', 10), ('lr', 1e-3)),
        (('epochs', 10), ('lr', 1e-4)),
        (('epochs', 20), ('lr', 1e-3)),
        (('epochs', 20), ('lr', 1e-4)),
    }
    assert observed == expected


def test_create_list_of_dicts_merges(tmp_path):
    options = {
        "rnn_params_dict": {
            "gru": {
                "lr": 1e-3,
            }
        },
        "other": 42,
    }
    combos = create_list_of_dicts(options, "gru", {"lr": [1e-3, 1e-4], "hidden": [32]})
    assert len(combos) == 2
    assert all(cfg["other"] == 42 for cfg in combos)
    assert any(cfg["rnn_params_dict"]["gru"]["lr"] == 1e-4 for cfg in combos)
    with pytest.raises(KeyError):
        create_list_of_dicts({}, "gru", {})
