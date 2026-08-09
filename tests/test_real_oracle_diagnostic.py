from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_real_oracle_diagnostic.py"
SPEC = spec_from_file_location("run_real_oracle_diagnostic", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_propagation_uses_weighted_frozen_predictions():
    result = MODULE.apply_propagation(
        np.array([1.0]),
        np.array([3.0, 5.0]),
        [np.array([0, 1])],
        [np.array([0.75, 0.25])],
        [np.array([False, False])],
        0.5,
    )
    assert np.allclose(result, [2.25])


def test_all_cut_and_no_neighbor_fall_back_to_base():
    result = MODULE.apply_propagation(
        np.array([1.0, 2.0]),
        np.array([3.0]),
        [np.array([0]), np.array([], dtype=int)],
        [np.array([0.8]), np.array([], dtype=float)],
        [np.array([True]), np.array([], dtype=bool)],
        0.75,
    )
    assert np.array_equal(result, [1.0, 2.0])


def test_metric_endpoint_count_is_unique():
    truth = pd.Series({"a": 0.0, "b": 1.2, "c": 1.4})
    prediction = pd.Series({"a": 0.1, "b": 1.0, "c": 1.3})
    edges = pd.DataFrame(
        {"mol_id_lo": ["a", "a"], "mol_id_hi": ["b", "c"]}
    )
    result = MODULE.metric_row(truth, prediction, edges)
    assert result["n_cliff_pairs"] == 2
    assert result["n_cliff_endpoints"] == 3


def test_random_gate_seed_is_reproducible():
    first = MODULE.stable_rng(1, "target", 2, 3, 4).choice(10, size=4, replace=False)
    second = MODULE.stable_rng(1, "target", 2, 3, 4).choice(10, size=4, replace=False)
    assert np.array_equal(first, second)


def test_binary_oracle_dual_channel_is_signed():
    result = MODULE.apply_dual_propagation(
        np.array([3.0]),
        np.array([1.0]),
        [np.array([0])],
        [np.array([0.8])],
        [np.ones(1)],
        0.5,
        0.25,
    )
    assert np.allclose(result, [3.5])
