import numpy as np

from scripts.build_chembl36_temporal_ablations import stable_rng


def test_ablation_random_control_is_stable_and_target_specific() -> None:
    first = stable_rng(2901, "CHEMBL1", 0).choice(100, size=10, replace=False)
    second = stable_rng(2901, "CHEMBL1", 0).choice(100, size=10, replace=False)
    other = stable_rng(2901, "CHEMBL2", 0).choice(100, size=10, replace=False)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
