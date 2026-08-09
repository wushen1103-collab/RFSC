import pandas as pd
import pytest

from scripts.build_svm_bps_selected import mix_predictions


def test_mix_predictions_is_index_aligned_convex_combination() -> None:
    index = pd.Index(["a", "b"], name="mol_id")
    svm = pd.Series([1.0, 3.0], index=index)
    fixed = pd.Series([3.0, 5.0], index=index)
    mixed = mix_predictions(svm, fixed, 0.5)
    assert mixed.tolist() == [2.0, 4.0]


def test_mix_predictions_rejects_partition_order_mismatch() -> None:
    svm = pd.Series([1.0, 2.0], index=["a", "b"])
    fixed = pd.Series([2.0, 1.0], index=["b", "a"])
    with pytest.raises(RuntimeError, match="partitions differ"):
        mix_predictions(svm, fixed, 0.5)
