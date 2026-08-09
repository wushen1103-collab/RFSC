import numpy as np

from scripts.analyze_phase4_confirmatory import holm_adjust


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adjusted["a"], 0.03)
    assert np.isclose(adjusted["c"], 0.06)
    assert np.isclose(adjusted["b"], 0.06)
