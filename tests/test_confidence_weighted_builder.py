import numpy as np
import pandas as pd


def test_weighted_disagreement_is_zero_only_when_bases_agree() -> None:
    svm = pd.Series([1.0, 2.0])
    rf = pd.Series([1.0, 3.0])
    gine = pd.Series([1.0, 4.0])
    mean = 0.5 * svm + 0.25 * rf + 0.25 * gine
    uncertainty = 0.5 * (svm - mean) ** 2 + 0.25 * (rf - mean) ** 2 + 0.25 * (gine - mean) ** 2
    assert uncertainty.iloc[0] == 0.0
    assert uncertainty.iloc[1] > 0.0
    assert np.isfinite((1.0 / (uncertainty + np.median(uncertainty) + 1e-6))).all()
