import numpy as np
import pandas as pd

from scripts.build_rf_gine_scale_calibration import scale_name, scale_prediction, validation_metrics


def test_scale_prediction_is_train_centered() -> None:
    prediction = pd.Series({"a": 2.0, "b": 4.0})
    calibrated = scale_prediction(prediction, center=3.0, scale=1.5)
    assert np.allclose(calibrated.to_numpy(), [1.5, 4.5])
    assert np.isclose(calibrated.mean(), 3.0)


def test_validation_metrics_include_pair_delta() -> None:
    prediction = pd.Series({"a": 1.0, "b": 4.0, "c": 3.0})
    truth = pd.Series({"a": 2.0, "b": 2.0, "c": 3.0})
    cliffs = pd.DataFrame(
        {"mol_id_lo": ["a"], "mol_id_hi": ["b"], "delta_y": [0.0]}
    )
    overall, cliff, pair_delta = validation_metrics(prediction, truth, cliffs)
    assert np.isclose(overall, 1.0)
    assert np.isclose(cliff, 1.5)
    assert np.isclose(pair_delta, 3.0)


def test_scale_name_is_stable() -> None:
    assert scale_name(1.0) == "scale_1p00"
    assert scale_name(1.25) == "scale_1p25"

