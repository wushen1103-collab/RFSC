import numpy as np
import pandas as pd

from scripts.build_rf_gine_ensemble import validation_metrics


def test_validation_metrics_use_internal_cliff_endpoints() -> None:
    prediction = pd.Series({"a": 1.0, "b": 4.0, "c": 3.0})
    truth = pd.Series({"a": 2.0, "b": 2.0, "c": 3.0})
    cliffs = pd.DataFrame({"mol_id_lo": ["a"], "mol_id_hi": ["b"]})
    overall, cliff = validation_metrics(prediction, truth, cliffs)
    assert np.isclose(overall, 1.0)
    assert np.isclose(cliff, 1.5)

