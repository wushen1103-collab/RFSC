from __future__ import annotations

import pandas as pd

from scripts.evaluate_bps_backbone_transfer import prediction_metrics


def test_prediction_metrics_include_point_cliff_and_pair_errors() -> None:
    truth = pd.Series({"a": 0.0, "b": 2.0, "c": 0.2})
    prediction = pd.Series({"a": 0.1, "b": 1.8, "c": 0.3})
    edges = pd.DataFrame(
        {"mol_id_lo": ["a", "a"], "mol_id_hi": ["b", "c"]}
    )
    metrics = prediction_metrics(prediction, truth, edges)
    assert set(metrics) == {
        "overall_mae",
        "overall_rmse",
        "cliff_mae",
        "cliff_rmse",
        "pair_delta_mae",
    }
    assert abs(metrics["overall_mae"] - 0.13333333333333333) < 1e-12
    assert abs(metrics["cliff_mae"] - 0.15) < 1e-12
    assert abs(metrics["pair_delta_mae"] - 0.3) < 1e-12

