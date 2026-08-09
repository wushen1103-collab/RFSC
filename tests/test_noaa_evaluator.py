import numpy as np
import pandas as pd

from scripts.evaluate_noaa_ghcnd_hcn import (
    daily_graph_metrics,
    moving_block_relative_interval,
    prediction_table,
)


def test_moving_block_interval_is_deterministic() -> None:
    baseline = np.linspace(1.0, 2.0, 28)
    candidate = baseline * 0.9
    first = moving_block_relative_interval(baseline, candidate, 200, 7, 29)
    second = moving_block_relative_interval(baseline, candidate, 200, 7, 29)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [-0.1, -0.1])


def test_daily_metrics_use_training_frozen_residual_boundary() -> None:
    date = pd.Timestamp("2024-01-01")
    stations = ["A", "B", "C"]
    edges = pd.DataFrame(
        {
            "station_id_lo": ["A", "B"],
            "station_id_hi": ["B", "C"],
            "weight": [1.0, 1.0],
        }
    )
    truth = pd.DataFrame([[10.0, 14.0, 15.0]], index=[date], columns=stations)
    climate = pd.DataFrame([[10.0, 10.0, 10.0]], index=[date], columns=stations)
    prediction = pd.DataFrame([[10.0, 13.0, 15.0]], index=[date], columns=stations)
    result = daily_graph_metrics(
        pd.DatetimeIndex([date]),
        stations,
        edges,
        truth,
        climate,
        {"candidate": prediction},
        boundary_threshold=3.0,
    ).iloc[0]
    assert result["boundary_edges"] == 1
    assert result["boundary_endpoints"] == 2
    assert np.isclose(result["overall_mae"], 1.0 / 3.0)
    assert np.isclose(result["boundary_endpoint_mae"], 0.5)
    assert np.isclose(result["boundary_pair_delta_mae"], 1.0)


def test_prediction_table_accepts_frozen_identifier_series() -> None:
    dates = pd.DatetimeIndex(["2024-01-01"])
    stations = ["A", "B"]
    series = pd.Series(
        [1.0, 2.0],
        index=["2024-01-01|A", "2024-01-01|B"],
        name="prediction",
    )
    table = prediction_table(series, dates, stations)
    assert table.loc[pd.Timestamp("2024-01-01"), "A"] == 1.0
    assert table.loc[pd.Timestamp("2024-01-01"), "B"] == 2.0
