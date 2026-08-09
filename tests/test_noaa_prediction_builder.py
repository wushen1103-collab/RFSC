import numpy as np
import pandas as pd

from scripts.build_noaa_ghcnd_hcn_predictions import (
    causal_lag_residual,
    fourier_matrix,
    normalized_adjacency,
)


def test_causal_lag_uses_only_previous_calendar_day() -> None:
    stations = ["A", "B"]
    source = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        columns=stations,
    )
    prediction_dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(causal_lag_residual(source, prediction_dates, stations), expected)

    perturbed = source.copy()
    perturbed.loc[pd.Timestamp("2024-01-02"), "A"] = 300.0
    actual = causal_lag_residual(perturbed, prediction_dates, stations)
    np.testing.assert_allclose(actual[0], expected[0])
    assert actual[1, 0] == 300.0


def test_missing_previous_day_residual_falls_back_to_zero() -> None:
    source = pd.DataFrame(
        {"A": [np.nan]}, index=pd.DatetimeIndex(["2024-01-01"])
    )
    actual = causal_lag_residual(source, pd.DatetimeIndex(["2024-01-02"]), ["A"])
    np.testing.assert_array_equal(actual, np.zeros((1, 1)))


def test_fourier_features_are_deterministic_and_finite() -> None:
    dates = pd.DatetimeIndex(["2023-01-01", "2024-02-29"])
    first = fourier_matrix(dates, harmonics=4)
    second = fourier_matrix(dates, harmonics=4)
    assert first.shape == (2, 8)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_smoke_graph_isolate_has_zero_neighbor_fallback() -> None:
    stations = pd.DataFrame({"station_id": ["A", "B", "C"]})
    edges = pd.DataFrame(
        {"station_id_lo": ["A"], "station_id_hi": ["B"], "weight": [0.5]}
    )
    adjacency = normalized_adjacency(stations, edges).toarray()
    np.testing.assert_array_equal(adjacency[2], np.zeros(3))
    assert adjacency[0, 1] == 1.0
    assert adjacency[1, 0] == 1.0
