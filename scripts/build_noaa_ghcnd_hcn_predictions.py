#!/usr/bin/env python3
"""Build response-free causal NOAA station-temperature predictions.

Performance evaluation is deliberately absent. The output is hash-frozen so a
separate evaluator can open same-date validation/test responses afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import yaml
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bps_gnn.io import freeze_prediction_file
from bps_gnn.projection import (
    boundary_projection,
    confidence_weighted_boundary_projection,
    select_active_edges,
)


FEATURE_NAMES = [
    "lag_station_residual_scaled",
    "lag_neighbor_mean_residual_scaled",
    "lag_neighbor_residual_std_scaled",
    "lag_absolute_station_neighbor_gap_scaled",
    *[f"day_sin_h{harmonic}" for harmonic in range(1, 5)],
    *[f"day_cos_h{harmonic}" for harmonic in range(1, 5)],
    "latitude",
    "longitude",
    "elevation_m",
    "graph_degree",
]


def fourier_matrix(dates: pd.DatetimeIndex, harmonics: int = 4) -> np.ndarray:
    phase = 2.0 * np.pi * (dates.dayofyear.to_numpy(dtype=float) - 1.0) / 365.2425
    sine = np.column_stack([np.sin(harmonic * phase) for harmonic in range(1, harmonics + 1)])
    cosine = np.column_stack([np.cos(harmonic * phase) for harmonic in range(1, harmonics + 1)])
    return np.column_stack([sine, cosine])


def fit_station_climatology(
    observations: pd.DataFrame,
    station_ids: list[str],
    train_dates: pd.DatetimeIndex,
    harmonics: int,
    alpha: float,
) -> np.ndarray:
    design = fourier_matrix(train_dates, harmonics)
    coefficients = np.zeros((len(station_ids), 1 + 2 * harmonics), dtype=float)
    for index, station_id in enumerate(station_ids):
        values = observations.loc[train_dates, station_id].to_numpy(dtype=float)
        observed = np.isfinite(values)
        if observed.sum() < 2 * harmonics + 2:
            raise RuntimeError(f"Insufficient climatology observations for {station_id}")
        model = Ridge(alpha=alpha).fit(design[observed], values[observed])
        coefficients[index, 0] = float(model.intercept_)
        coefficients[index, 1:] = model.coef_
    return coefficients


def predict_station_climatology(
    dates: pd.DatetimeIndex,
    coefficients: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(dates)), fourier_matrix(dates, harmonics)])
    return design @ coefficients.T


def normalized_adjacency(stations: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    station_ids = stations["station_id"].tolist()
    index = {station_id: position for position, station_id in enumerate(station_ids)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for left, right, weight in edges[["station_id_lo", "station_id_hi", "weight"]].itertuples(
        index=False, name=None
    ):
        if left not in index or right not in index:
            raise RuntimeError("Station graph contains an unknown station")
        for source, target in [(left, right), (right, left)]:
            rows.append(index[source])
            columns.append(index[target])
            values.append(float(weight))
    adjacency = sparse.csr_matrix((values, (rows, columns)), shape=(len(stations), len(stations)))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_degree = np.divide(
        1.0,
        degree,
        out=np.zeros_like(degree, dtype=float),
        where=degree > 0.0,
    )
    return sparse.diags(inverse_degree) @ adjacency


def causal_lag_residual(
    source_residual: pd.DataFrame,
    prediction_dates: pd.DatetimeIndex,
    station_ids: list[str],
) -> np.ndarray:
    """Return only d-1 residuals for prediction date d, with missing values zeroed."""
    source_dates = prediction_dates - pd.Timedelta(days=1)
    lag = source_residual.reindex(index=source_dates, columns=station_ids).to_numpy(dtype=float)
    return np.nan_to_num(lag, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(
    prediction_dates: pd.DatetimeIndex,
    source_residual: pd.DataFrame,
    stations: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    station_ids = stations["station_id"].tolist()
    lag = causal_lag_residual(source_residual, prediction_dates, station_ids)
    neighbor_mean = np.asarray((adjacency @ lag.T).T)
    neighbor_second = np.asarray((adjacency @ np.square(lag).T).T)
    neighbor_std = np.sqrt(np.maximum(neighbor_second - np.square(neighbor_mean), 0.0))
    scaled = np.column_stack(
        [
            (lag / residual_scale).reshape(-1),
            (neighbor_mean / residual_scale).reshape(-1),
            (neighbor_std / residual_scale).reshape(-1),
            (np.abs(lag - neighbor_mean) / residual_scale).reshape(-1),
        ]
    )
    seasonal = np.repeat(fourier_matrix(prediction_dates, 4), len(stations), axis=0)
    graph_degree = np.asarray(adjacency.astype(bool).sum(axis=1)).ravel().astype(float)
    metadata = stations[["latitude", "longitude", "elevation_m"]].to_numpy(dtype=float)
    metadata = np.column_stack([metadata, graph_degree])
    station_features = np.tile(metadata, (len(prediction_dates), 1))
    features = np.column_stack([scaled, seasonal, station_features])
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise RuntimeError("Invalid frozen NOAA feature matrix")
    return features, lag


def load_observation_pivot(data_root: Path, station_ids: list[str]) -> pd.DataFrame:
    dataset = pads.dataset(data_root / "labels_private.parquet", format="parquet")
    frame = dataset.to_table(columns=["station_id", "date", "tmean_c"]).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"])
    pivot = frame.pivot(index="date", columns="station_id", values="tmean_c")
    return pivot.reindex(columns=station_ids).sort_index()


def prediction_ids(dates: pd.DatetimeIndex, station_ids: list[str]) -> np.ndarray:
    return np.asarray(
        [f"{date:%Y-%m-%d}|{station_id}" for date in dates for station_id in station_ids],
        dtype=object,
    )


def training_boundary_threshold(
    residual: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    edges: pd.DataFrame,
    quantile: float,
) -> tuple[float, int]:
    values: list[np.ndarray] = []
    for left, right in edges[["station_id_lo", "station_id_hi"]].itertuples(index=False, name=None):
        delta = (residual.loc[train_dates, right] - residual.loc[train_dates, left]).abs().to_numpy()
        values.append(delta[np.isfinite(delta)])
    pooled = np.concatenate(values)
    if not len(pooled):
        raise RuntimeError("No paired training residuals for boundary threshold")
    return float(np.quantile(pooled, quantile)), int(len(pooled))


def project_partition(
    dates: pd.DatetimeIndex,
    station_ids: list[str],
    edges: pd.DataFrame,
    residual_predictions: dict[str, np.ndarray],
    base_weights: dict[str, float],
    active_threshold: float,
    scale: float,
    floor_additive: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    bases = np.column_stack([residual_predictions[name] for name in base_weights])
    weights = np.asarray([base_weights[name] for name in base_weights], dtype=float)
    unscaled = bases @ weights
    ordinary = np.empty_like(unscaled)
    confidence_weighted = np.empty_like(unscaled)
    active_counts: list[int] = []
    max_invariant_error = 0.0
    station_count = len(station_ids)
    for date_index, _ in enumerate(dates):
        start, stop = date_index * station_count, (date_index + 1) * station_count
        signal = pd.Series(unscaled[start:stop], index=station_ids, dtype=float)
        active = select_active_edges(
            signal,
            edges,
            active_threshold,
            left_column="station_id_lo",
            right_column="station_id_hi",
        )
        ordinary_signal = boundary_projection(
            signal,
            active,
            scale,
            left_column="station_id_lo",
            right_column="station_id_hi",
        )
        base_slice = bases[start:stop]
        uncertainty = np.sum(weights * np.square(base_slice - unscaled[start:stop, None]), axis=1)
        uncertainty_floor = float(np.median(uncertainty)) + floor_additive
        precision = pd.Series(1.0 / (uncertainty + uncertainty_floor), index=station_ids)
        weighted_signal = confidence_weighted_boundary_projection(
            signal,
            active,
            scale,
            precision,
            left_column="station_id_lo",
            right_column="station_id_hi",
        )
        if len(active):
            ordinary_delta = active["station_id_hi"].map(ordinary_signal) - active["station_id_lo"].map(
                ordinary_signal
            )
            weighted_delta = active["station_id_hi"].map(weighted_signal) - active["station_id_lo"].map(
                weighted_signal
            )
            max_invariant_error = max(
                max_invariant_error,
                float(np.max(np.abs(ordinary_delta.to_numpy() - weighted_delta.to_numpy()))),
            )
        ordinary[start:stop] = ordinary_signal.to_numpy()
        confidence_weighted[start:stop] = weighted_signal.to_numpy()
        active_counts.append(len(active))
    residual_predictions = {
        **residual_predictions,
        "unscaled_reference": unscaled,
        "bps": ordinary,
        "confidence_weighted_bps": confidence_weighted,
    }
    diagnostics = {
        "mean_active_edges_per_date": float(np.mean(active_counts)),
        "minimum_active_edges_per_date": int(np.min(active_counts)),
        "maximum_active_edges_per_date": int(np.max(active_counts)),
        "max_active_delta_difference_vs_ordinary_bps": max_invariant_error,
    }
    return residual_predictions, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stations = pd.read_parquet(args.data_root / "stations.parquet").sort_values("station_id")
    edges = pd.read_parquet(args.data_root / "graph_edges.parquet")
    station_ids = stations["station_id"].tolist()
    observations = load_observation_pivot(args.data_root, station_ids)
    date_index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    observations = observations.reindex(date_index)
    train_dates = date_index[date_index.year <= 2022]
    validation_dates = date_index[date_index.year == 2023]
    test_dates = date_index[date_index.year == 2024]
    forecast = config["forecast"]
    harmonics = int(forecast["climatology_harmonics"])
    coefficients = fit_station_climatology(
        observations,
        station_ids,
        train_dates,
        harmonics,
        float(forecast["climatology_ridge_alpha"]),
    )
    climatology = pd.DataFrame(
        predict_station_climatology(date_index, coefficients, harmonics),
        index=date_index,
        columns=station_ids,
    )
    residual = observations - climatology
    train_residual = residual.loc[train_dates].to_numpy(dtype=float)
    residual_scale = float(np.nanmedian(np.abs(train_residual))) + 1e-6
    adjacency = normalized_adjacency(stations, edges)
    train_features, _ = build_features(train_dates, residual, stations, adjacency, residual_scale)
    train_targets = train_residual.reshape(-1)
    observed = np.isfinite(train_targets)
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(forecast["ridge_alpha"])),
    ).fit(train_features[observed], train_targets[observed])
    hgb_config = dict(forecast["hist_gradient_boosting"])
    if args.smoke:
        hgb_config["max_iter"] = min(5, int(hgb_config["max_iter"]))
    hgb = HistGradientBoostingRegressor(**hgb_config).fit(
        train_features[observed], train_targets[observed]
    )
    boundary_threshold, boundary_pairs = training_boundary_threshold(
        residual,
        train_dates,
        edges,
        float(config["boundary_evaluation"]["train_quantile"]),
    )
    base_weights = {name: float(weight) for name, weight in forecast["base_weights"].items()}
    if list(base_weights) != ["persistence", "ridge", "hist_gradient_boosting"]:
        raise RuntimeError("Unexpected frozen NOAA base-model order")
    if not np.isclose(sum(base_weights.values()), 1.0):
        raise RuntimeError("Frozen NOAA base weights must sum to one")
    operator = config["operator"]
    active_threshold = float(operator["predicted_delta_threshold_train_scale_units"]) * residual_scale
    model_manifest: dict[str, object] = {
        "status": "smoke_not_paper_evidence" if args.smoke else "frozen_before_response_evaluation",
        "stations": len(stations),
        "edges": len(edges),
        "train_observed_rows": int(observed.sum()),
        "features": FEATURE_NAMES,
        "residual_scale_celsius": residual_scale,
        "active_threshold_celsius": active_threshold,
        "training_boundary_threshold_celsius": boundary_threshold,
        "training_boundary_edge_date_pairs": boundary_pairs,
        "same_or_future_response_dependencies": False,
        "allowed_lag_days": 1,
        "test_prediction_latest_response_source": "2024-12-30",
        "hgb_max_iter": int(hgb_config["max_iter"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_columns = ["intercept", *FEATURE_NAMES[4:12]]
    pd.DataFrame(coefficients, columns=coefficient_columns).assign(station_id=station_ids).to_parquet(
        args.output_dir / "climatology_coefficients.parquet", index=False
    )
    partition_summaries: dict[str, dict[str, float]] = {}
    for partition, dates in [("validation", validation_dates), ("test", test_dates)]:
        features, lag = build_features(dates, residual, stations, adjacency, residual_scale)
        residual_predictions = {
            "climatology": np.zeros(len(features), dtype=float),
            "persistence": lag.reshape(-1),
            "ridge": ridge.predict(features),
            "hist_gradient_boosting": hgb.predict(features),
        }
        residual_predictions, diagnostics = project_partition(
            dates,
            station_ids,
            edges,
            residual_predictions,
            base_weights,
            active_threshold,
            float(operator["scale"]),
            1e-6,
        )
        ids = prediction_ids(dates, station_ids)
        climate_values = climatology.loc[dates].to_numpy().reshape(-1)
        for method, predicted_residual in residual_predictions.items():
            frame = pd.DataFrame(
                {"mol_id": ids, "prediction": climate_values + predicted_residual}
            )
            metadata = {
                **model_manifest,
                **diagnostics,
                "partition": partition,
                "method": method,
                "prediction_rows": len(frame),
                "same_date_response_used_as_feature": False,
                "future_response_used_as_feature": False,
                "causal_lag_response_allowed": True,
            }
            freeze_prediction_file(
                frame,
                args.output_dir / partition / method / "predictions.parquet",
                metadata,
            )
        partition_summaries[partition] = diagnostics
    model_manifest["partition_diagnostics"] = partition_summaries
    model_manifest["wall_time_seconds"] = time.perf_counter() - started
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(model_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
