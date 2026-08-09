#!/usr/bin/env python3
"""Evaluate hash-frozen NOAA forecasts after the same-date response firewall opens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from bps_gnn.io import read_frozen_prediction
try:
    from scripts.build_noaa_ghcnd_hcn_predictions import predict_station_climatology
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from build_noaa_ghcnd_hcn_predictions import predict_station_climatology


METHODS = [
    "climatology",
    "persistence",
    "ridge",
    "hist_gradient_boosting",
    "unscaled_reference",
    "bps",
    "confidence_weighted_bps",
]
METRICS = ["overall_mae", "boundary_endpoint_mae", "boundary_pair_delta_mae"]


def moving_block_relative_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    block_length: int,
    seed: int,
) -> tuple[float, float]:
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline, candidate = baseline[valid], candidate[valid]
    if len(baseline) < block_length or float(np.mean(baseline)) <= 0.0:
        raise RuntimeError("Insufficient paired dates for moving-block interval")
    generator = np.random.default_rng(seed)
    blocks = int(np.ceil(len(baseline) / block_length))
    values = np.empty(iterations, dtype=float)
    offsets = np.arange(block_length)
    for iteration in range(iterations):
        starts = generator.integers(0, len(baseline), size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % len(baseline)).reshape(-1)[: len(baseline)]
        values[iteration] = candidate[indices].mean() / baseline[indices].mean() - 1.0
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def daily_graph_metrics(
    dates: pd.DatetimeIndex,
    station_ids: list[str],
    edges: pd.DataFrame,
    observed_temperature: pd.DataFrame,
    climatology: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    boundary_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date in dates:
        truth = observed_temperature.loc[date]
        climate = climatology.loc[date]
        observed_ids = truth.index[np.isfinite(truth.to_numpy(dtype=float))].tolist()
        true_residual = truth - climate
        eligible_edges = edges[
            edges["station_id_lo"].isin(observed_ids) & edges["station_id_hi"].isin(observed_ids)
        ]
        true_delta = eligible_edges["station_id_hi"].map(true_residual) - eligible_edges[
            "station_id_lo"
        ].map(true_residual)
        boundary_edges = eligible_edges.loc[true_delta.abs() >= boundary_threshold]
        boundary_endpoints = sorted(
            set(boundary_edges["station_id_lo"]) | set(boundary_edges["station_id_hi"])
        )
        for method, prediction_table in predictions.items():
            prediction = prediction_table.loc[date]
            absolute_error = (prediction.loc[observed_ids] - truth.loc[observed_ids]).abs()
            predicted_residual = prediction - climate
            if len(boundary_edges):
                pair_error = (
                    boundary_edges["station_id_hi"].map(predicted_residual)
                    - boundary_edges["station_id_lo"].map(predicted_residual)
                    - (
                        boundary_edges["station_id_hi"].map(true_residual)
                        - boundary_edges["station_id_lo"].map(true_residual)
                    )
                ).abs()
                endpoint_mae = float(absolute_error.loc[boundary_endpoints].mean())
                pair_mae = float(pair_error.mean())
            else:
                endpoint_mae = float("nan")
                pair_mae = float("nan")
            records.append(
                {
                    "date": date,
                    "method": method,
                    "overall_mae": float(absolute_error.mean()),
                    "boundary_endpoint_mae": endpoint_mae,
                    "boundary_pair_delta_mae": pair_mae,
                    "observed_stations": len(observed_ids),
                    "boundary_endpoints": len(boundary_endpoints),
                    "boundary_edges": len(boundary_edges),
                }
            )
    return pd.DataFrame.from_records(records)


def parse_prediction_id(identifier: str) -> tuple[pd.Timestamp, str]:
    date_text, station_id = identifier.split("|", maxsplit=1)
    return pd.Timestamp(date_text), station_id


def prediction_table(series: pd.Series, dates: pd.DatetimeIndex, station_ids: list[str]) -> pd.DataFrame:
    expected_ids = {f"{date:%Y-%m-%d}|{station_id}" for date in dates for station_id in station_ids}
    if set(series.index) != expected_ids:
        raise RuntimeError("Frozen NOAA prediction IDs do not match the response-free date/station grid")
    frame = series.rename("prediction").rename_axis("prediction_id").reset_index()
    parsed = frame["prediction_id"].map(parse_prediction_id)
    frame["date"] = parsed.map(lambda item: item[0])
    frame["station_id"] = parsed.map(lambda item: item[1])
    return frame.pivot(index="date", columns="station_id", values="prediction").reindex(
        index=dates, columns=station_ids
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.prediction_root / "build_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_before_response_evaluation":
        raise RuntimeError("Only the full pre-evaluation NOAA prediction bundle may be evaluated")
    stations = pd.read_parquet(args.data_root / "stations.parquet").sort_values("station_id")
    station_ids = stations["station_id"].tolist()
    edges = pd.read_parquet(args.data_root / "graph_edges.parquet")
    date_index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    validation_dates = date_index[date_index.year == 2023]
    test_dates = date_index[date_index.year == 2024]

    # Hash preflight is completed for every method before same-date responses are opened.
    frozen: dict[str, dict[str, pd.DataFrame]] = {}
    for partition, dates in [("validation", validation_dates), ("test", test_dates)]:
        frozen[partition] = {
            method: prediction_table(
                read_frozen_prediction(
                    args.prediction_root / partition / method / "predictions.parquet"
                ),
                dates,
                station_ids,
            )
            for method in METHODS
        }

    coefficient_frame = pd.read_parquet(args.prediction_root / "climatology_coefficients.parquet")
    coefficient_frame = coefficient_frame.set_index("station_id").loc[station_ids]
    coefficient_columns = ["intercept", *[f"day_sin_h{x}" for x in range(1, 5)], *[f"day_cos_h{x}" for x in range(1, 5)]]
    coefficients = coefficient_frame[coefficient_columns].to_numpy(dtype=float)
    climatology = pd.DataFrame(
        predict_station_climatology(date_index, coefficients, harmonics=4),
        index=date_index,
        columns=station_ids,
    )

    # Same-date responses are first materialized here, after the complete hash preflight.
    labels = pd.read_parquet(args.data_root / "labels_private.parquet")
    labels["date"] = pd.to_datetime(labels["date"])
    observed = labels.pivot(index="date", columns="station_id", values="tmean_c").reindex(
        index=date_index, columns=station_ids
    )
    daily_frames = []
    for partition, dates in [("validation", validation_dates), ("test", test_dates)]:
        daily = daily_graph_metrics(
            dates,
            station_ids,
            edges,
            observed,
            climatology,
            frozen[partition],
            float(manifest["training_boundary_threshold_celsius"]),
        )
        daily.insert(0, "partition", partition)
        daily_frames.append(daily)
    daily_metrics = pd.concat(daily_frames, ignore_index=True)
    aggregate = (
        daily_metrics.groupby(["partition", "method"])[METRICS]
        .mean()
        .reset_index()
    )
    comparison_records: list[dict[str, object]] = []
    evaluation = config["evaluation"]
    iterations = int(evaluation["bootstrap_iterations"])
    seed = int(evaluation["seed"])
    for partition_index, partition in enumerate(["validation", "test"]):
        subset = daily_metrics[daily_metrics["partition"] == partition]
        for method_index, method in enumerate(["bps", "confidence_weighted_bps"]):
            for metric_index, metric in enumerate(METRICS):
                baseline = subset[subset["method"] == "unscaled_reference"].set_index("date")[metric]
                candidate = subset[subset["method"] == method].set_index("date")[metric]
                paired = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
                relative = float(paired["candidate"].mean() / paired["baseline"].mean() - 1.0)
                lower, upper = moving_block_relative_interval(
                    paired["baseline"].to_numpy(),
                    paired["candidate"].to_numpy(),
                    iterations,
                    7,
                    seed + 1000 * partition_index + 100 * method_index + metric_index,
                )
                if np.allclose(paired["baseline"], paired["candidate"]):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(
                            paired["candidate"], paired["baseline"], alternative="less"
                        ).pvalue
                    )
                comparison_records.append(
                    {
                        "partition": partition,
                        "method": method,
                        "baseline": "unscaled_reference",
                        "metric": metric,
                        "dates": len(paired),
                        "baseline_macro": float(paired["baseline"].mean()),
                        "candidate_macro": float(paired["candidate"].mean()),
                        "relative_change": relative,
                        "moving_block95_lower": lower,
                        "moving_block95_upper": upper,
                        "favorable_dates": int((paired["candidate"] < paired["baseline"]).sum()),
                        "favorable_fraction": float((paired["candidate"] < paired["baseline"]).mean()),
                        "paired_wilcoxon_less_p": p_value,
                    }
                )
    comparisons = pd.DataFrame.from_records(comparison_records)
    primary = comparisons[
        (comparisons["partition"] == "test")
        & (comparisons["method"] == "confidence_weighted_bps")
    ].set_index("metric")
    decision_config = config["decision"]
    checks = {
        "pair_relative_improvement": bool(
            primary.loc["boundary_pair_delta_mae", "relative_change"]
            <= -float(decision_config["minimum_pair_relative_improvement"])
        ),
        "pair_moving_block_upper_below_zero": bool(
            primary.loc["boundary_pair_delta_mae", "moving_block95_upper"]
            < float(decision_config["maximum_pair_bootstrap_upper"])
        ),
        "pair_favorable_date_fraction": bool(
            primary.loc["boundary_pair_delta_mae", "favorable_fraction"]
            >= float(decision_config["minimum_pair_favorable_date_fraction"])
        ),
        "overall_point_cost": bool(
            primary.loc["overall_mae", "relative_change"]
            <= float(decision_config["maximum_overall_relative_cost"])
        ),
        "boundary_endpoint_point_cost": bool(
            primary.loc["boundary_endpoint_mae", "relative_change"]
            <= float(decision_config["maximum_boundary_endpoint_relative_cost"])
        ),
    }
    decision = {
        "dataset": config["name"],
        "candidate": "confidence_weighted_bps",
        "baseline": "unscaled_reference",
        "checks": checks,
        "cross_domain_go": all(checks.values()),
        "predictions_hash_preflight_before_response_open": True,
        "same_domain_causal_equivalence_claimed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_metrics.to_csv(args.output_dir / "daily_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "comparison_statistics.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
