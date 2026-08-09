#!/usr/bin/env python3
"""Apply the preregistered validation-only gate to the Multi-RF feature smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--gine-validation", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    return parser.parse_args()


def verified_frame(path: Path) -> pd.DataFrame:
    manifest = json.loads(
        path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8")
    )
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_parquet(path)


def prediction_series(path: Path) -> pd.Series:
    frame = verified_frame(path)
    if set(frame.columns) != {"mol_id", "prediction"} or frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Unexpected validation prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def gine_series(path: Path, seed: int) -> pd.Series:
    frame = verified_frame(path)
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError("Duplicate GINE validation molecule")
    return frame.set_index("mol_id")["prediction"].sort_index()


def metrics(
    prediction: pd.Series,
    truth: pd.Series,
    cliff_edges: pd.DataFrame,
) -> dict[str, float]:
    endpoint_ids = sorted(
        (set(cliff_edges["mol_id_lo"]) | set(cliff_edges["mol_id_hi"]))
        & set(prediction.index)
    )
    predicted_delta = (
        cliff_edges["mol_id_hi"].map(prediction)
        - cliff_edges["mol_id_lo"].map(prediction)
    )
    true_delta = (
        cliff_edges["mol_id_hi"].map(truth)
        - cliff_edges["mol_id_lo"].map(truth)
    )
    error = prediction - truth.loc[prediction.index]
    return {
        "overall_mae": float(np.mean(np.abs(error))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "cliff_endpoint_mae": float(
            np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))
        ),
        "cliff_pair_delta_mae": float(np.mean(np.abs(predicted_delta - true_delta))),
        "n_cliff_endpoints": len(endpoint_ids),
        "n_cliff_pairs": len(cliff_edges),
    }


def main() -> None:
    args = parse_args()
    new_run = json.loads((args.new_run / "run.json").read_text(encoding="utf-8"))
    if not new_run["validation_only"] or new_run["outer_test_labels_materialized"]:
        raise RuntimeError("Smoke must remain validation-only")
    new_rf = prediction_series(args.new_run / "validation_predictions.parquet")
    reference_rf = prediction_series(args.reference_run / "validation_predictions.parquet")
    gine = gine_series(args.gine_validation, args.seed)
    if not new_rf.index.equals(reference_rf.index) or not new_rf.index.equals(gine.index):
        raise RuntimeError("Validation molecule mismatch")
    truth = read_labels_for_ids(args.data_root, args.target, new_rf.index).set_index("mol_id")[
        "y_pactivity"
    ]
    edges = pd.read_parquet(args.mmp_root / args.target / "eligible_mmp_edges.parquet")
    internal = edges[
        edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
    ].copy()
    internal_delta = (
        internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
    )
    cliffs = internal[internal_delta.abs() >= 1.0].copy()
    if cliffs.empty:
        raise RuntimeError("No internal validation cliff pair")
    predictions = {
        "reference_rf": reference_rf,
        "multifingerprint_rf": new_rf,
        "reference_fixed_equal": 0.5 * reference_rf + 0.5 * gine,
        "multifingerprint_fixed_equal": 0.5 * new_rf + 0.5 * gine,
    }
    rows = []
    for method, prediction in predictions.items():
        rows.append({"method": method, **metrics(prediction, truth, cliffs)})
    table = pd.DataFrame(rows).set_index("method")
    reference_rf_metrics = table.loc["reference_rf"]
    new_rf_metrics = table.loc["multifingerprint_rf"]
    reference_equal = table.loc["reference_fixed_equal"]
    new_equal = table.loc["multifingerprint_fixed_equal"]
    criteria = {
        "rf_rmse_improves_at_least_1pct": bool(
            new_rf_metrics["overall_rmse"] <= 0.99 * reference_rf_metrics["overall_rmse"]
        ),
        "fixed_equal_overall_nonworse": bool(
            new_equal["overall_mae"] <= reference_equal["overall_mae"]
        ),
        "fixed_equal_cliff_nonworse": bool(
            new_equal["cliff_endpoint_mae"] <= reference_equal["cliff_endpoint_mae"]
        ),
        "fixed_equal_pair_within_0_5pct": bool(
            new_equal["cliff_pair_delta_mae"]
            <= 1.005 * reference_equal["cliff_pair_delta_mae"]
        ),
    }
    decision = {
        "status": "GO" if all(criteria.values()) else "NO-GO",
        "validation_only": True,
        "test_labels_materialized": False,
        "criteria": criteria,
        "relative_changes_pct": {
            metric: float(
                100.0 * (new_equal[metric] / reference_equal[metric] - 1.0)
            )
            for metric in ["overall_mae", "cliff_endpoint_mae", "cliff_pair_delta_mae"]
        },
        "rf_rmse_relative_change_pct": float(
            100.0
            * (new_rf_metrics["overall_rmse"] / reference_rf_metrics["overall_rmse"] - 1.0)
        ),
        "action": (
            "preregister_two_point_replication"
            if all(criteria.values())
            else "stop_without_test_evaluation_or_feature_tuning"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    table.reset_index().to_csv(args.output_table, index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
