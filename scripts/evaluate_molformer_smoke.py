#!/usr/bin/env python3
"""Evaluate one preregistered frozen MoLFormer validation point."""

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
    parser.add_argument("--rf-run", type=Path, required=True)
    parser.add_argument("--gine-run", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    return parser.parse_args()


def verified_frame(path: Path) -> pd.DataFrame:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_parquet(path)


def prediction_series(path: Path) -> pd.Series:
    frame = verified_frame(path)
    if set(frame.columns) != {"mol_id", "prediction"} or frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Unexpected prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def gine_series(path: Path, seed: int) -> pd.Series:
    frame = verified_frame(path)
    required = {"mol_id", "prediction", "method", "seed"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Unexpected GINE validation schema: {path}")
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError("Duplicate GINE validation molecule")
    return frame.set_index("mol_id")["prediction"].sort_index()


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict[str, float]:
    error = prediction - truth.loc[prediction.index]
    endpoints = sorted((set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])) & set(prediction.index))
    predicted_delta = prediction.reindex(cliffs["mol_id_hi"]).to_numpy() - prediction.reindex(cliffs["mol_id_lo"]).to_numpy()
    true_delta = truth.reindex(cliffs["mol_id_hi"]).to_numpy() - truth.reindex(cliffs["mol_id_lo"]).to_numpy()
    return {
        "overall_mae": float(np.mean(np.abs(error))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "cliff_endpoint_mae": float(np.mean(np.abs(prediction.loc[endpoints] - truth.loc[endpoints]))),
        "cliff_pair_delta_mae": float(np.mean(np.abs(predicted_delta - true_delta))),
        "n_cliff_endpoints": len(endpoints),
        "n_cliff_pairs": int(len(cliffs)),
    }


def main() -> None:
    args = parse_args()
    run = json.loads((args.new_run / "run.json").read_text(encoding="utf-8"))
    if not run["validation_only"] or run["outer_test_labels_materialized"] or run["outer_test_predictions_generated"]:
        raise RuntimeError("Candidate must remain validation-only")
    candidate = prediction_series(args.new_run / "validation_predictions.parquet")
    rf = prediction_series(args.rf_run / "validation_predictions.parquet")
    gine = gine_series(args.gine_run / "validation_predictions.parquet", seed=0)
    if not candidate.index.equals(rf.index) or not candidate.index.equals(gine.index):
        raise RuntimeError("Validation molecule mismatch")
    truth = read_labels_for_ids(args.data_root, args.target, candidate.index).set_index("mol_id")["y_pactivity"]
    edges = pd.read_parquet(args.mmp_root / args.target / "eligible_mmp_edges.parquet")
    internal = edges[edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)].copy()
    true_delta = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
    cliffs = internal[true_delta.abs() >= 1.0].copy()
    if cliffs.empty:
        raise RuntimeError("No internal validation cliff pair")
    predictions = {
        "gine": gine,
        "rf": rf,
        "molformer_ridge": candidate,
        "fixed_rf_gine": 0.5 * rf + 0.5 * gine,
        "equal_rf_gine_molformer": (rf + gine + candidate) / 3.0,
    }
    table = pd.DataFrame([{"method": name, **metrics(pred, truth, cliffs)} for name, pred in predictions.items()]).set_index("method")
    base = table.loc["fixed_rf_gine"]
    ensemble = table.loc["equal_rf_gine_molformer"]
    candidate_row = table.loc["molformer_ridge"]
    criteria = {
        "candidate_global_rmse_noninferior_to_gine_10pct": bool(candidate_row["overall_rmse"] <= 1.10 * table.loc["gine", "overall_rmse"]),
        "ensemble_overall_not_worse_than_base_0_5pct": bool(ensemble["overall_mae"] <= 1.005 * base["overall_mae"]),
        "ensemble_cliff_improves_at_least_0_5pct": bool(ensemble["cliff_endpoint_mae"] <= 0.995 * base["cliff_endpoint_mae"]),
        "ensemble_pair_not_worse_than_base_2pct": bool(ensemble["cliff_pair_delta_mae"] <= 1.02 * base["cliff_pair_delta_mae"]),
    }
    decision = {
        "status": "GO" if all(criteria.values()) else "NO-GO",
        "validation_only": True,
        "outer_test_labels_materialized": False,
        "criteria": criteria,
        "candidate_relative_to_gine_pct": {
            metric: float(100.0 * (candidate_row[metric] / table.loc["gine", metric] - 1.0))
            for metric in ["overall_mae", "cliff_endpoint_mae", "cliff_pair_delta_mae"]
        },
        "ensemble_relative_to_fixed_rf_gine_pct": {
            metric: float(100.0 * (ensemble[metric] / base[metric] - 1.0))
            for metric in ["overall_mae", "cliff_endpoint_mae", "cliff_pair_delta_mae"]
        },
        "action": "preregister_two_point_replication" if all(criteria.values()) else "stop_without_test_evaluation_or_parameter_search",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    table.reset_index().to_csv(args.output_table, index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
