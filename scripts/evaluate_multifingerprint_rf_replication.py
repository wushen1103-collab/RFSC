#!/usr/bin/env python3
"""Evaluate the frozen two-point Multi-RF feature replication on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--reference-rf-root", type=Path, required=True)
    parser.add_argument("--gine-validation-root", type=Path, required=True)
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
    if frame.empty or frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Invalid GINE validation prediction: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> dict[str, float]:
    endpoint_ids = sorted(
        (set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])) & set(prediction.index)
    )
    error = prediction - truth.loc[prediction.index]
    predicted_delta = cliffs["mol_id_hi"].map(prediction) - cliffs["mol_id_lo"].map(
        prediction
    )
    true_delta = cliffs["mol_id_hi"].map(truth) - cliffs["mol_id_lo"].map(truth)
    return {
        "overall_mae": float(np.mean(np.abs(error))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "cliff_endpoint_mae": float(
            np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids]))
        ),
        "cliff_pair_delta_mae": float(np.mean(np.abs(predicted_delta - true_delta))),
        "n_cliff_endpoints": len(endpoint_ids),
        "n_cliff_pairs": len(cliffs),
    }


def relative_change(new: float, reference: float) -> float:
    return float(100.0 * (new / reference - 1.0))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["replication_gate"]
    records = []
    for task in config["replication_tasks"]:
        target, fold, seed = task["target"], int(task["fold"]), int(task["seed"])
        relative = Path(target) / f"fold_{fold}" / f"seed_{seed}"
        new_run_path = args.new_root / relative
        reference_run_path = args.reference_rf_root / relative
        run = json.loads((new_run_path / "run.json").read_text(encoding="utf-8"))
        if not run["validation_only"] or run["outer_test_predictions_generated"]:
            raise RuntimeError(f"Replication is not validation-only: {relative}")
        new_rf = prediction_series(new_run_path / "validation_predictions.parquet")
        reference_rf = prediction_series(
            reference_run_path / "validation_predictions.parquet"
        )
        gine = gine_series(
            args.gine_validation_root
            / target
            / f"fold_{fold}"
            / "validation_predictions.parquet",
            seed,
        )
        if not new_rf.index.equals(reference_rf.index) or not new_rf.index.equals(gine.index):
            raise RuntimeError(f"Validation molecule mismatch: {relative}")
        truth = read_labels_for_ids(args.data_root, target, new_rf.index).set_index("mol_id")[
            "y_pactivity"
        ]
        edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = edges[
            edges["mol_id_lo"].isin(truth.index) & edges["mol_id_hi"].isin(truth.index)
        ].copy()
        true_delta = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
        cliffs = internal[true_delta.abs() >= 1.0].copy()
        if cliffs.empty:
            raise RuntimeError(f"No validation cliff pair: {relative}")
        reference_rf_metrics = metrics(reference_rf, truth, cliffs)
        new_rf_metrics = metrics(new_rf, truth, cliffs)
        reference_equal = metrics(0.5 * reference_rf + 0.5 * gine, truth, cliffs)
        new_equal = metrics(0.5 * new_rf + 0.5 * gine, truth, cliffs)
        changes = {
            "rf_rmse_change_pct": relative_change(
                new_rf_metrics["overall_rmse"], reference_rf_metrics["overall_rmse"]
            ),
            "fixed_equal_overall_change_pct": relative_change(
                new_equal["overall_mae"], reference_equal["overall_mae"]
            ),
            "fixed_equal_cliff_change_pct": relative_change(
                new_equal["cliff_endpoint_mae"], reference_equal["cliff_endpoint_mae"]
            ),
            "fixed_equal_pair_change_pct": relative_change(
                new_equal["cliff_pair_delta_mae"],
                reference_equal["cliff_pair_delta_mae"],
            ),
        }
        criteria = {
            "rf_rmse_improves_at_least_1pct": changes["rf_rmse_change_pct"]
            <= -100.0 * float(gate["minimum_rf_rmse_improvement"]),
            "fixed_equal_overall_nonworse": changes["fixed_equal_overall_change_pct"]
            <= 100.0 * float(gate["fixed_equal_overall_max_relative_change"]),
            "fixed_equal_cliff_nonworse": changes["fixed_equal_cliff_change_pct"]
            <= 100.0 * float(gate["fixed_equal_cliff_max_relative_change"]),
            "fixed_equal_pair_within_0_5pct": changes["fixed_equal_pair_change_pct"]
            <= 100.0 * float(gate["fixed_equal_pair_delta_max_relative_change"]),
        }
        records.append(
            {
                "target": target,
                "fold": fold,
                "seed": seed,
                **changes,
                **criteria,
                "task_status": "GO" if all(criteria.values()) else "NO-GO",
                "n_cliff_endpoints": new_equal["n_cliff_endpoints"],
                "n_cliff_pairs": new_equal["n_cliff_pairs"],
                "validation_prediction_sha256": run["validation_prediction_sha256"],
            }
        )
    table = pd.DataFrame(records)
    matrix_go = bool((table["task_status"] == "GO").all())
    decision = {
        "status": "GO" if matrix_go else "NO-GO",
        "validation_only": True,
        "test_labels_materialized": False,
        "required_task_passes": len(records),
        "observed_task_passes": int((table["task_status"] == "GO").sum()),
        "mean_relative_changes_pct": {
            column: float(table[column].mean())
            for column in [
                "rf_rmse_change_pct",
                "fixed_equal_overall_change_pct",
                "fixed_equal_cliff_change_pct",
                "fixed_equal_pair_change_pct",
            ]
        },
        "action": (
            "preregister_full_87_run_matrix"
            if matrix_go
            else "stop_without_test_evaluation_or_feature_tuning"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    table.to_csv(args.output_table, index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
