#!/usr/bin/env python3
"""Freeze exploratory low-weight PrismNet pair-expert ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids
from scripts.build_rf_gine_scale_calibration import (
    gine_validation,
    read_series,
    validation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def weight_name(weight: float) -> str:
    return f"prism_weight_{weight:.2f}".replace(".", "p")


def select_weight(frame: pd.DataFrame, overall_tolerance: float, cliff_tolerance: float) -> float:
    identity = frame.loc[frame["prism_weight"] == 0.0].iloc[0]
    feasible = frame[
        (frame["validation_overall_mae"] <= (1.0 + overall_tolerance) * identity["validation_overall_mae"])
        & (frame["validation_cliff_mae"] <= (1.0 + cliff_tolerance) * identity["validation_cliff_mae"])
        & (frame["validation_pair_delta_mae"] <= identity["validation_pair_delta_mae"])
    ].copy()
    if feasible.empty:
        raise RuntimeError("Identity candidate was unexpectedly infeasible")
    feasible = feasible.sort_values(
        ["validation_pair_delta_mae", "validation_cliff_mae", "validation_overall_mae", "prism_weight"],
        kind="stable",
    )
    return float(feasible.iloc[0]["prism_weight"])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    components = config["components"]
    weights = [float(value) for value in config["selection"]["prism_weight_candidates"]]
    if 0.0 not in weights:
        raise RuntimeError("Identity PrismNet weight is required")
    tasks = []
    for root_text in components["prismnet_run_roots"]:
        for path in sorted(Path(root_text).glob("*/fold_*/seed_*/predictions.parquet")):
            tasks.append((path.parts[-4], int(path.parts[-3].split("_")[-1]), int(path.parts[-2].split("_")[-1]), path))
    tasks = sorted(tasks)
    if len(tasks) != int(config["evaluation"]["runs"]):
        raise RuntimeError(f"Expected three frozen PrismNet runs, observed {len(tasks)}")
    if args.smoke:
        tasks = tasks[:1]

    records = []
    for target, fold, seed, prism_test_path in tasks:
        prism_test, prism_test_hash = read_series(prism_test_path)
        prism_valid, prism_valid_hash = read_series(prism_test_path.parent / "validation_predictions.parquet")
        fixed_test, fixed_test_hash = read_series(
            Path(components["fixed_equal_ensemble"]) / target / f"fold_{fold}" / f"seed_{seed}" / "fixed_equal" / "predictions.parquet"
        )
        rf_valid, rf_valid_hash = read_series(
            Path(components["rf"]) / target / f"fold_{fold}" / f"seed_{seed}" / "validation_predictions.parquet"
        )
        gine_valid, gine_valid_hash = gine_validation(
            Path(components["gine_validation_source"]) / target / f"fold_{fold}" / "validation_predictions.parquet", seed
        )
        if not fixed_test.index.equals(prism_test.index):
            raise RuntimeError("PrismNet/fixed-equal test molecule mismatch")
        if not rf_valid.index.equals(gine_valid.index) or not rf_valid.index.equals(prism_valid.index):
            raise RuntimeError("PrismNet/RF/GINE validation molecule mismatch")
        fixed_valid = 0.5 * rf_valid + 0.5 * gine_valid
        truth = read_labels_for_ids(args.data_root, target, fixed_valid.index).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[mmp["mol_id_lo"].isin(truth.index) & mmp["mol_id_hi"].isin(truth.index)].copy()
        internal["delta_y"] = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
        cliffs = internal[internal["delta_y"].abs() >= 1.0]
        rows = []
        for weight in weights:
            prediction = (1.0 - weight) * fixed_valid + weight * prism_valid
            overall, cliff, pair = validation_metrics(prediction, truth, cliffs)
            rows.append({"prism_weight": weight, "validation_overall_mae": overall,
                         "validation_cliff_mae": cliff, "validation_pair_delta_mae": pair})
        selection = pd.DataFrame(rows)
        selected = select_weight(
            selection,
            float(config["selection"]["maximum_validation_overall_relative_change_vs_identity"]),
            float(config["selection"]["maximum_validation_cliff_relative_change_vs_identity"]),
        )
        output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        selection.to_csv(output / "weight_selection.csv", index=False)
        metadata = {
            "target": target, "fold": fold, "seed": seed, "method": config["method"],
            "selected_prism_weight": selected, "validation_labels_materialized": True,
            "outer_test_labels_materialized": False, "exploratory_after_component_test_metrics_seen": True,
            "component_hashes_verified": True, "prism_test_hash": prism_test_hash,
            "prism_validation_hash": prism_valid_hash, "fixed_equal_test_hash": fixed_test_hash,
            "rf_validation_hash": rf_valid_hash, "gine_validation_hash": gine_valid_hash,
            "smoke": args.smoke,
        }
        for label, weight in [*((weight_name(x), x) for x in weights), ("validation_selected", selected)]:
            prediction = (1.0 - weight) * fixed_test + weight * prism_test
            digest = freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                output / label / "predictions.parquet",
                {**metadata, "output": label, "prism_weight": weight},
            )
            metadata[f"{label}_prediction_sha256"] = digest
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "smoke": args.smoke,
                      "selected_weights": pd.Series([x["selected_prism_weight"] for x in records]).value_counts().sort_index().to_dict(),
                      "outer_test_labels_materialized": False}, indent=2))


if __name__ == "__main__":
    main()

