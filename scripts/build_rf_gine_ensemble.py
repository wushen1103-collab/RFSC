#!/usr/bin/env python3
"""Select RF/GINE weights on validation and freeze label-private test ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, default=Path("graphs/mmp"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/moleculeace"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def verify(path: Path) -> None:
    manifest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Frozen component hash mismatch: {path}")


def read_series(path: Path) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Unexpected component prediction schema: {path}")
    return frame.set_index("mol_id")["prediction"].sort_index()


def gine_validation(path: Path, seed: int) -> pd.Series:
    verify(path)
    frame = pd.read_parquet(path)
    frame = frame[(frame["method"] == "base") & (frame["seed"] == seed)]
    if frame["mol_id"].duplicated().any():
        raise RuntimeError("Duplicate GINE validation prediction")
    return frame.set_index("mol_id")["prediction"].sort_index()


def validation_metrics(prediction: pd.Series, truth: pd.Series, cliffs: pd.DataFrame) -> tuple[float, float]:
    endpoints = set(cliffs["mol_id_lo"]) | set(cliffs["mol_id_hi"])
    endpoint_ids = [mol_id for mol_id in prediction.index if mol_id in endpoints]
    if not endpoint_ids:
        raise RuntimeError("Validation fold contains no cliff endpoints")
    overall = float(np.mean(np.abs(prediction - truth.loc[prediction.index])))
    cliff = float(np.mean(np.abs(prediction.loc[endpoint_ids] - truth.loc[endpoint_ids])))
    return overall, cliff


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rf_root = Path(config["components"]["rf"])
    gine_root = Path(config["components"]["gine"])
    gine_validation_root = Path(config["components"]["gine_validation_source"])
    candidates = [float(value) for value in config["selection"]["rf_weight_candidates"]]
    alpha = float(config["selection"]["overall_alpha"])
    equal_weight = float(config["controls"]["fixed_equal_rf_weight"])
    tasks = []
    for rf_test_path in sorted(rf_root.glob("*/fold_*/seed_*/predictions.parquet")):
        target = rf_test_path.parts[-4]
        fold = int(rf_test_path.parts[-3].split("_")[-1])
        seed = int(rf_test_path.parts[-2].split("_")[-1])
        tasks.append((target, fold, seed, rf_test_path))
    if len(tasks) != 87:
        raise RuntimeError(f"Expected 87 RF component runs, observed {len(tasks)}")
    if args.smoke:
        tasks = tasks[:1]
    records = []
    for target, fold, seed, rf_test_path in tasks:
        rf_run = rf_test_path.parent
        rf_validation_path = rf_run / "validation_predictions.parquet"
        gine_test_path = gine_root / target / f"fold_{fold}" / f"seed_{seed}" / "predictions.parquet"
        gine_validation_path = gine_validation_root / target / f"fold_{fold}" / "validation_predictions.parquet"
        rf_validation = read_series(rf_validation_path)
        gine_valid = gine_validation(gine_validation_path, seed)
        rf_test = read_series(rf_test_path)
        gine_test = read_series(gine_test_path)
        if not rf_validation.index.equals(gine_valid.index) or not rf_test.index.equals(gine_test.index):
            raise RuntimeError(f"RF/GINE molecule mismatch: {target} fold {fold} seed {seed}")
        truth = read_labels_for_ids(args.data_root, target, rf_validation.index).set_index("mol_id")["y_pactivity"]
        mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        internal = mmp[mmp["mol_id_lo"].isin(truth.index) & mmp["mol_id_hi"].isin(truth.index)].copy()
        internal["delta_y"] = internal["mol_id_hi"].map(truth) - internal["mol_id_lo"].map(truth)
        cliffs = internal[internal["delta_y"].abs() >= 1.0]
        selection = []
        for weight in candidates:
            prediction = weight * rf_validation + (1.0 - weight) * gine_valid
            overall, cliff = validation_metrics(prediction, truth, cliffs)
            selection.append({"rf_weight": weight, "validation_overall_mae": overall, "validation_cliff_mae": cliff, "selection_objective": cliff + alpha * overall})
        selection_frame = pd.DataFrame(selection)
        selection_frame["equal_distance"] = (selection_frame["rf_weight"] - 0.5).abs()
        selection_frame = selection_frame.sort_values(["selection_objective", "equal_distance", "rf_weight"], kind="stable")
        selected_weight = float(selection_frame.iloc[0]["rf_weight"])
        output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        selection_frame.drop(columns="equal_distance").to_csv(output / "weight_selection.csv", index=False)
        metadata = {
            "target": target,
            "fold": fold,
            "seed": seed,
            "method": config["method"],
            "validation_labels_materialized": True,
            "outer_test_labels_materialized": False,
            "selected_rf_weight": selected_weight,
            "fixed_equal_rf_weight": equal_weight,
            "component_prediction_hashes_verified": True,
            "smoke": args.smoke,
        }
        for method, weight in [("validation_selected", selected_weight), ("fixed_equal", equal_weight)]:
            prediction = weight * rf_test + (1.0 - weight) * gine_test
            path = output / method / "predictions.parquet"
            digest = freeze_prediction_file(
                pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                path,
                {**metadata, "ensemble_method": method, "rf_weight": weight},
            )
            metadata[f"{method}_prediction_sha256"] = digest
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    pd.DataFrame(records).to_csv(args.output_dir / "build_summary.csv", index=False)
    print(json.dumps({"runs": len(records), "smoke": args.smoke, "weights": pd.Series([row["selected_rf_weight"] for row in records]).value_counts().sort_index().to_dict()}, indent=2))


if __name__ == "__main__":
    main()

